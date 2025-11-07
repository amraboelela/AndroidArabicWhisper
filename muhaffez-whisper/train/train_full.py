#!/usr/bin/env python3
"""
Universal training script for encoder-decoder model
Usage: python3 train_full.py <dataset_name> <surah_part>
Examples:
  python3 train_full.py Quran-A 001       # Train on Al-Fatiha (001)
  python3 train_full.py Quran-A 002-01    # Train on Al-Baqara part 1
  python3 train_full.py Quran-A 002-04    # Train on Al-Baqara part 4
"""
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Force unbuffered output for real-time logging
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import json
import torch
import torch.nn as nn
import torchaudio
import glob
import os
import random
import time
sys.path.append("..")
from custom_scripts.encoder_decoder_transformer import EncoderDecoderTransformer

# ==============================================================
# Device setup
# ==============================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# ==============================================================
# Audio feature extraction
# ==============================================================
def extract_mel_features(audio_path, n_mels=80, target_seconds=None):
    """Extract mel features from audio, optionally trimming to target_seconds"""
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert stereo to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16kHz (Whisper standard)
    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
        sample_rate = target_sample_rate

    # Trim to target seconds if specified
    if target_seconds is not None:
        num_samples = int(sample_rate * target_seconds)
        if waveform.shape[1] > num_samples:
            waveform = waveform[:, :num_samples]

    # Whisper parameters (100 fps: 16000 / 160 = 100)
    n_fft = 400
    hop_length = 160

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=0,
        f_max=sample_rate // 2
    )
    mel_spec = mel_transform(waveform)
    mel_spec = torch.log(mel_spec + 1e-9)
    mel_features = mel_spec.squeeze(0).transpose(0, 1)

    # Global Whisper normalization
    # These are the standard Whisper mel spectrogram statistics
    mel_mean = -4.2677
    mel_std = 4.5689
    mel_features = (mel_features - mel_mean) / (mel_std + 1e-8)

    return mel_features, sample_rate

# ==============================================================
# Tokenization
# ==============================================================
def tokenize_text(text, vocab):
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    words = text.split()
    return [word_to_idx.get(word, 0) for word in words]  # 0 = unknown

# ==============================================================
# Comprehensive accuracy calculation for all segments
# ==============================================================
def calculate_comprehensive_accuracy(model, segment_files, transcriptions, vocab, target_seconds, target_words, device, debug=False):
    """Calculate accuracy across all segments"""
    model.eval()
    total_correct = 0
    total_expected = 0
    segment_accuracies = []

    with torch.no_grad():
        for idx, (seg_file, transcription) in enumerate(zip(segment_files, transcriptions)):
            # Extract audio features
            audio_features, sample_rate = extract_mel_features(seg_file, target_seconds=target_seconds)
            audio_batch = audio_features.transpose(0, 1).unsqueeze(0).to(device)

            # Get expected text
            expected_words = transcription.split()[:target_words] if target_words else transcription.split()
            expected_text = " ".join(expected_words)

            if not expected_text:
                continue

            # Generate with timeout protection
            # For large datasets, reduce max_tokens to avoid hanging
            if len(segment_files) > 20:
                max_tokens = min((target_words * 5) if target_words else 30, 50)
            else:
                max_tokens = min((target_words * 10) if target_words else 50, 100)

            waveform, sr = torchaudio.load(seg_file)
            audio_duration = target_seconds if target_seconds else (waveform.shape[1] / sr)

            try:
                generated = model.generate(audio_batch, max_new_tokens=max_tokens, audio_duration_seconds=audio_duration, use_sampling=False)
                generated_ids = generated[0].tolist()
            except Exception as e:
                print(f"    Warning: Generation failed for segment {idx}: {e}", flush=True)
                continue

            # Clean up generated IDs
            if generated_ids and generated_ids[0] == 1:
                generated_ids = generated_ids[1:]
            if 2 in generated_ids:
                generated_ids = generated_ids[:generated_ids.index(2)]

            generated_words = [vocab[idx] for idx in generated_ids if idx < len(vocab)]

            # Calculate confidence and filter low confidence words
            if len(generated_ids) > 0:
                encoder_output = model.encode(audio_batch)
                text_ids = torch.tensor([[1] + generated_ids[:len(generated_words)]], dtype=torch.long, device=device)
                logits = model.decode(text_ids, encoder_output)
                probs = torch.softmax(logits, dim=-1)

                # Get confident words only (>= 20% threshold)
                confident_words = []
                for i, token_id in enumerate(generated_ids[:len(generated_words)]):
                    if i < logits.shape[1] - 1:
                        token_prob = probs[0, i, token_id].item()
                        if token_prob >= 0.2:  # 20% threshold
                            confident_words.append(generated_words[i])

                # Count correct confident words
                correct = sum(1 for i, word in enumerate(confident_words) if i < len(expected_words) and word == expected_words[i])
            else:
                correct = 0

            # Calculate segment accuracy
            segment_acc = (correct / len(expected_words) * 100) if expected_words else 0
            segment_accuracies.append(segment_acc)
            total_correct += correct
            total_expected += len(expected_words)

            if debug:
                print(f"  Seg {idx}: expected={len(expected_words)} words, correct={correct}, acc={segment_acc:.1f}%")

    # Calculate overall accuracy
    overall_accuracy = (total_correct / total_expected * 100) if total_expected > 0 else 0
    avg_segment_accuracy = sum(segment_accuracies) / len(segment_accuracies) if segment_accuracies else 0

    return overall_accuracy, avg_segment_accuracy, segment_accuracies

# ==============================================================
# Training
# ==============================================================
def train_model(model, segment_files, transcriptions, vocab, surah_part,
                target_seconds=None, target_words=None, num_epochs=500, learning_rate=1e-3):
    """
    Universal training function

    Args:
        target_seconds: Number of seconds to use from audio (None = full audio)
        target_words: Number of words to predict (None = all words)
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

    print(f"Initial Learning Rate: {learning_rate:.1e}")

    best_loss = float('inf')
    best_epoch = -1
    prev_loss = float('inf')
    start_time = time.time()
    checkpoint_time = start_time  # Time of last checkpoint (for relative timing)

    # Calculate initial accuracy before training
    model.eval()
    overall_acc, avg_acc, seg_accuracies = calculate_comprehensive_accuracy(
        model, segment_files, transcriptions, vocab,
        target_seconds, target_words, device
    )
    print(f"Initial accuracy: {overall_acc:.1f}%\n")

    # If already perfect, no need to train
    if overall_acc > 90.0:
        print(f"✓ Model already at {overall_acc:.1f}% accuracy. Skipping training.\n")
        return model

    # Build description
    audio_desc = f"first {target_seconds}s" if target_seconds else "full"
    text_desc = f"first {target_words} words" if target_words else "full"
    checkpoint_name = "checkpoint_best.pt"

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_iterations = 0
        indices = list(range(len(segment_files)))
        random.shuffle(indices)

        for i in indices:
            seg_file = segment_files[i]
            text = transcriptions[i]

            # Extract audio features
            audio_features, sample_rate = extract_mel_features(seg_file, target_seconds=target_seconds)

            # Extract target text
            if target_words:
                words = text.split()
                target_text = " ".join(words[:target_words]) if len(words) >= target_words else text
            else:
                target_text = text

            if not target_text:
                continue

            text_tokens = tokenize_text(target_text, vocab)
            audio_batch = audio_features.transpose(0, 1).unsqueeze(0).to(device)

            full_sequence = [1] + text_tokens + [2]  # <s> + tokens + </s>
            input_ids = torch.tensor([full_sequence[:-1]], dtype=torch.long, device=device)
            labels = torch.tensor([full_sequence[1:]], dtype=torch.long, device=device)

            logits = model(mel_features=audio_batch, text_ids=input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_iterations += 1

        avg_loss = total_loss / total_iterations

        # Dynamic learning rate: reduce by 10% if loss increases
        if avg_loss > prev_loss:
            current_lr = optimizer.param_groups[0]['lr']
            new_lr = max(current_lr * 0.9, 1e-7)  # Minimum LR = 1e-7
            if new_lr != current_lr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                # Only print if LR actually changed
                if new_lr > 1e-7:
                    # Format LR without trailing zero (e.g., 9e-04 instead of 9.0e-04)
                    lr_str = f"{new_lr:.0e}" if new_lr >= 1e-6 else f"{new_lr:.1e}"
                    print(f"  Loss increased ({prev_loss:.4f} → {avg_loss:.4f}), reducing LR to: {lr_str}", flush=True)

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch
            }, checkpoint_name)

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        # Format elapsed time: seconds if < 60s, minutes if >= 60s
        elapsed_from_checkpoint = time.time() - checkpoint_time
        if elapsed_from_checkpoint >= 60:
            time_str = f"{int(round(elapsed_from_checkpoint / 60))}m"
        else:
            time_str = f"{int(round(elapsed_from_checkpoint))}s"

        # Calculate accuracy after epoch 1 and every 10 epochs for display and early stopping
        accuracy_str = ""
        if epoch == 0 or (epoch + 1) % 10 == 0:
            # Save current model state
            current_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            # Load best checkpoint for accuracy evaluation
            if os.path.exists(checkpoint_name):
                checkpoint = torch.load(checkpoint_name, map_location=device)
                model.load_state_dict(checkpoint["model"])

            # Calculate accuracy on best model
            overall_acc, avg_acc, seg_accuracies = calculate_comprehensive_accuracy(
                model, segment_files, transcriptions, vocab,
                target_seconds, target_words, device
            )

            # Build accuracy string for display
            accuracy_str = f" | Accuracy={overall_acc:.0f}%"

            # Early stopping if accuracy > 90%
            if overall_acc > 90.0:
                # Print current epoch info with accuracy before stopping
                print(f"Epoch {epoch+1}/{num_epochs} | Loss={avg_loss:.4f}{accuracy_str} | Time={time_str}")
                print(f"✓ Early stopping: accuracy {overall_acc:.1f}% at epoch {best_epoch}", flush=True)
                # Keep best model loaded, don't restore
                break

            # Restore current model to continue training
            model.load_state_dict({k: v.to(device) for k, v in current_model_state.items()})

        # Print epoch 1, every 10 epochs, or on last epoch
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1}/{num_epochs} | Loss={avg_loss:.4f}{accuracy_str} | Time={time_str}")
            # Reset checkpoint time after printing
            checkpoint_time = time.time()

        prev_loss = avg_loss

    total_time = time.time() - start_time
    # Format total time: seconds if < 60s, minutes if >= 60s
    if total_time >= 60:
        print(f"Training complete in {int(round(total_time / 60))}m")
    else:
        print(f"Training complete in {int(round(total_time))}s")

    # Load best checkpoint
    if os.path.exists(checkpoint_name):
        print(f"\n✓ Loading best checkpoint from {checkpoint_name}...")
        checkpoint = torch.load(checkpoint_name, map_location=device)
        model.load_state_dict(checkpoint["model"])
        print(f"✓ Loaded best model from epoch {checkpoint['epoch'] + 1}")

    # Calculate comprehensive accuracy on all segments
    model.eval()
    overall_acc, avg_acc, seg_accuracies = calculate_comprehensive_accuracy(
        model, segment_files, transcriptions, vocab,
        target_seconds, target_words, device
    )
    print(f"Accuracy: {overall_acc:.0f}%\n")

    # Sample generation for visualization (first segment)
    test_audio, sample_rate = extract_mel_features(segment_files[0], target_seconds=target_seconds)
    test_audio = test_audio.transpose(0, 1).unsqueeze(0).to(device)

    # Get expected text
    if target_words:
        words = transcriptions[0].split()
        expected_text = " ".join(words[:target_words]) if len(words) >= target_words else transcriptions[0]
    else:
        expected_text = transcriptions[0]

    # Calculate audio duration for generation
    waveform, sr = torchaudio.load(segment_files[0])
    audio_duration = target_seconds if target_seconds else (waveform.shape[1] / sr)

    with torch.no_grad():
        max_tokens = (target_words * 10) if target_words else 50
        generated = model.generate(test_audio, max_new_tokens=max_tokens, audio_duration_seconds=audio_duration, use_sampling=False)
        generated_ids = generated[0].tolist()
        if generated_ids and generated_ids[0] == 1:
            generated_ids = generated_ids[1:]
        if 2 in generated_ids:
            generated_ids = generated_ids[:generated_ids.index(2)]
        generated_words = [vocab[idx] for idx in generated_ids if idx < len(vocab)]

        # Show only target words if specified
        if target_words:
            display_words = generated_words[:target_words] if len(generated_words) >= target_words else generated_words
            num_tokens_to_check = target_words
        else:
            display_words = generated_words
            num_tokens_to_check = len(generated_words)

        # Calculate confidence for each token
        encoder_output = model.encode(test_audio)
        text_ids = torch.tensor([[1] + generated_ids[:num_tokens_to_check]], dtype=torch.long, device=device)
        logits = model.decode(text_ids, encoder_output)
        probs = torch.softmax(logits, dim=-1)

        # Get probability of each generated token
        token_confidences = []
        for i, token_id in enumerate(generated_ids[:num_tokens_to_check]):
            if i < logits.shape[1] - 1:  # -1 because we prepended <s>
                token_prob = probs[0, i, token_id].item()
                token_confidences.append(token_prob)

        # Calculate accuracy (percentage of correct words)
        expected_words = expected_text.split()
        correct_words = sum(1 for i, word in enumerate(display_words) if i < len(expected_words) and word == expected_words[i])
        accuracy = (correct_words / len(expected_words) * 100) if expected_words else 0

        # Filter out words with 0% confidence and calculate accuracy based on confident words
        filtered_words = []
        filtered_confidences = []
        correct_confident_words = 0
        total_confident_words = 0

        if len(display_words) == len(token_confidences):
            for i, (word, conf) in enumerate(zip(display_words, token_confidences)):
                if conf < 0.01:  # Hide words with < 1% confidence (rounds to 0%)
                    continue
                elif conf >= 0.2:  # 20% threshold
                    filtered_words.append(word)
                    filtered_confidences.append(f'{conf:.0%}')
                    total_confident_words += 1
                    if i < len(expected_words) and word == expected_words[i]:
                        correct_confident_words += 1
                else:
                    filtered_words.append(f"[{word}]")  # Mark low confidence with brackets
                    filtered_confidences.append(f'{conf:.0%}')

            # Accuracy: correct confident words out of total EXPECTED words
            accuracy = (correct_confident_words / len(expected_words) * 100) if expected_words else 0
        else:
            filtered_words = display_words
            filtered_confidences = [f'{c:.0%}' for c in token_confidences] if token_confidences else []
            # Original accuracy calculation if confidences don't match
            accuracy = (correct_words / len(expected_words) * 100) if expected_words else 0

        # Build confidence text
        if filtered_confidences:
            confidence_text = ', '.join(filtered_confidences)
        else:
            confidence_text = "N/A"

        display_text = ' '.join(filtered_words) if filtered_words else ""

        print(f"🔸 Expected: {expected_text}")
        print(f"🔹 Generated: {display_text}")
        print(f"   Confidence: {confidence_text}")

    return model

# ==============================================================
# Replay Buffer - Prevent Catastrophic Forgetting
# ==============================================================
def collect_replay_samples(dataset_name, current_surah_part, datasets_dir, current_set_size):
    """
    Collect a small sample from all previously trained surahs to prevent catastrophic forgetting.

    Args:
        dataset_name: Name of dataset (e.g., "Quran-A")
        current_surah_part: Current surah being trained (e.g., "002-01")
        datasets_dir: Path to datasets directory
        current_set_size: Size of current training set (to calculate 10% replay buffer with minimum 20)

    Returns:
        (replay_segment_files, replay_transcriptions): Lists of replay samples
    """
    current_surah_num = current_surah_part.split('-')[0]

    replay_segment_files = []
    replay_transcriptions = []

    # Find all text files for previous surah parts (including previous parts from same surah)
    text_dir = f"../datasets/{dataset_name}/text"
    all_text_files = sorted(glob.glob(os.path.join(text_dir, "*.txt")))

    # Count previous surah parts and total available samples
    # Only include parts that are < current_surah_part (not <=, we don't replay current)
    previous_surah_parts = []
    total_previous_samples = 0
    for text_file in all_text_files:
        basename = os.path.basename(text_file)
        surah_part = basename.replace('.txt', '')

        # Compare surah parts properly: "002-01" < "002-02" is TRUE
        if surah_part < current_surah_part:
            # Count samples in this surah part
            with open(text_file, "r", encoding="utf-8") as f:
                num_samples = len([line for line in f if line.strip()])
            previous_surah_parts.append((text_file, surah_part))
            total_previous_samples += num_samples

    if not previous_surah_parts:
        return replay_segment_files, replay_transcriptions

    # Calculate total replay buffer size as min(max(10% of current set, 20), total previous samples)
    total_replay_size = min(max(int(current_set_size * 0.1), 20), total_previous_samples)

    # Distribute replay budget evenly across previous surahs
    samples_per_surah = max(1, total_replay_size // len(previous_surah_parts))

    for text_file, surah_part in previous_surah_parts:
        # Load transcriptions
        with open(text_file, "r", encoding="utf-8") as f:
            transcriptions = [line.strip() for line in f if line.strip()]

        # Extract surah number for audio path
        surah_num = surah_part.split('-')[0]

        # Load corresponding audio segments
        segment_files = sorted(glob.glob(os.path.join(datasets_dir, surah_num, f"{surah_part}-*.wav")))

        if len(segment_files) > 0 and len(segment_files) == len(transcriptions):
            # Sample up to samples_per_surah, but not more than available
            num_samples = min(samples_per_surah, len(segment_files))

            # Randomly sample indices
            import random
            indices = random.sample(range(len(segment_files)), num_samples)

            for idx in indices:
                replay_segment_files.append(segment_files[idx])
                replay_transcriptions.append(transcriptions[idx])

    if len(replay_segment_files) > 0:
        # Extract segment names and format as comma-separated list
        segment_names = [os.path.basename(f).replace('.wav', '') for f in replay_segment_files]
        segments_str = ', '.join(segment_names)
        print(f"  Replay buffer segments: ({segments_str})")
        print(f"  Replay buffer size: {len(replay_segment_files)}\n")

    return replay_segment_files, replay_transcriptions

# ==============================================================
# Main
# ==============================================================
def main():
    if len(sys.argv) < 3:
        print("Usage: python3 train_full.py <dataset_name> <surah_part>")
        print("Examples:")
        print("  python3 train_full.py Quran-A 001")
        print("  python3 train_full.py Quran-A 002-01")
        print("  python3 train_full.py Quran-A 002-04")
        sys.exit(1)

    dataset_name = sys.argv[1]  # e.g., "Quran-A"
    surah_part = sys.argv[2]  # e.g., "001", "002-01", "002-04"

    datasets_dir = f"../datasets/{dataset_name}/audio"
    vocab_path = "../models/vocabulary.json"
    model_path = "../models/muhaffez_whisper.pt"

    # Load vocab
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # Parse surah part name to determine surah number
    surah_num = surah_part.split('-')[0]  # "001" or "002"

    # Load transcriptions and segments
    text_path = f"../datasets/{dataset_name}/text/{surah_part}.txt"
    if not os.path.exists(text_path):
        print(f"❌ Error: Text file not found: {text_path}")
        sys.exit(1)

    with open(text_path, "r", encoding="utf-8") as f:
        transcriptions = [line.strip() for line in f if line.strip()]

    segment_files = sorted(glob.glob(os.path.join(datasets_dir, surah_num, f"{surah_part}-*.wav")))

    if not segment_files:
        print(f"❌ Error: No audio segments found in {datasets_dir}/{surah_num}/{surah_part}-*.wav")
        sys.exit(1)

    print(f"Loaded {len(transcriptions)} transcriptions, {len(segment_files)} audio segments")

    if len(transcriptions) != len(segment_files):
        print(f"⚠️  Warning: Mismatch between transcriptions ({len(transcriptions)}) and segments ({len(segment_files)})")

    # Determine training configuration based on surah part name
    # For Al-Fatiha (001) or full Baqara segments, train on full audio/text
    # For specific parts, can adjust target_seconds and target_words as needed
    target_seconds = None  # Full audio by default
    target_words = None    # Full transcription by default

    # You can customize this based on surah part patterns if needed
    # For example:
    # if surah_part == "002-04":
    #     target_seconds = 4.0
    #     target_words = 3

    print(f"\n✓ Training on surah part: {surah_part}")
    print(f"   Audio: {'full' if not target_seconds else f'first {target_seconds}s'}")
    print(f"   Text: {'full' if not target_words else f'first {target_words} words'}")

    # Collect replay buffer samples from previous surahs to prevent catastrophic forgetting
    # Replay buffer size = 10% of current training set
    replay_segment_files, replay_transcriptions = collect_replay_samples(
        dataset_name, surah_part, datasets_dir, len(segment_files)
    )

    # Combine current surah samples with replay buffer
    all_segment_files = segment_files + replay_segment_files
    all_transcriptions = transcriptions + replay_transcriptions

    print(f"   Training samples: {len(segment_files)} current + {len(replay_segment_files)} replay = {len(all_segment_files)} total\n")

    # Create model
    model = EncoderDecoderTransformer(
        vocab_size=len(vocab),
        d_model=128,
        n_encoder_layers=4,
        n_decoder_layers=4,
        n_heads=4,
        d_ff=512,
        dropout=0.1
    )

    # Load existing model and continue training
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✓ Model loaded successfully! Continuing training on {surah_part}.")
    else:
        print(f"No existing model found. Starting with fresh weights for {surah_part} training.")

    # Train
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{timestamp}] Starting training for up to 500 epochs on {len(all_segment_files)} segments...\n")
    model = train_model(
        model,
        all_segment_files,
        all_transcriptions,
        vocab,
        surah_part,
        target_seconds=target_seconds,
        target_words=target_words,
        num_epochs=500,
        learning_rate=1e-3
    )

    # Save final model
    torch.save(model.state_dict(), model_path)
    print(f"Final model saved to: {model_path}")

    # Calculate and output final accuracy for train.sh to capture
    model.eval()
    overall_acc, avg_acc, seg_accuracies = calculate_comprehensive_accuracy(
        model, segment_files, transcriptions, vocab,
        target_seconds=None, target_words=None, device=device
    )
    print(f"FINAL_ACCURACY: {overall_acc:.0f}%")


if __name__ == "__main__":
    main()
