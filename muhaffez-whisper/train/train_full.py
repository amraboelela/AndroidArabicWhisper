#!/usr/bin/env python3
"""
Universal training script for encoder-decoder model
Usage: python3 train_full.py <dataset_name> <surah_part>
Examples:
  python3 train_full.py Quran-A 001       # Train on Al-Fatiha (001)
  python3 train_full.py Quran-A 002-01    # Train on Al-Baqara part 1
  python3 train_full.py Quran-A 002-04    # Train on Al-Baqara part 4
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import json
import torch
import torch.nn as nn
import torchaudio
import glob
import os
import random
import time
import sys
sys.path.append("..")
from custom_scripts.encoder_decoder_transformer import EncoderDecoderTransformer

# ==============================================================
# Device setup
# ==============================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 Using Metal GPU (Apple Silicon)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("🚀 Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("⚠️  Using CPU (slower)")

print(f"Device: {device}")

# ==============================================================
# Audio feature extraction
# ==============================================================
def extract_mel_features(audio_path, n_mels=80, target_seconds=None):
    """Extract mel features from audio, optionally trimming to target_seconds"""
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert stereo to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

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

    # Per-segment normalization
    mel_mean = mel_features.mean()
    mel_std = mel_features.std()
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
# Training
# ==============================================================
def train_model(model, segment_files, transcriptions, vocab, surah_part,
                target_seconds=None, target_words=None, num_epochs=100, learning_rate=1e-5):
    """
    Universal training function

    Args:
        target_seconds: Number of seconds to use from audio (None = full audio)
        target_words: Number of words to predict (None = all words)
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

    best_loss = float('inf')
    prev_loss = float('inf')
    prev_checkpoint_loss = float('inf')  # Track loss at every 10th epoch
    patience_counter = 0  # Track consecutive epochs with low loss change
    min_delta = 1e-3  # Minimum change to consider as improvement
    patience = 3  # Number of epochs to wait before stopping
    start_time = time.time()

    # Build description
    audio_desc = f"first {target_seconds}s" if target_seconds else "full"
    text_desc = f"first {target_words} words" if target_words else "full"
    checkpoint_name = f"checkpoint_best_{surah_part}.pt"

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

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch
            }, checkpoint_name)
            best_marker = " ⭐ NEW BEST!"
        else:
            best_marker = ""

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        # Print every 10 epochs or on last epoch
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1}/{num_epochs} | Loss={avg_loss:.4f} | LR={current_lr:.6f} | Time={elapsed:.1f}s{best_marker}")

        # Check if loss increased at checkpoint epochs (every 10 epochs)
        if (epoch + 1) % 10 == 0:
            if prev_checkpoint_loss != float('inf') and avg_loss > prev_checkpoint_loss:
                print(f"⚠️  Early stopping: loss increased from {prev_checkpoint_loss:.4f} to {avg_loss:.4f}")
                break
            prev_checkpoint_loss = avg_loss

        # Early stopping check
        loss_change = prev_loss - avg_loss
        if loss_change < min_delta:  # Includes small improvement, no change, or getting worse
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⚠️  Early stopping: loss not improving for {patience} consecutive epochs")
                break
        else:
            patience_counter = 0  # Reset counter if loss change is significant
        # Reduce learning rate if loss increased
        if prev_loss != float('inf') and avg_loss > prev_loss:
            for param_group in optimizer.param_groups:
                old_lr = param_group['lr']
                new_lr = max(old_lr * 0.9, 1e-7)  # Reduce by 10%, but not below 1e-7
                param_group['lr'] = new_lr
                if new_lr != old_lr:
                    print(f"⚠️  Loss increased: reducing LR from {old_lr:.6f} to {new_lr:.6f}")

        prev_loss = avg_loss

    total_time = time.time() - start_time
    print(f"Training complete in {total_time:.1f}s | Best loss: {best_loss:.4f}")

    # Load best checkpoint
    if os.path.exists(checkpoint_name):
        print(f"\n✓ Loading best checkpoint from {checkpoint_name}...")
        checkpoint = torch.load(checkpoint_name, map_location=device)
        model.load_state_dict(checkpoint["model"])
        print(f"✓ Loaded best model from epoch {checkpoint['epoch'] + 1}")

    # Sample generation at the end
    model.eval()
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
        generated = model.generate(test_audio, max_new_tokens=max_tokens, audio_duration_seconds=audio_duration)
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
                elif conf >= 0.3:  # 30% threshold
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
        print(f"   Accuracy: {accuracy:.0f}%")

    return model

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

    # Load existing model and continue training (backup created once per surah, not per part)
    import shutil
    import time
    surah_num = surah_part.split('-')[0]  # Get surah number (e.g., "001" or "002")
    day_num = time.strftime("%u")  # Day of week (1=Monday, 7=Sunday)
    backup_path = model_path.replace(".pt", f"_backup_{surah_num}.pt")
    day_backup_path = model_path.replace(".pt", f"_backup_{surah_num}_{day_num}.pt")

    if os.path.exists(model_path):
        # Only create backup if it doesn't exist yet for this day (once per surah per day)
        if not os.path.exists(backup_path) or not os.path.exists(day_backup_path):
            # Move existing backup to day-specific backup
            if os.path.exists(backup_path):
                shutil.move(backup_path, day_backup_path)
            # Create new backup
            shutil.copy2(model_path, backup_path)
            print(f"✓ Backup created: {backup_path}")

        print(f"Loading existing model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✓ Model loaded successfully! Continuing training on {surah_part}.")
    else:
        print(f"No existing model found. Starting with fresh weights for {surah_part} training.")

    # Train
    print(f"\nStarting training for up to 100 epochs on {len(segment_files)} segments...\n")
    model = train_model(
        model,
        segment_files,
        transcriptions,
        vocab,
        surah_part,
        target_seconds=target_seconds,
        target_words=target_words,
        num_epochs=100,
        learning_rate=1e-5
    )

    # Save final model
    torch.save(model.state_dict(), model_path)
    print(f"Final model saved to: {model_path}")


if __name__ == "__main__":
    main()
