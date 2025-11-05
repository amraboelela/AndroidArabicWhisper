#!/usr/bin/env python3
"""
Curriculum training script - trains incrementally on chunks
Usage: python3 train_curriculum.py <segment> [dataset_name]

Examples:
  python3 train_curriculum.py 002-04 base
  python3 train_curriculum.py 001 base

This script trains the model using curriculum learning:
- Stage 1: First 1.3s → first 1 word
- Stage 2: First 2.6s → first 2 words
- Stage 3: First 3.9s → first 3 words
- ... and so on until full audio → full transcription
"""
import json
import torch
import torch.nn as nn
import torchaudio
import glob
import os
import random
import time
import sys
sys.path.append("../..")
from encoder_decoder_transformer import EncoderDecoderTransformer

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
# Configuration
# ==============================================================
CHUNK_DURATION = 1.3  # seconds per chunk
WORDS_PER_CHUNK = 1   # words per chunk

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
# Calculate curriculum stages
# ==============================================================
def calculate_curriculum_stages(segment_files, transcriptions):
    """
    Determine the curriculum stages based on the data
    Returns list of (target_seconds, target_words) tuples
    """
    # Find the maximum number of words in any transcription
    max_words = max(len(t.split()) for t in transcriptions)

    stages = []
    stage_num = 1

    # Create stages: 1 word per 1.3 seconds
    while stage_num <= max_words:
        target_seconds = stage_num * CHUNK_DURATION
        target_words = stage_num * WORDS_PER_CHUNK
        stages.append((target_seconds, target_words, stage_num))
        stage_num += 1

    # Final stage: full audio → full transcription
    stages.append((None, None, stage_num))  # None means full

    return stages

# ==============================================================
# Training for one curriculum stage
# ==============================================================
def train_stage(model, segment_files, transcriptions, vocab, segment_name,
                stage_num, target_seconds, target_words, num_epochs=100, learning_rate=1e-5):
    """
    Train model on one curriculum stage

    Args:
        stage_num: The curriculum stage number (for logging)
        target_seconds: Audio duration to use (None = full)
        target_words: Number of words to predict (None = all)
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    best_loss = float('inf')
    prev_loss = float('inf')
    start_time = time.time()

    # Build description
    audio_desc = f"{target_seconds:.1f}s" if target_seconds else "full"
    text_desc = f"{target_words} word(s)" if target_words else "full"
    checkpoint_name = f"checkpoint_curriculum_{segment_name}_stage{stage_num:02d}.pt"

    print(f"\n{'='*60}")
    print(f"CURRICULUM STAGE {stage_num}")
    print(f"Audio: {audio_desc} → Text: {text_desc}")
    print(f"{'='*60}\n")

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
                # Only process if we have enough words
                if len(words) < target_words:
                    continue
                target_text = " ".join(words[:target_words])
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

        if total_iterations == 0:
            print(f"⚠️  Warning: No valid training samples in this stage. Skipping.")
            break

        avg_loss = total_loss / total_iterations

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "stage": stage_num
            }, checkpoint_name)
            best_marker = " ⭐ NEW BEST!"
        else:
            best_marker = ""

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} | Loss={avg_loss:.4f} | LR={current_lr:.6f} | Time={elapsed:.1f}s{best_marker}")

        # Early stopping check
        loss_change = prev_loss - avg_loss
        if loss_change < 0.001 and epoch > 0:
            print(f"⚠️  Early stopping: loss change ({loss_change:.6f}) < 0.001")
            break
        prev_loss = avg_loss

        # Step the learning rate scheduler
        scheduler.step()

        # Sample generation every 5 epochs
        if epoch % 5 == 0 or epoch == num_epochs - 1:
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
            if target_seconds:
                audio_duration = target_seconds
            else:
                waveform, sr = torchaudio.load(segment_files[0])
                audio_duration = waveform.shape[1] / sr

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
                else:
                    display_words = generated_words

                print(f"  🔹 Generated: {' '.join(display_words)}")
                print(f"  🔸 Expected: {expected_text}")
            model.train()

    total_time = time.time() - start_time
    print(f"\n✓ Stage {stage_num} complete in {total_time:.1f}s | Best loss: {best_loss:.4f}")

    # Load best checkpoint
    if os.path.exists(checkpoint_name):
        print(f"✓ Loading best checkpoint from {checkpoint_name}...")
        checkpoint = torch.load(checkpoint_name, map_location=device)
        model.load_state_dict(checkpoint["model"])
        print(f"✓ Loaded best model from epoch {checkpoint['epoch'] + 1}")

    return model

# ==============================================================
# Main
# ==============================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python3 train_curriculum.py <segment> [dataset_name]")
        print("Examples:")
        print("  python3 train_curriculum.py 002-04 base")
        print("  python3 train_curriculum.py 001 base")
        sys.exit(1)

    segment_name = sys.argv[1]  # e.g., "001", "002-01", "002-04"
    dataset_name = sys.argv[2] if len(sys.argv) > 2 else "base"

    datasets_dir = f"../{dataset_name}/audio"
    vocab_path = "../../vocabulary.json"
    model_path = "../../models/encoder_decoder_model.pt"

    print(f"\n{'='*60}")
    print(f"CURRICULUM LEARNING - SEGMENT: {segment_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Chunk size: {CHUNK_DURATION}s → {WORDS_PER_CHUNK} word(s)")
    print(f"{'='*60}\n")

    # Load vocab
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"Vocabulary size: {len(vocab)}")

    # Parse segment name to determine surah number
    surah_num = segment_name.split('-')[0]  # "001" or "002"

    # Load transcriptions and segments
    text_path = f"../{dataset_name}/text/{segment_name}.txt"
    if not os.path.exists(text_path):
        print(f"❌ Error: Text file not found: {text_path}")
        sys.exit(1)

    with open(text_path, "r", encoding="utf-8") as f:
        transcriptions = [line.strip() for line in f if line.strip()]

    segment_files = sorted(glob.glob(os.path.join(datasets_dir, surah_num, f"{segment_name}-*.wav")))

    if not segment_files:
        print(f"❌ Error: No audio segments found in {datasets_dir}/{surah_num}/{segment_name}-*.wav")
        sys.exit(1)

    print(f"Loaded {len(transcriptions)} transcriptions, {len(segment_files)} audio segments")

    if len(transcriptions) != len(segment_files):
        print(f"⚠️  Warning: Mismatch between transcriptions ({len(transcriptions)}) and segments ({len(segment_files)})")

    # Calculate curriculum stages
    stages = calculate_curriculum_stages(segment_files, transcriptions)
    print(f"\nCurriculum has {len(stages)} stages:")
    for i, (secs, words, stage_num) in enumerate(stages):
        audio_desc = f"{secs:.1f}s" if secs else "full"
        text_desc = f"{words} word(s)" if words else "full"
        print(f"  Stage {stage_num}: {audio_desc} → {text_desc}")

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

    # Load existing model if available
    import shutil
    if os.path.exists(model_path):
        backup_path = model_path.replace(".pt", f"_backup_curriculum_{segment_name}.pt")
        shutil.copy2(model_path, backup_path)
        print(f"\n✓ Backup created: {backup_path}")

        print(f"Loading existing model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✓ Model loaded successfully! Starting curriculum training on {segment_name}.")
    else:
        print(f"\nNo existing model found. Starting with fresh weights for curriculum training.")

    # Train through all stages
    total_start_time = time.time()

    for target_seconds, target_words, stage_num in stages:
        model = train_stage(
            model,
            segment_files,
            transcriptions,
            vocab,
            segment_name,
            stage_num,
            target_seconds,
            target_words,
            num_epochs=100,
            learning_rate=1e-5
        )

        # Save intermediate model after each stage
        intermediate_path = model_path.replace(".pt", f"_stage{stage_num:02d}.pt")
        torch.save(model.state_dict(), intermediate_path)
        print(f"✓ Stage {stage_num} model saved to: {intermediate_path}\n")

    # Save final model
    torch.save(model.state_dict(), model_path)

    total_time = time.time() - total_start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)

    print(f"\n{'='*60}")
    print(f"✓ CURRICULUM TRAINING COMPLETED!")
    print(f"Total time: {minutes}m {seconds}s")
    print(f"Final model saved to: {model_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
