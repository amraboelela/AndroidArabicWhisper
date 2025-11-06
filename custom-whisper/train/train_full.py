#!/usr/bin/env python3
"""
Universal training script for encoder-decoder model
Usage: python3 train_full.py <dataset_name> <surah_part>
Examples:
  python3 train_full.py Quran-A 001       # Train on Al-Fatiha (001)
  python3 train_full.py Quran-A 002-01    # Train on Al-Baqara part 1
  python3 train_full.py Quran-A 002-04    # Train on Al-Baqara part 4
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

    # Learning rate scheduler - reduces LR by 0.5x after each epoch
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    best_loss = float('inf')
    prev_loss = float('inf')
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
        print(f"Epoch {epoch+1}/{num_epochs} | Loss={avg_loss:.4f} | LR={current_lr:.6f} | Time={elapsed:.1f}s{best_marker}")

        # Early stopping check
        loss_change = prev_loss - avg_loss
        if loss_change < min_delta:  # Includes small improvement, no change, or getting worse
            patience_counter += 1
            print(f"⚠️  Low/negative loss change ({loss_change:.6f}) | Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"⚠️  Early stopping: loss not improving for {patience} consecutive epochs")
                break
        else:
            patience_counter = 0  # Reset counter if loss change is significant
        prev_loss = avg_loss

        # Step the learning rate scheduler
        scheduler.step()

        # Sample generation
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
            else:
                display_words = generated_words

            print(f"  🔹 Generated: {' '.join(display_words)}")
            print(f"  🔸 Expected: {expected_text}")
        model.train()

    total_time = time.time() - start_time
    print(f"Training complete in {total_time:.1f}s | Best loss: {best_loss:.4f}")

    # Load best checkpoint
    if os.path.exists(checkpoint_name):
        print(f"\n✓ Loading best checkpoint from {checkpoint_name}...")
        checkpoint = torch.load(checkpoint_name, map_location=device)
        model.load_state_dict(checkpoint["model"])
        print(f"✓ Loaded best model from epoch {checkpoint['epoch'] + 1}")

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
    model_path = "../models/encoder_decoder_model.pt"

    # Load vocab
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"Vocabulary size: {len(vocab)}")

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

    # Load existing model and continue training
    import shutil
    if os.path.exists(model_path):
        backup_path = model_path.replace(".pt", f"_backup_{surah_part}.pt")
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
