#!/usr/bin/env python3
"""
Train encoder-decoder model combining both approaches:
- Full segments (complete audio + complete transcription)
- First second (1 second audio + first word only)
"""
import json
import torch
import torch.nn as nn
import torchaudio
import glob
import os
import random
import time
import shutil
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
# Audio feature extraction
# ==============================================================
def extract_mel_features(audio_path, n_mels=80, max_duration_seconds=None):
    """Extract Whisper-compatible mel spectrogram features"""
    waveform, sample_rate = torchaudio.load(audio_path)

    # Truncate audio if max_duration_seconds is specified
    if max_duration_seconds is not None:
        max_samples = int(max_duration_seconds * sample_rate)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

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
    mel_features = (mel_features - mel_features.mean()) / (mel_features.std() + 1e-5)
    return mel_features, sample_rate

# ==============================================================
# Tokenization
# ==============================================================
def tokenize_text(text, vocab):
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    words = text.split()
    return [word_to_idx.get(word, 0) for word in words]  # 0 = unknown

# ==============================================================
# Combined Training
# ==============================================================
def train_combined(model, segment_files, transcriptions, vocab, num_epochs=100, initial_lr=1e-3, min_lr=1e-6):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

    best_loss = float('inf')
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_iterations = 0
        indices = list(range(len(segment_files)))
        random.shuffle(indices)

        for i in indices:
            seg_file = segment_files[i]
            full_text = transcriptions[i]

            # Randomly choose between full segment or first-second training
            use_first_second = random.random() < 0.5

            if use_first_second:
                # First-second training (1 second audio + first word)
                first_word = full_text.split()[0] if full_text.split() else full_text
                audio_features, _ = extract_mel_features(seg_file, max_duration_seconds=1.0)
                text_to_use = first_word
            else:
                # Full segment training
                audio_features, _ = extract_mel_features(seg_file)
                text_to_use = full_text

            # audio_features is (time, n_mels), need (n_mels, time) for Whisper
            audio_batch = audio_features.transpose(0, 1).unsqueeze(0).to(device)

            # Tokenize text
            text_tokens = tokenize_text(text_to_use, vocab)
            full_sequence = [1] + text_tokens + [2]  # <s> + tokens + </s>
            input_ids = torch.tensor([full_sequence[:-1]], dtype=torch.long, device=device)
            labels = torch.tensor([full_sequence[1:]], dtype=torch.long, device=device)

            # Forward + loss
            logits = model(mel_features=audio_batch, text_ids=input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_iterations += 1

        avg_loss = total_loss / total_iterations
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch
            }, "checkpoint_best.pt")
            best_marker = " ⭐ NEW BEST!"
        else:
            best_marker = ""

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} | Loss={avg_loss:.4f} | LR={current_lr:.6f} | Time={elapsed:.1f}s{best_marker}")

        # Sample generation every 50 epochs
        if (epoch + 1) % 50 == 0:
            model.eval()

            # Test on full segment
            test_audio_full, sample_rate = extract_mel_features(segment_files[0])
            waveform, sr = torchaudio.load(segment_files[0])
            audio_duration = waveform.shape[1] / sr
            test_audio_full = test_audio_full.transpose(0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                generated = model.generate(test_audio_full, max_new_tokens=50, audio_duration_seconds=audio_duration)
                generated_ids = generated[0].tolist()
                if generated_ids and generated_ids[0] == 1:
                    generated_ids = generated_ids[1:]
                if 2 in generated_ids:
                    generated_ids = generated_ids[:generated_ids.index(2)]
                generated_words = [vocab[idx] for idx in generated_ids if idx < len(vocab)]
                print(f"  🔹 Full segment: {' '.join(generated_words)}")
                print(f"  🔸 Expected: {transcriptions[0]}")

            # Test on first second
            test_audio_first, _ = extract_mel_features(segment_files[0], max_duration_seconds=1.0)
            test_audio_first = test_audio_first.transpose(0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                generated = model.generate(test_audio_first, max_new_tokens=10, audio_duration_seconds=1.0)
                generated_ids = generated[0].tolist()
                if generated_ids and generated_ids[0] == 1:
                    generated_ids = generated_ids[1:]
                if 2 in generated_ids:
                    generated_ids = generated_ids[:generated_ids.index(2)]
                generated_words = [vocab[idx] for idx in generated_ids if idx < len(vocab)]
                first_word_expected = transcriptions[0].split()[0] if transcriptions[0].split() else transcriptions[0]
                print(f"  🔹 First second: {' '.join(generated_words)}")
                print(f"  🔸 Expected: {first_word_expected}")

            model.train()

    total_time = time.time() - start_time
    print(f"Training complete in {total_time:.1f}s | Best loss: {best_loss:.4f}")
    return model

# ==============================================================
# Main
# ==============================================================
def main():
    segments_dir = "segments"
    vocab_path = "vocabulary.json"
    model_path = "encoder_decoder_model.pt"

    # Load vocab
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"Vocabulary size: {len(vocab)}")

    # Load combined data from Al-Fatiha (001) and Al-Baqara (002-01)
    transcriptions = []
    segment_files = []

    # Load Al-Fatiha (001)
    fatiha_text_path = os.path.join(segments_dir, "001.txt")
    with open(fatiha_text_path, "r", encoding="utf-8") as f:
        fatiha_transcriptions = [line.strip() for line in f if line.strip()]
    fatiha_segments = sorted(glob.glob(os.path.join(segments_dir, "001-*.wav")))
    print(f"Loaded {len(fatiha_transcriptions)} Al-Fatiha transcriptions, {len(fatiha_segments)} segments")

    # Load Al-Baqara (002-01)
    baqara_text_path = os.path.join(segments_dir, "002-01.txt")
    with open(baqara_text_path, "r", encoding="utf-8") as f:
        baqara_transcriptions = [line.strip() for line in f if line.strip()]
    baqara_segments = sorted(glob.glob(os.path.join(segments_dir, "002-01-*.wav")))
    print(f"Loaded {len(baqara_transcriptions)} Al-Baqara transcriptions, {len(baqara_segments)} segments")

    # Combine both datasets
    transcriptions = fatiha_transcriptions + baqara_transcriptions
    segment_files = fatiha_segments + baqara_segments
    print(f"Total combined: {len(transcriptions)} transcriptions, {len(segment_files)} segments")
    print(f"Training with combined approach: 50% full segments, 50% first-second")

    # Create smaller 128-dimension encoder-decoder
    model = EncoderDecoderTransformer(
        vocab_size=len(vocab),
        d_model=128,           # Smaller dimension
        n_encoder_layers=4,
        n_decoder_layers=4,
        n_heads=4,             # 128/4 = 32 dim per head
        d_ff=512,              # 4x d_model
        dropout=0.1
    )

    # Load existing model if available
    if os.path.exists(model_path):
        # Create backup before training
        backup_path = model_path.replace(".pt", "_backup.pt")
        shutil.copy2(model_path, backup_path)
        print(f"✓ Backup created: {backup_path}")

        print(f"Loading existing model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✓ Model loaded successfully! Continuing training with combined approach.")
    else:
        print("No existing model found. Starting with fresh weights for combined training")

    # Train with combined approach
    model = train_combined(
        model,
        segment_files,
        transcriptions,
        vocab,
        num_epochs=100,
        initial_lr=1e-3,
        min_lr=1e-6
    )

    # Save final model
    torch.save(model.state_dict(), model_path)
    print(f"Final model saved to: {model_path}")


if __name__ == "__main__":
    main()
