#!/usr/bin/env python3
"""
Train encoder-decoder model on Al-Fatiha segments (improved version)
"""
import json
import torch
import torch.nn as nn
import torchaudio
import glob
import os
import random
import time
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
def extract_mel_features(audio_path, n_mels=128, target_fps=20):
    """Extract mel spectrogram features with normalization"""
    waveform, sample_rate = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    hop_length = sample_rate // target_fps
    n_fft = 2048

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

    # Normalize features for stable training
    mel_features = (mel_features - mel_features.mean()) / (mel_features.std() + 1e-5)

    return mel_features, sample_rate


# ==============================================================
# Tokenization
# ==============================================================
def tokenize_text(text, vocab):
    """Tokenize text into vocabulary indices"""
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    words = text.split()
    return [word_to_idx.get(word, 0) for word in words]  # 0 = unknown


# ==============================================================
# Training
# ==============================================================
def train_on_segments(model, segment_files, transcriptions, vocab, num_epochs=100, initial_lr=1e-3, min_lr=1e-5):
    """Train encoder-decoder model on segments"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=0.01)

    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=min_lr
    )

    # Label smoothing
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"{'='*60}")
    print(f"Number of segments: {len(segment_files)}")
    print(f"Initial learning rate: {initial_lr}")
    print(f"Minimum learning rate: {min_lr}")
    print(f"Epochs: {num_epochs}")
    print(f"Optimizer: AdamW + CosineAnnealingLR")
    print(f"{'='*60}\n")

    start_time = time.time()
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        # Shuffle order each epoch
        indices = list(range(len(segment_files)))
        random.shuffle(indices)

        for i in indices:
            seg_file = segment_files[i]
            text = transcriptions[i]

            # Extract mel features
            audio_features, _ = extract_mel_features(seg_file)
            audio_batch = audio_features.unsqueeze(0).to(device)

            # Tokenize and prepare sequences
            text_tokens = tokenize_text(text, vocab)
            full_sequence = [1] + text_tokens + [2]  # <s> + tokens + </s>

            input_ids = torch.tensor([full_sequence[:-1]], dtype=torch.long, device=device)
            labels = torch.tensor([full_sequence[1:]], dtype=torch.long, device=device)

            # Forward + loss
            logits = model(audio_features=audio_batch, text_ids=input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(segment_files)
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        is_best = False

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch
            }, "checkpoint_best.pt")
            is_best = True

        # Print progress
        elapsed = time.time() - start_time
        best_marker = " ⭐ NEW BEST!" if is_best else ""
        print(f"Epoch {epoch+1:3d}/{num_epochs}: Avg Loss = {avg_loss:.4f} | LR = {current_lr:.6f} | Time = {elapsed:.1f}s{best_marker}")

        # Sample generation every 50 epochs
        if (epoch + 1) % 50 == 0:
            model.eval()
            test_audio, _ = extract_mel_features(segment_files[0])
            test_audio = test_audio.unsqueeze(0).to(device)
            with torch.no_grad():
                generated = model.generate(test_audio, max_new_tokens=10)
                generated_ids = generated[0].tolist()
                if generated_ids and generated_ids[0] == 1:
                    generated_ids = generated_ids[1:]
                if 2 in generated_ids:
                    generated_ids = generated_ids[:generated_ids.index(2)]
                generated_words = [vocab[idx] for idx in generated_ids if idx < len(vocab)]
                print(f"  🔹 Sample: {' '.join(generated_words)}")
                print(f"  🔸 Expected: {transcriptions[0]}")
            model.train()

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training Complete! Total time: {total_time:.1f}s")
    print(f"Best loss: {best_loss:.4f}")
    print(f"{'='*60}\n")

    return model


# ==============================================================
# Main
# ==============================================================
def main():
    """Main training function"""
    segments_dir = "segments"
    text_path = "segments/001.txt"
    vocab_path = "vocabulary.json"
    model_path = "encoder_decoder_model.pt"

    # Load vocabulary
    print("Loading vocabulary...")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"Vocabulary size: {len(vocab)}")

    # Load text
    print(f"\nLoading transcriptions from: {text_path}")
    with open(text_path, "r", encoding="utf-8") as f:
        transcriptions = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(transcriptions)} transcriptions")

    # Find audio segments
    segment_files = sorted(glob.glob(os.path.join(segments_dir, "001-*.wav")))
    print(f"Found {len(segment_files)} audio segments")

    if len(segment_files) != len(transcriptions):
        print(f"⚠️  Mismatch: {len(segment_files)} segments vs {len(transcriptions)} texts")

    print("\nSegment-text pairs:")
    for s, t in zip(segment_files, transcriptions):
        print(f"  {os.path.basename(s)}: {t}")

    # Create model
    print("\nCreating encoder-decoder model...")
    model = EncoderDecoderTransformer(
        vocab_size=len(vocab),
        d_model=64,            # Much smaller!
        n_encoder_layers=1,    # Single layer
        n_decoder_layers=1,    # Single layer
        n_heads=2,             # 2 heads
        d_ff=256,              # Small FFN
        dropout=0.2
    )

    # Load checkpoint if exists
    if os.path.exists("checkpoint_best.pt"):
        print("✓ Loading checkpoint_best.pt ...")
        checkpoint = torch.load("checkpoint_best.pt", map_location=device)
        model.load_state_dict(checkpoint["model"])
        print(f"Resuming from epoch {checkpoint.get('epoch', '?')}")
    elif os.path.exists(model_path):
        print(f"✓ Loading model weights from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("✗ Starting with fresh weights")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Model size (FP32): ~{total_params * 4 / (1024**2):.1f} MB")

    # Train
    model = train_on_segments(
        model,
        segment_files,
        transcriptions,
        vocab,
        num_epochs=100,
        initial_lr=1e-3,
        min_lr=1e-5
    )

    # Save final model
    torch.save(model.state_dict(), model_path)
    print(f"\n✓ Final model saved to: {model_path}")


if __name__ == "__main__":
    main()
