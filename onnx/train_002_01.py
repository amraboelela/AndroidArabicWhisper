#!/usr/bin/env python3
"""
Train model on 002-01 segments, building on existing quran_model.pt
"""
import json
import torch
import torch.nn as nn
import torchaudio
import glob
import os
from improved_transformer import ImprovedDecoderTransformer

# Force CPU - MPS has a bug with this model
device = torch.device("cpu")
print("Using CPU (MPS has a bug with backward pass)")

print(f"Device: {device}")


def extract_mel_features(audio_path, n_mels=800, target_fps=20):
    """Extract mel spectrogram features"""
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

    return mel_features, sample_rate


def tokenize_text(text, vocab):
    """Tokenize text"""
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    words = text.split()
    return [word_to_idx.get(word, 0) for word in words]


def train_on_segments(model, segment_files, transcriptions, vocab, num_epochs=50, initial_lr=1e-3, min_lr=1e-5):
    """Train model on segments"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=0.01)

    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=min_lr
    )

    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"{'='*60}")
    print(f"Number of segments: {len(segment_files)}")
    print(f"Initial learning rate: {initial_lr}")
    print(f"Minimum learning rate: {min_lr}")
    print(f"Epochs: {num_epochs}")
    print(f"Optimizer: AdamW with Cosine Annealing")

    model.train()

    print(f"\n{'='*60}")
    print(f"Training Progress:")
    print(f"{'='*60}")

    import time
    start_time = time.time()

    best_loss = float('inf')

    for epoch in range(num_epochs):
        total_loss = 0

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Train on each segment
        for segment_file, transcription in zip(segment_files, transcriptions):
            optimizer.zero_grad()

            # Extract audio features
            audio_features, _ = extract_mel_features(segment_file)
            audio_features = audio_features.unsqueeze(0).to(device)

            # Tokenize transcription
            tokens = tokenize_text(transcription, vocab)
            if len(tokens) == 0:
                continue

            target = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

            # Forward pass using the ImprovedDecoderTransformer API
            # Add .contiguous() to fix MPS memory layout issues
            audio_features_cont = audio_features.contiguous()
            target_cont = target.contiguous()

            logits = model(audio_features=audio_features_cont, text_ids=target_cont[:, :-1])

            # Compute loss
            loss = nn.CrossEntropyLoss()(
                logits.reshape(-1, logits.shape[-1]).contiguous(),
                target_cont[:, 1:].reshape(-1).contiguous()
            )

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(segment_files)

        # Print progress every epoch
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1:3d}/{num_epochs} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f} | Time: {elapsed:.1f}s")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'quran_model.pt')
            print(f"  ✓ Saved best model (loss: {best_loss:.4f})")

        # Step scheduler
        scheduler.step()

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Saved to: quran_model.pt")


def main():
    # Load vocabulary from vocabulary.json
    print("\nLoading vocabulary from vocabulary.json...")
    with open("vocabulary.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)

    print(f"✓ Loaded vocabulary with {len(vocab)} words")

    # Initialize model with same config as train_baqarah.py
    print("\nInitializing model...")
    model = ImprovedDecoderTransformer(
        vocab_size=len(vocab),
        d_model=800,
        n_layers=5,
        n_heads=10,
        d_ff=3200,
        dropout=0.1
    )

    # Load existing weights
    model_path = "quran_model.pt"
    print(f"Loading existing model weights from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✓ Model weights loaded successfully")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load 002-01 segments and transcriptions
    segment_files = sorted(glob.glob("segments/002-01-*.wav"))

    with open("segments/002-01.txt", "r", encoding="utf-8") as f:
        transcriptions = [line.strip() for line in f.readlines()]

    print(f"\nLoaded {len(segment_files)} audio segments")
    print(f"Loaded {len(transcriptions)} transcriptions")

    # Train on segments
    train_on_segments(model, segment_files, transcriptions, vocab, num_epochs=50)


if __name__ == "__main__":
    main()
