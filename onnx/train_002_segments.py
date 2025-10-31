#!/usr/bin/env python3
"""
Train model on Al-Fatiha segments
"""
import json
import torch
import torch.nn as nn
import torchaudio
import glob
import os
from improved_transformer import ImprovedDecoderTransformer

# Check for Metal GPU support
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
            audio_features, sample_rate = extract_mel_features(segment_file)
            audio_batch = audio_features.unsqueeze(0).to(device)

            # Tokenize text
            text_tokens = tokenize_text(transcription, vocab)

            input_tokens = [1] + text_tokens  # Add <s>
            target_tokens = text_tokens + [2]  # Add </s>

            input_ids = torch.tensor([input_tokens], dtype=torch.long, device=device)
            labels = torch.tensor([target_tokens], dtype=torch.long, device=device)

            # Forward pass
            logits, loss = model(
                audio_features=audio_batch,
                text_ids=input_ids,
                labels=labels
            )

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(segment_files)

        # Step the scheduler
        scheduler.step()

        # Save best model
        is_best = False
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "quran_model.pt")
            is_best = True

        # Print progress every 5 epochs OR when there's a new best
        if (epoch + 1) % 5 == 0 or epoch == 0 or is_best:
            elapsed = time.time() - start_time
            best_marker = " ⭐ NEW BEST!" if is_best else ""
            print(f"Epoch {epoch+1:3d}/{num_epochs}: Avg Loss = {avg_loss:.4f} | LR = {current_lr:.6f} | Time: {elapsed:.1f}s{best_marker}")
            if is_best:
                print(f"  ✓ Best model saved (loss: {best_loss:.4f})")

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Total time: {total_time:.1f}s ({total_time/num_epochs:.2f}s per epoch)")
    print(f"Best loss: {best_loss:.4f}")
    print(f"{'='*60}")

    return model


def main():
    """Main training function"""

    # Paths
    segments_dir = "segments"
    text_path = "segments/002-01.txt"
    vocab_path = "vocabulary.json"
    model_path = "quran_model.pt"

    # Load vocabulary
    print("Loading vocabulary...")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"Vocabulary size: {len(vocab)}")

    # Load transcriptions
    print(f"\nLoading transcriptions from: {text_path}")
    with open(text_path, "r", encoding="utf-8") as f:
        transcriptions = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(transcriptions)} transcriptions")

    # Get segment files
    segment_files = sorted(glob.glob(os.path.join(segments_dir, "002-01-*.wav")))
    print(f"Found {len(segment_files)} audio segments")

    if len(segment_files) != len(transcriptions):
        print(f"⚠️  Warning: Mismatch between segments ({len(segment_files)}) and transcriptions ({len(transcriptions)})")

    # Display segment-text pairs
    print(f"\nSegment-Text pairs:")
    for seg, text in zip(segment_files, transcriptions):
        print(f"  {os.path.basename(seg)}: {text}")

    # Create model
    print("\nCreating model...")
    model = ImprovedDecoderTransformer(
        vocab_size=len(vocab),
        d_model=800,
        n_layers=5,
        n_heads=10,
        d_ff=3200,
        dropout=0.1
    )

    # Load existing weights if available
    if os.path.exists(model_path):
        print(f"✓ Loading existing model weights from: {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        print(f"✗ Starting with fresh model weights")

    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / (1024**2)
    print(f"Model parameters: {total_params:,}")
    print(f"Model size (FP32): ~{model_size_mb:.1f} MB")

    # Train model
    model = train_on_segments(
        model,
        segment_files,
        transcriptions,
        vocab,
        num_epochs=50,
        initial_lr=1e-3,
        min_lr=1e-5
    )

    # Save final model
    torch.save(model.state_dict(), "quran_model.pt")
    print(f"\n✓ Final model saved to: quran_model.pt")


if __name__ == "__main__":
    main()
