#!/usr/bin/env python3
import json
import torch
import torch.nn as nn
import torchaudio
import numpy as np
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


def split_into_variable_chunks(audio_features, text_tokens, fps=20, silent_threshold=-5.0):
  """
  Split audio into 1-second chunks
  Handle silence at beginning and prev/next word interpolation
  """
  total_frames = audio_features.shape[0]

  chunks = []
  chunk_duration = 1
  frames_per_chunk = chunk_duration * fps
  num_chunks = int(total_frames / frames_per_chunk)

  if num_chunks == 0:
    return chunks

  # Estimate tokens per chunk
  tokens_per_chunk = len(text_tokens) / num_chunks

  print(f"\n{chunk_duration}s chunks:")
  print(f"Total chunks: {num_chunks}")
  print(f"Total tokens: {len(text_tokens)}")
  print(f"Tokens per chunk (avg): {tokens_per_chunk:.2f}")

  # First pass: assign words to chunks
  chunk_assignments = []
  for i in range(num_chunks):
    # Get audio chunk
    start_frame = i * frames_per_chunk
    end_frame = min(start_frame + frames_per_chunk, total_frames)
    audio_chunk = audio_features[start_frame:end_frame]

    # Get corresponding text chunk
    start_token = int(i * tokens_per_chunk)
    end_token = int((i + 1) * tokens_per_chunk) if i < num_chunks - 1 else len(text_tokens)
    text_chunk = text_tokens[start_token:end_token]

    chunk_assignments.append((audio_chunk, text_chunk, i))

  # Second pass: fill in missing words using prev/next or detect silence
  import random
  for idx, (audio_chunk, text_chunk, chunk_idx) in enumerate(chunk_assignments):
    if audio_chunk.shape[0] > 0:
      # Calculate audio energy
      audio_energy = audio_chunk.abs().mean().item()

      # Determine output
      if len(text_chunk) == 0:
        # No text assigned to this chunk
        if audio_energy > silent_threshold:
          # Has audio - use 50% prev, 50% next word
          prev_word = None
          next_word = None

          # Find previous word
          for i in range(idx - 1, -1, -1):
            if len(chunk_assignments[i][1]) > 0:
              prev_word = chunk_assignments[i][1][-1]
              break

          # Find next word
          for i in range(idx + 1, len(chunk_assignments)):
            if len(chunk_assignments[i][1]) > 0:
              next_word = chunk_assignments[i][1][0]
              break

          # Randomly choose prev or next (50/50)
          if prev_word is not None and next_word is not None:
            text_chunk = [random.choice([prev_word, next_word])]
            token_label = "interpolated"
          elif prev_word is not None:
            text_chunk = [prev_word]
            token_label = "prev word"
          elif next_word is not None:
            text_chunk = [next_word]
            token_label = "next word"
          else:
            text_chunk = []  # silence
            token_label = "silence (no neighbors)"
        else:
          # Silent chunk
          text_chunk = []
          token_label = "silence"
      else:
        token_label = f"{len(text_chunk)} tokens"

      chunks.append((audio_chunk, text_chunk, chunk_duration))

      # Print progress every 100 chunks
      if (chunk_idx + 1) % 100 == 0 or chunk_idx == 0 or chunk_idx == num_chunks - 1:
        print(f"  Chunk {chunk_idx+1}: 1s = {audio_chunk.shape[0]} frames, energy={audio_energy:.2f}, {token_label}")

  return chunks


def tokenize_text(text, vocab):
  """Tokenize text"""
  word_to_idx = {word: idx for idx, word in enumerate(vocab)}
  words = text.split()
  return [word_to_idx.get(word, 0) for word in words]


def train_on_chunks(model, chunks, vocab, num_epochs=50, initial_lr=1e-3, min_lr=1e-5):
  """Train model on variable-length chunks with dynamic learning rate"""
  model = model.to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=0.01)

  # Cosine annealing scheduler: starts high, smoothly decreases to min_lr
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs, eta_min=min_lr
  )

  print(f"\n{'='*60}")
  print(f"Training Configuration:")
  print(f"{'='*60}")
  print(f"Number of chunks: {len(chunks)}")
  print(f"Initial learning rate: {initial_lr}")
  print(f"Minimum learning rate: {min_lr}")
  print(f"Epochs: {num_epochs}")
  print(f"Optimizer: AdamW with Cosine Annealing")
  print(f"Scheduler: CosineAnnealingLR")

  model.train()

  print(f"\n{'='*60}")
  print(f"Training Progress:")
  print(f"{'='*60}")

  import time
  start_time = time.time()

  best_loss = float('inf')  # Track the best loss

  for epoch in range(num_epochs):
    total_loss = 0

    # Get current learning rate
    current_lr = optimizer.param_groups[0]['lr']

    # Train on each chunk
    for chunk_idx, (audio_chunk, text_chunk, chunk_duration) in enumerate(chunks):
      optimizer.zero_grad()

      # Prepare data
      audio_batch = audio_chunk.unsqueeze(0).to(device)
      input_tokens = [1] + text_chunk  # Add <s>
      target_tokens = text_chunk + [2]  # Add </s>

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

    avg_loss = total_loss / len(chunks)

    # Step the scheduler after each epoch
    scheduler.step()

    # Save best model if this is the lowest loss so far
    is_best = False
    if avg_loss < best_loss:
      best_loss = avg_loss
      best_model_path = "quran_model.pt"
      torch.save(model.state_dict(), best_model_path)
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
  print(f"{'='*60}")

  return model


def main():
  """Main training function for Baqarah 002-01"""

  # Paths
  audio_path = "/Users/amraboelela/develop/android/AndroidArabicWhisper/app/src/main/assets/002-01.wav"
  text_path = "baqarah_002-01_text.txt"
  vocab_path = "vocabulary.json"
  model_path = "quran_model.pt"

  # Load vocabulary
  print("Loading vocabulary...")
  with open(vocab_path, "r", encoding="utf-8") as f:
    vocab = json.load(f)
  print(f"Vocabulary size: {len(vocab)}")

  # Load Baqarah text
  print(f"\nLoading Baqarah text from: {text_path}")
  with open(text_path, "r", encoding="utf-8") as f:
    baqarah_text = f.read().strip()

  print(f"Text preview (first 100 chars): {baqarah_text[:100]}")
  print(f"Text preview (last 100 chars): {baqarah_text[-100:]}")

  # Create model architecture
  print("\nCreating model architecture...")
  print("  - Dimension: 800")
  print("  - 5 transformer layers")
  print("  - 10 attention heads")

  model = ImprovedDecoderTransformer(
    vocab_size=len(vocab),
    d_model=800,
    n_layers=5,
    n_heads=10,
    d_ff=3200,
    dropout=0.1
  )

  # Load existing weights if available
  import os
  if os.path.exists(model_path):
    print(f"\n✓ Loading existing model weights from: {model_path}")
    model.load_state_dict(torch.load(model_path))
    print("  Model weights loaded successfully!")
  else:
    print(f"\n✗ No existing model found at {model_path}")
    print("  Starting with fresh model weights")

  total_params = sum(p.numel() for p in model.parameters())
  model_size_mb = total_params * 4 / (1024**2)  # FP32
  print(f"Model parameters: {total_params:,}")
  print(f"Model size (FP32): ~{model_size_mb:.1f} MB")

  # Prepare data
  print(f"\n{'='*60}")
  print(f"Preparing Baqarah Data:")
  print(f"{'='*60}")

  # Extract audio
  print(f"\nExtracting audio features...")
  audio_features, sample_rate = extract_mel_features(audio_path)
  print(f"Total audio: {audio_features.shape[0]} frames ({audio_features.shape[0]/20:.1f} seconds)")

  # Tokenize text
  text_tokens = tokenize_text(baqarah_text, vocab)
  print(f"Total text: {len(text_tokens)} tokens")

  # Split into 1-second chunks
  print(f"\nSplitting into 1-second chunks:")
  chunks = split_into_variable_chunks(audio_features, text_tokens, fps=20)
  print(f"\nTotal chunks created: {len(chunks)}")

  # Train model (50 epochs with dynamic learning rate)
  model = train_on_chunks(
    model,
    chunks,
    vocab,
    num_epochs=50,
    initial_lr=1e-3,  # Start with high learning rate
    min_lr=1e-5       # End with low learning rate
  )

  # Save model
  save_path = "quran_model.pt"
  torch.save(model.state_dict(), save_path)
  print(f"\n{'='*60}")
  print(f"Model saved to: {save_path}")
  print(f"Model size: ~{model_size_mb:.1f} MB")
  print(f"{'='*60}")


if __name__ == "__main__":
  main()
