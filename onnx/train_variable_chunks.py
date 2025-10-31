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


def split_into_variable_chunks(audio_features, text_tokens, fps=20):
  """
  Split audio into chunks of 1, 2, 3, 4, and 5 seconds
  For chunks without assigned words:
  - If silent: output <s></s>
  - If has audio: use 50% previous word, 50% next word

  Returns:
    List of (audio_chunk, text_chunk, chunk_duration) tuples
  """
  total_frames = audio_features.shape[0]
  total_duration = total_frames / fps

  chunks = []

  # For each chunk duration (just 1 second for now)
  for chunk_duration in [1]:
    frames_per_chunk = chunk_duration * fps
    num_chunks = int(total_frames / frames_per_chunk)

    if num_chunks == 0:
      continue

    # Estimate tokens per chunk
    tokens_per_chunk = len(text_tokens) / num_chunks

    print(f"\n{chunk_duration}s chunks:")

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

    # Second pass: fill in missing words using prev/next
    import random
    for idx, (audio_chunk, text_chunk, chunk_idx) in enumerate(chunk_assignments):
      if audio_chunk.shape[0] > 0:
        # Calculate audio energy
        audio_energy = audio_chunk.abs().mean().item()
        energy_threshold = -5.0

        # Determine output
        if len(text_chunk) == 0:
          # No text assigned to this chunk
          if audio_energy > energy_threshold:
            # Has audio - use 50% prev, 50% next word
            prev_word = None
            next_word = None

            # Find previous word
            for i in range(idx - 1, -1, -1):
              if len(chunk_assignments[i][1]) > 0:
                prev_word = chunk_assignments[i][1][-1]  # Last word of previous chunk
                break

            # Find next word
            for i in range(idx + 1, len(chunk_assignments)):
              if len(chunk_assignments[i][1]) > 0:
                next_word = chunk_assignments[i][1][0]  # First word of next chunk
                break

            # Randomly choose prev or next (50/50)
            if prev_word is not None and next_word is not None:
              text_chunk = [random.choice([prev_word, next_word])]
              token_label = "interpolated (prev/next)"
            elif prev_word is not None:
              text_chunk = [prev_word]
              token_label = "prev word"
            elif next_word is not None:
              text_chunk = [next_word]
              token_label = "next word"
            else:
              text_chunk = [0]  # <unk> if no neighbors
              token_label = "<unk>"
          else:
            # Silent chunk
            text_chunk = []
            token_label = "silence (<s></s>)"
        else:
          token_label = f"{len(text_chunk)} tokens"

        chunks.append((audio_chunk, text_chunk, chunk_duration))
        print(f"  Chunk {chunk_idx+1}: 1s = {audio_chunk.shape[0]} frames, {token_label}")

  return chunks


def tokenize_text(text, vocab):
  """Tokenize text"""
  word_to_idx = {word: idx for idx, word in enumerate(vocab)}
  words = text.split()
  return [word_to_idx.get(word, 0) for word in words]


def train_on_chunks(model, chunks, vocab, num_epochs=300, lr=5e-5):
  """Train model on variable-length chunks"""
  # Move model to device
  model = model.to(device)

  optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

  print(f"\n{'='*60}")
  print(f"Training Configuration:")
  print(f"{'='*60}")
  print(f"Number of chunks: {len(chunks)}")
  print(f"Chunk durations: 1s, 2s, 3s, 4s, 5s")
  print(f"Learning rate: {lr}")
  print(f"Epochs: {num_epochs}")
  print(f"Optimizer: AdamW")

  model.train()

  print(f"\n{'='*60}")
  print(f"Training Progress:")
  print(f"{'='*60}")

  import time
  start_time = time.time()

  for epoch in range(num_epochs):
    total_loss = 0

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

    # Print progress
    if (epoch + 1) % 20 == 0 or epoch == 0:
      elapsed = time.time() - start_time
      print(f"Epoch {epoch+1:3d}/{num_epochs}: Avg Loss = {avg_loss:.4f} | Time: {elapsed:.1f}s")

    # Generate sample every 50 epochs
    if (epoch + 1) % 50 == 0:
      model.eval()
      with torch.no_grad():
        # Test on first 5-second chunk (if available)
        test_idx = -1  # Last chunk (likely 5 seconds)
        audio_test = chunks[test_idx][0].unsqueeze(0).to(device)
        generated = model.generate(audio_test, max_new_tokens=10, temperature=0.5)
        generated_words = [vocab[idx] for idx in generated[0].cpu().tolist()]
        print(f"  Sample ({chunks[test_idx][2]}s): {' '.join(generated_words[:8])}...")
      model.train()

  total_time = time.time() - start_time
  print(f"\n{'='*60}")
  print(f"Training Complete!")
  print(f"Total time: {total_time:.1f}s ({total_time/num_epochs:.2f}s per epoch)")
  print(f"{'='*60}")

  return model


def main():
  """Main training function"""

  # Paths
  audio_path = "/Users/amraboelela/develop/android/AndroidArabicWhisper/app/src/main/assets/001.wav"
  vocab_path = "vocabulary.json"

  # Load vocabulary
  print("Loading vocabulary...")
  with open(vocab_path, "r", encoding="utf-8") as f:
    vocab = json.load(f)
  print(f"Vocabulary size: {len(vocab)}")

  # Create improved model
  print("\nCreating improved model...")
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

  total_params = sum(p.numel() for p in model.parameters())
  model_size_mb = total_params * 4 / (1024**2)  # FP32
  print(f"Model parameters: {total_params:,}")
  print(f"Model size (FP32): ~{model_size_mb:.1f} MB")

  # Prepare data
  print(f"\n{'='*60}")
  print(f"Preparing Al-Fatiha Data (variable chunks):")
  print(f"{'='*60}")

  alfatiha_text = "اعوذ بالله من الشيطان الرجيم بسم الله الرحمن الرحيم الحمد لله رب العالمين الرحمن الرحيم مالك يوم الدين اياك نعبد واياك نستعين اهدنا الصراط المستقيم صراط الذين انعمت عليهم غير المغضوب عليهم ولا الضالين"

  print(f"Al-Fatiha text:")
  print(f"  {alfatiha_text}")

  # Extract audio
  print(f"\nExtracting audio features...")
  audio_features, sample_rate = extract_mel_features(audio_path)
  print(f"Total audio: {audio_features.shape[0]} frames ({audio_features.shape[0]/10:.1f} seconds)")

  # Tokenize text
  text_tokens = tokenize_text(alfatiha_text, vocab)
  print(f"Total text: {len(text_tokens)} tokens")

  # Split into variable chunks (1s, 2s, 3s, 4s, 5s)
  print(f"\nSplitting into variable-length chunks:")
  chunks = split_into_variable_chunks(audio_features, text_tokens, fps=20)
  print(f"\nTotal chunks created: {len(chunks)}")

  # Show chunk distribution
  from collections import Counter
  chunk_durations = [c[2] for c in chunks]
  duration_counts = Counter(chunk_durations)
  print("\nChunk distribution:")
  for duration in sorted(duration_counts.keys()):
    print(f"  {duration}s: {duration_counts[duration]} chunks")

  # Train model
  model = train_on_chunks(
    model,
    chunks,
    vocab,
    num_epochs=100,
    lr=5e-5
  )

  # Test on various chunk sizes
  print(f"\n{'='*60}")
  print(f"Testing on Different Chunk Sizes:")
  print(f"{'='*60}")

  model.eval()

  # Test one example from each duration
  tested_durations = set()
  for audio_chunk, text_chunk, chunk_duration in chunks:
    if chunk_duration not in tested_durations:
      with torch.no_grad():
        test_audio = audio_chunk.unsqueeze(0).to(device)
        generated = model.generate(test_audio, max_new_tokens=len(text_chunk)+5, temperature=0.1)
        generated_words = [vocab[idx] for idx in generated[0].cpu().tolist()]
        expected_words = [vocab[idx] for idx in text_chunk]

        print(f"\n{chunk_duration}s chunk:")
        print(f"Expected:  {' '.join(expected_words)}")
        print(f"Generated: {' '.join(generated_words)}")

      tested_durations.add(chunk_duration)

      if len(tested_durations) == 5:  # Tested all durations
        break

  # Save model
  save_path = "alfatiha_model_variable.pt"
  torch.save(model.state_dict(), save_path)
  print(f"\n{'='*60}")
  print(f"Model saved to: {save_path}")
  print(f"Model size: ~{model_size_mb:.1f} MB")
  print(f"{'='*60}")


if __name__ == "__main__":
  main()
