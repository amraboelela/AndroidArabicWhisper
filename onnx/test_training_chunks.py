#!/usr/bin/env python3
import json
import torch
import torchaudio
from improved_transformer import ImprovedDecoderTransformer

# Use CPU for testing
device = torch.device("cpu")

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


def split_into_chunks(audio_features, text_tokens, fps=20):
  """Split into 1-second chunks (same as training)"""
  chunk_duration = 1
  frames_per_chunk = chunk_duration * fps
  total_frames = audio_features.shape[0]
  num_chunks = int(total_frames / frames_per_chunk)

  tokens_per_chunk = len(text_tokens) / num_chunks

  chunks = []

  for i in range(num_chunks):
    # Get audio chunk
    start_frame = i * frames_per_chunk
    end_frame = min(start_frame + frames_per_chunk, total_frames)
    audio_chunk = audio_features[start_frame:end_frame]

    # Get corresponding text chunk
    start_token = int(i * tokens_per_chunk)
    end_token = int((i + 1) * tokens_per_chunk) if i < num_chunks - 1 else len(text_tokens)
    text_chunk = text_tokens[start_token:end_token]

    if len(text_chunk) > 0 and audio_chunk.shape[0] > 0:
      chunks.append((audio_chunk, text_chunk, i))

  return chunks


def main():
  """Test the trained model on the same chunks used during training"""

  # Paths
  audio_path = "/Users/amraboelela/develop/android/AndroidArabicWhisper/app/src/main/assets/001.wav"
  vocab_path = "vocabulary.json"
  model_path = "alfatiha_model_variable.pt"

  print("="*60)
  print("Testing on Training Chunks (1-second)")
  print("="*60)

  # Load vocabulary
  print("\nLoading vocabulary...")
  with open(vocab_path, "r", encoding="utf-8") as f:
    vocab = json.load(f)
  print(f"Vocabulary size: {len(vocab)}")

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

  # Load trained weights
  print(f"Loading model from: {model_path}")
  model.load_state_dict(torch.load(model_path))
  model.eval()
  model = model.to(device)

  # Extract audio features
  print(f"\nExtracting audio features...")
  audio_features, sample_rate = extract_mel_features(audio_path)
  print(f"Audio features: {audio_features.shape[0]} frames")

  # Prepare text
  alfatiha_text = "اعوذ بالله من الشيطان الرجيم بسم الله الرحمن الرحيم الحمد لله رب العالمين الرحمن الرحيم مالك يوم الدين اياك نعبد واياك نستعين اهدنا الصراط المستقيم"
  text_tokens = tokenize_text(alfatiha_text, vocab)

  # Split into same chunks as training
  print(f"\nSplitting into 1-second chunks (same as training)...")
  chunks = split_into_chunks(audio_features, text_tokens, fps=20)
  print(f"Total chunks: {len(chunks)}")

  # Test on all training chunks
  print("\n" + "="*60)
  print("Testing on Training Chunks:")
  print("="*60)

  all_generated_words = []
  all_expected_words = []
  correct = 0
  total = 0

  for audio_chunk, text_chunk, chunk_idx in chunks:
    with torch.no_grad():
      audio_batch = audio_chunk.unsqueeze(0).to(device)
      generated = model.generate(audio_batch, max_new_tokens=3, temperature=0.1)
      generated_ids = [idx for idx in generated[0].tolist() if idx not in [0, 1, 2]]
      generated_words = [vocab[idx] for idx in generated_ids]

    expected_words = [vocab[idx] for idx in text_chunk]

    # Check if matches
    match = "✓" if generated_words == expected_words else "✗"
    if generated_words == expected_words:
      correct += 1
    total += 1

    print(f"Chunk {chunk_idx+1:2d} ({chunk_idx:2d}-{chunk_idx+1:2d}s) {match}")
    print(f"  Expected:  {' '.join(expected_words)}")
    print(f"  Generated: {' '.join(generated_words)}")
    print()

    all_generated_words.extend(generated_words)
    all_expected_words.extend(expected_words)

  # Show full transcription
  print("="*60)
  print("Full Transcription:")
  print("="*60)
  print(f"Expected:  {' '.join(all_expected_words)}")
  print(f"Generated: {' '.join(all_generated_words)}")

  print("\n" + "="*60)
  print(f"Accuracy: {correct}/{total} = {correct/total*100:.1f}%")
  print("="*60)


if __name__ == "__main__":
  main()
