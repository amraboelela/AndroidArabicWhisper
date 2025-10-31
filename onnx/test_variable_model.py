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


def main():
  """Test the trained model on full audio"""

  # Paths
  audio_path = "/Users/amraboelela/develop/android/AndroidArabicWhisper/app/src/main/assets/001.wav"
  vocab_path = "vocabulary.json"
  model_path = "alfatiha_model_variable.pt"

  print("="*60)
  print("Testing Trained Model on 001.wav (Al-Fatiha)")
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
  print("Model loaded successfully!")

  # Extract audio features
  print(f"\nExtracting audio features from 001.wav...")
  audio_features, sample_rate = extract_mel_features(audio_path)
  print(f"Audio features: {audio_features.shape}")
  print(f"Duration: {audio_features.shape[0] / 20:.2f} seconds")

  # Test on 1-second chunks
  print("\n" + "="*60)
  print("Testing on 1-second chunks:")
  print("="*60)

  fps = 20
  chunk_duration = 1
  frames_per_chunk = chunk_duration * fps
  total_frames = audio_features.shape[0]
  num_chunks = total_frames // frames_per_chunk

  all_generated_words = []

  for i in range(min(num_chunks, 10)):  # Test first 10 chunks
    start_frame = i * frames_per_chunk
    end_frame = start_frame + frames_per_chunk
    audio_chunk = audio_features[start_frame:end_frame]

    with torch.no_grad():
      audio_batch = audio_chunk.unsqueeze(0).to(device)
      generated = model.generate(audio_batch, max_new_tokens=5, temperature=0.1)
      generated_words = [vocab[idx] for idx in generated[0].tolist() if idx not in [0, 1, 2]]

    if generated_words:
      all_generated_words.extend(generated_words)
      print(f"Chunk {i+1} ({i}-{i+1}s): {' '.join(generated_words)}")

  # Show full transcription
  print("\n" + "="*60)
  print("Full Transcription (first 10 seconds):")
  print("="*60)
  print(f"Generated: {' '.join(all_generated_words)}")

  print("\nExpected (Al-Fatiha):")
  expected = "اعوذ بالله من الشيطان الرجيم بسم الله الرحمن الرحيم الحمد لله رب العالمين الرحمن الرحيم مالك يوم الدين اياك نعبد واياك نستعين اهدنا الصراط المستقيم"
  print(f"{expected}")

  print("\n" + "="*60)


if __name__ == "__main__":
  main()
