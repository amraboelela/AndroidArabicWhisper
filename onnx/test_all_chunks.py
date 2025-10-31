#!/usr/bin/env python3
import json
import torch
import torchaudio
from improved_transformer import ImprovedDecoderTransformer

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
  """Test model on all 1-second chunks"""

  # Paths
  audio_path = "/Users/amraboelela/develop/android/AndroidArabicWhisper/app/src/main/assets/001.wav"
  vocab_path = "vocabulary.json"
  model_path = "alfatiha_model_variable.pt"

  print("="*60)
  print("Testing All 1-Second Chunks")
  print("="*60)

  # Load vocabulary
  with open(vocab_path, "r", encoding="utf-8") as f:
    vocab = json.load(f)

  # Create and load model
  model = ImprovedDecoderTransformer(
    vocab_size=len(vocab),
    d_model=800,
    n_layers=5,
    n_heads=10,
    d_ff=3200,
    dropout=0.1
  )
  model.load_state_dict(torch.load(model_path))
  model.eval()
  model = model.to(device)

  # Extract audio features
  audio_features, sample_rate = extract_mel_features(audio_path)
  print(f"\nTotal audio: {audio_features.shape[0]} frames ({audio_features.shape[0]/20:.1f} seconds)")

  # Test on all 1-second chunks
  fps = 20
  frames_per_chunk = fps  # 1 second = 20 frames
  total_frames = audio_features.shape[0]
  num_chunks = total_frames // frames_per_chunk

  print(f"\n{'='*60}")
  print(f"Testing {num_chunks} one-second chunks:")
  print(f"{'='*60}\n")

  all_words = []

  for i in range(num_chunks):
    start_frame = i * frames_per_chunk
    end_frame = start_frame + frames_per_chunk
    audio_chunk = audio_features[start_frame:end_frame]

    with torch.no_grad():
      audio_batch = audio_chunk.unsqueeze(0).to(device)
      generated = model.generate(audio_batch, max_new_tokens=3, temperature=0.1)
      generated_ids = [idx for idx in generated[0].tolist() if idx not in [0, 1, 2]]
      generated_words = [vocab[idx] for idx in generated_ids]

    if generated_words:
      all_words.extend(generated_words)
      words_str = ' '.join(generated_words)
    else:
      words_str = "<empty>"

    print(f"Chunk {i+1:2d} ({i:2d}-{i+1:2d}s): {words_str}")

  # Show full transcription
  print(f"\n{'='*60}")
  print("Full Transcription:")
  print(f"{'='*60}")
  print(f"Generated: {' '.join(all_words)}")

  print("\nExpected (Al-Fatiha):")
  expected = "اعوذ بالله من الشيطان الرجيم بسم الله الرحمن الرحيم الحمد لله رب العالمين الرحمن الرحيم مالك يوم الدين اياك نعبد واياك نستعين اهدنا الصراط المستقيم"
  print(f"{expected}")

  print(f"\n{'='*60}")
  print(f"Total words generated: {len(all_words)}")
  print(f"Expected words: 25")
  print(f"{'='*60}")


if __name__ == "__main__":
  main()
