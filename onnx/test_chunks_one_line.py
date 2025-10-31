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
  """Test model chunk by chunk"""

  # Paths
  audio_path = "/Users/amraboelela/develop/android/AndroidArabicWhisper/app/src/main/assets/001.wav"
  vocab_path = "vocabulary.json"
  model_path = "alfatiha_model_variable.pt"

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

  # Test on all 1-second chunks
  fps = 20
  frames_per_chunk = fps  # 1 second = 20 frames
  total_frames = audio_features.shape[0]
  num_chunks = total_frames // frames_per_chunk

  previous_transcription = ""
  all_words = []

  for i in range(num_chunks):
    start_frame = i * frames_per_chunk
    end_frame = start_frame + frames_per_chunk
    audio_chunk = audio_features[start_frame:end_frame]

    audio_batch = audio_chunk.unsqueeze(0).to(device)

    with torch.no_grad():
      generated = model.generate(audio_batch, max_new_tokens=3, temperature=0.1)
      generated_ids = [idx for idx in generated[0].tolist() if idx not in [0, 1, 2]]
      generated_words = [vocab[idx] for idx in generated_ids]

    transcription = ' '.join(generated_words)

    if transcription == previous_transcription:
      words_str = ""
    elif generated_words:
      words_str = transcription
      all_words.append(words_str)
    else:
      words_str = "<silence>"

    previous_transcription = transcription

    print(f"Chunk {i+1}: {words_str}")

  # Print final concatenated text
  print(f"\n{'='*60}")
  print("Final Transcription:")
  print(f"{'='*60}")
  print(' '.join(all_words))


if __name__ == "__main__":
  main()
