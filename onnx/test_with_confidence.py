#!/usr/bin/env python3
import json
import torch
import torch.nn.functional as F
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


def generate_with_confidence(model, audio_features, max_new_tokens=3, temperature=0.1, confidence_threshold=0.6):
  """Generate text with confidence scores"""
  model.eval()
  batch_size = audio_features.shape[0]
  text_ids = torch.ones((batch_size, 1), dtype=torch.long, device=audio_features.device)  # <s>

  generated_tokens = []
  confidences = []

  with torch.no_grad():
    for _ in range(max_new_tokens):
      logits = model.forward(audio_features=audio_features, text_ids=text_ids)
      logits = logits[:, -1, :] / temperature
      probs = F.softmax(logits, dim=-1)

      # Get top prediction and its confidence
      confidence, next_token = torch.max(probs, dim=-1)

      generated_tokens.append(next_token.item())
      confidences.append(confidence.item())

      # Stop if </s> or low confidence
      if next_token.item() == 2:  # </s>
        break

      text_ids = torch.cat([text_ids, next_token.unsqueeze(0)], dim=1)

  return generated_tokens, confidences


def main():
  """Test model chunk by chunk with confidence scores"""

  # Paths
  audio_path = "/Users/amraboelela/develop/android/AndroidArabicWhisper/app/src/main/assets/001.wav"
  vocab_path = "vocabulary.json"
  model_path = "alfatiha_model_variable.pt"

  print("="*70)
  print("Testing Chunks with Confidence Threshold (60%)")
  print("="*70)

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

  print(f"\n{'='*70}")
  print(f"Chunk-by-Chunk Results (Confidence Threshold: 60%)")
  print(f"{'='*70}\n")

  all_words = []
  confidence_threshold = 0.60

  for i in range(num_chunks):
    start_frame = i * frames_per_chunk
    end_frame = start_frame + frames_per_chunk
    audio_chunk = audio_features[start_frame:end_frame]

    audio_batch = audio_chunk.unsqueeze(0).to(device)
    generated_tokens, confidences = generate_with_confidence(
      model, audio_batch, max_new_tokens=3, temperature=0.1, confidence_threshold=confidence_threshold
    )

    # Filter out special tokens and low confidence
    words_with_conf = []
    for token_id, conf in zip(generated_tokens, confidences):
      if token_id not in [0, 1, 2]:  # Skip <unk>, <s>, </s>
        word = vocab[token_id]
        words_with_conf.append((word, conf))

    # Apply confidence threshold
    if words_with_conf and words_with_conf[0][1] >= confidence_threshold:
      word, conf = words_with_conf[0]
      all_words.append(word)
      status = "✓"
      result = f"{word} (conf: {conf:.1%})"
    else:
      if words_with_conf:
        word, conf = words_with_conf[0]
        status = "✗"
        result = f"<silent> (rejected: {word} at {conf:.1%})"
      else:
        status = "○"
        result = "<silent> (no output)"

    print(f"Chunk {i+1:2d} ({i:2d}-{i+1:2d}s) {status}  {result}")

  # Show final transcription
  print(f"\n{'='*70}")
  print("Final Transcription (after confidence filtering):")
  print(f"{'='*70}")
  print(f"Generated: {' '.join(all_words)}")

  print("\nExpected (Al-Fatiha):")
  expected = "اعوذ بالله من الشيطان الرجيم بسم الله الرحمن الرحيم الحمد لله رب العالمين الرحمن الرحيم مالك يوم الدين اياك نعبد واياك نستعين اهدنا الصراط المستقيم"
  print(f"{expected}")

  print(f"\n{'='*70}")
  print(f"Total words after filtering: {len(all_words)}")
  print(f"Expected words: 25")
  print(f"{'='*70}")


if __name__ == "__main__":
  main()
