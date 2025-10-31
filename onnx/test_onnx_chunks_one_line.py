#!/usr/bin/env python3
import json
import torch
import torchaudio
import onnxruntime
import numpy as np

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

  return mel_features.numpy(), sample_rate


def main():
  """Test model chunk by chunk"""

  # Paths
  audio_path = "/Users/amraboelela/develop/android/AndroidArabicWhisper/app/src/main/assets/001.wav"
  vocab_path = "vocabulary.json"
  model_path = "alfatiha_model.onnx" # FP32 ONNX version of the variable model

  # Load vocabulary
  with open(vocab_path, "r", encoding="utf-8") as f:
    vocab = json.load(f)

  # Create and load ONNX session
  session = onnxruntime.InferenceSession(model_path)

  # Extract audio features
  audio_features, sample_rate = extract_mel_features(audio_path)

  # Test on all 1-second chunks
  fps = 20
  frames_per_chunk = fps  # 1 second = 20 frames
  total_frames = audio_features.shape[0]
  num_chunks = total_frames // frames_per_chunk

  previous_transcription = ""

  for i in range(num_chunks):
    start_frame = i * frames_per_chunk
    end_frame = start_frame + frames_per_chunk
    audio_chunk = audio_features[start_frame:end_frame]

    audio_batch = np.expand_dims(audio_chunk, axis=0) # Add batch dimension

    input_feed = {
        "audio_features": audio_batch,
        "text_ids": np.array([[1]], dtype=np.int64) # Start with <s> token
    }
    
    generated_ids_chunk = []
    max_new_tokens = 3 # Same as in test_chunks_one_line.py

    for _ in range(max_new_tokens):
        result = session.run(["logits"], input_feed)
        generated_logits = result[0]
        last_logits = generated_logits[:, -1, :]
        next_token_id = np.argmax(last_logits, axis=-1)
        
        input_feed["text_ids"] = np.append(input_feed["text_ids"], [[next_token_id[0]]], axis=1)
        
        if next_token_id[0] == 2: # Stop if </s> token is generated
            break
    
    generated_tokens = input_feed["text_ids"][0].tolist()
    generated_ids = [idx for idx in generated_tokens if idx not in [0, 1, 2]]
    generated_words = [vocab[idx] if idx < len(vocab) else "<UNK>" for idx in generated_ids]

    transcription = ' '.join(generated_words)

    if transcription == previous_transcription:
      words_str = ""
    elif generated_words:
      words_str = transcription
    else:
      words_str = "<silence>"

    previous_transcription = transcription

    print(f"Chunk {i+1}: {words_str}")


if __name__ == "__main__":
  main()
