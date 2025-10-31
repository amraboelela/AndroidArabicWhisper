#!/usr/bin/env python3
import json
import torch
import torchaudio
import onnxruntime
import numpy as np

def extract_mel_features(waveform, sample_rate, n_mels=800, target_fps=20):
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
    return mel_features

def test_onnx_model_chunked_inference():
    audio_path = "/Users/amraboelela/develop/android/AndroidArabicWhisper/app/src/main/assets/001.wav"
    vocab_path = "vocabulary.json"
    model_path = "alfatiha_model.onnx" # This is the FP32 ONNX model of the variable model

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    session = onnxruntime.InferenceSession(model_path)

    waveform, sample_rate = torchaudio.load(audio_path)
    
    chunk_size = 1 * sample_rate # 1 second chunks
    num_chunks = waveform.shape[1] // chunk_size

    for i in range(num_chunks):
        print(f"--- Chunk {i+1}/{num_chunks} ---")
        chunk = waveform[:, i*chunk_size : (i+1)*chunk_size]
        
        audio_features = extract_mel_features(chunk, sample_rate)
        audio_features = audio_features.unsqueeze(0).numpy() # FP32 input

        text_ids = np.array([[1]], dtype=np.int64) # Start with <s> token

        for _ in range(30): # max_new_tokens
            input_feed = {
                "audio_features": audio_features,
                "text_ids": text_ids
            }
            
            result = session.run(["logits"], input_feed)
            
            generated_logits = result[0]
            
            last_logits = generated_logits[:, -1, :]
            
            next_token_id = np.argmax(last_logits, axis=-1)
            
            text_ids = np.append(text_ids, [[next_token_id[0]]], axis=1)
            
            if next_token_id[0] == 2: # Stop if </s> token is generated
                break

        generated_tokens = text_ids[0].tolist()
        generated_words = [vocab[idx] if idx < len(vocab) else "<UNK>" for idx in generated_tokens]

        transcription = " ".join(generated_words)
        print(transcription)

if __name__ == "__main__":
    test_onnx_model_chunked_inference()