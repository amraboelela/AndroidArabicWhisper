#!/usr/bin/env python3
import json
import torch
import torchaudio
from improved_transformer import ImprovedDecoderTransformer

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

def test_pytorch_chunked_inference():
    audio_path = "/Users/amraboelela/develop/android/AndroidArabicWhisper/app/src/main/assets/001.wav"
    vocab_path = "vocabulary.json"
    model_path = "alfatiha_model_variable.pt"

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

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

    waveform, sample_rate = torchaudio.load(audio_path)
    
    chunk_size = 1 * sample_rate # 1 second chunks
    num_chunks = waveform.shape[1] // chunk_size

    for i in range(num_chunks):
        print(f"--- Chunk {i+1}/{num_chunks} ---")
        chunk = waveform[:, i*chunk_size : (i+1)*chunk_size]
        
        audio_features = extract_mel_features(chunk, sample_rate)
        audio_features = audio_features.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            generated_ids = model.generate(
                audio_features,
                max_new_tokens=30,
                temperature=1.0
            )

        generated_tokens = generated_ids[0].tolist()
        generated_words = [vocab[idx] for idx in generated_tokens]

        transcription = " ".join(generated_words)
        print(transcription)

if __name__ == "__main__":
    test_pytorch_chunked_inference()