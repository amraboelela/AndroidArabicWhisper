#!/usr/bin/env python3
"""
Test the trained model on Al-Fatiha segments
"""
import json
import torch
import torchaudio
from improved_transformer import ImprovedDecoderTransformer

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}\n")

# Load vocabulary
with open("vocabulary.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

# Load model
model = ImprovedDecoderTransformer(
    vocab_size=len(vocab),
    d_model=800,
    n_layers=5,
    n_heads=10,
    d_ff=3200,
    dropout=0.1
)
model.load_state_dict(torch.load("quran_model.pt", map_location=device))
model = model.to(device)
model.eval()

# Extract mel features
def extract_mel_features(audio_path, n_mels=800, target_fps=20):
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

    return mel_features

# Load expected transcriptions
with open("segments/001.txt", "r", encoding="utf-8") as f:
    expected = [line.strip() for line in f.readlines()]

# Test on all 8 segments
print("="*80)
print("Testing Al-Fatiha Model:")
print("="*80)

idx_to_word = {idx: word for idx, word in enumerate(vocab)}

for idx in range(min(len(expected), 8)):
    segment_file = f"segments/001-{idx+1:03d}.wav"

    print(f"\nSegment {idx+1}:")
    print(f"  Audio: {segment_file}")

    # Extract features
    audio_features = extract_mel_features(segment_file)
    audio_batch = audio_features.unsqueeze(0).to(device)

    # Start with <s> token (id=1)
    generated = [1]

    with torch.no_grad():
        for step in range(50):  # max 50 tokens
            input_ids = torch.tensor([generated], dtype=torch.long, device=device)

            # Forward pass
            logits = model(audio_features=audio_batch, text_ids=input_ids, labels=None)

            # Get next token
            next_token = logits[0, -1, :].argmax().item()

            # Stop if </s> token (id=2)
            if next_token == 2:
                break

            generated.append(next_token)

    # Convert to words (skip <s> token)
    words = [idx_to_word.get(idx, "<UNK>") for idx in generated[1:]]
    transcription = " ".join(words)

    print(f"  Expected:    {expected[idx]}")
    print(f"  Transcribed: {transcription}")

    # Calculate word-level accuracy
    expected_words = expected[idx].split()
    transcribed_words = transcription.split()

    if len(expected_words) > 0:
        matches = sum(1 for e, t in zip(expected_words, transcribed_words) if e == t)
        accuracy = matches / len(expected_words) * 100
        print(f"  Accuracy: {accuracy:.1f}% ({matches}/{len(expected_words)} words)")
