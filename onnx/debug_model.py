#!/usr/bin/env python3
"""
Debug model output to see why it always predicts token 2
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

# Test on first segment
audio_path = "segments/001-001.wav"
audio_features = extract_mel_features(audio_path)
audio_batch = audio_features.unsqueeze(0).to(device)

print(f"Testing: {audio_path}")
print(f"Audio features shape: {audio_features.shape}")

# Start with <s> token (id=1)
input_ids = torch.tensor([[1]], dtype=torch.long, device=device)

with torch.no_grad():
    logits = model(audio_features=audio_batch, text_ids=input_ids, labels=None)

    print(f"\nLogits shape: {logits.shape}")
    print(f"Logits for last position shape: {logits[0, -1, :].shape}")

    # Get predictions for the last position
    last_logits = logits[0, -1, :]

    # Show top 10 predictions
    probs = torch.softmax(last_logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, 10)

    print(f"\nTop 10 predictions:")
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        word = vocab[idx.item()] if idx.item() < len(vocab) else "<UNK>"
        print(f"  {i+1}. Token {idx.item():5d} ({word:20s}): {prob.item()*100:6.2f}%")

    # Check token 2 specifically
    token_2_prob = probs[2].item()
    print(f"\nToken 2 (end token) probability: {token_2_prob*100:.2f}%")

    # Show some statistics
    print(f"\nLogits statistics:")
    print(f"  Min: {last_logits.min().item():.4f}")
    print(f"  Max: {last_logits.max().item():.4f}")
    print(f"  Mean: {last_logits.mean().item():.4f}")
    print(f"  Std: {last_logits.std().item():.4f}")
