#!/usr/bin/env python3
"""
Verify encoder-decoder model configuration and debug issues
"""
import json
import torch
import torchaudio
from encoder_decoder_transformer import EncoderDecoderTransformer

# Load vocabulary
with open("vocabulary.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

print("="*60)
print("MODEL CONFIGURATION VERIFICATION")
print("="*60)

# 1. Check audio feature dimensions
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

audio_path = "segments/001-001.wav"
audio_features = extract_mel_features(audio_path)

print(f"\n1. Audio Feature Dimensions:")
print(f"   Shape: {audio_features.shape}")
print(f"   Expected: (num_frames, 800)")
print(f"   ✓ Matches!" if audio_features.shape[1] == 800 else f"   ✗ Mismatch! Got {audio_features.shape[1]}")
print(f"   Min value: {audio_features.min().item():.4f}")
print(f"   Max value: {audio_features.max().item():.4f}")
print(f"   Mean value: {audio_features.mean().item():.4f}")

# 2. Check max_seq_len
model = EncoderDecoderTransformer(
    vocab_size=len(vocab),
    d_model=512,
    n_encoder_layers=4,
    n_decoder_layers=4,
    n_heads=8,
    d_ff=2048,
    dropout=0.1
)

max_seq_len = 512  # from model default
max_audio_frames = audio_features.shape[0]
max_text_tokens = 20

print(f"\n2. Positional Embeddings:")
print(f"   max_seq_len: {max_seq_len}")
print(f"   Max audio frames: {max_audio_frames}")
print(f"   Max text tokens: {max_text_tokens}")
print(f"   Total needed: {max_audio_frames + max_text_tokens}")
print(f"   ✓ Sufficient!" if max_seq_len >= max_audio_frames + max_text_tokens else "   ✗ Insufficient!")

# 3. Check tokenization
test_text = "اعوذ بالله"
print(f"\n3. Tokenization Check:")
print(f"   Text: {test_text}")

word_to_idx = {word: idx for idx, word in enumerate(vocab)}
words = test_text.split()
token_ids = [word_to_idx.get(word, 0) for word in words]

print(f"   Words: {words}")
print(f"   Token IDs: {token_ids}")
print(f"   Decoded: {[vocab[idx] for idx in token_ids]}")

# Check for <unk> tokens
unk_count = sum(1 for idx in token_ids if idx == 0)
print(f"   <unk> tokens: {unk_count}/{len(token_ids)}")
if unk_count > 0:
    print(f"   ✗ WARNING: Some words mapped to <unk>")
else:
    print(f"   ✓ All words found in vocabulary")

# 4. Check special tokens
print(f"\n4. Special Tokens:")
print(f"   vocab[0]: {vocab[0]} (should be <unk>)")
print(f"   vocab[1]: {vocab[1]} (should be <s>)")
print(f"   vocab[2]: {vocab[2]} (should be </s>)")

# 5. Check labels format
input_tokens = [1] + token_ids  # <s> + text
target_tokens = token_ids + [2]  # text + </s>

print(f"\n5. Training Labels:")
print(f"   Input: {input_tokens} = {[vocab[idx] for idx in input_tokens]}")
print(f"   Target: {target_tokens} = {[vocab[idx] for idx in target_tokens]}")
print(f"   Contains -100: {-100 in target_tokens}")
print(f"   ✓ No ignore tokens" if -100 not in target_tokens else "   ✗ Has ignore tokens")

# 6. Test forward pass
model.load_state_dict(torch.load("encoder_decoder_model.pt"))
model.eval()

audio_batch = audio_features.unsqueeze(0)
input_ids = torch.tensor([input_tokens], dtype=torch.long)

print(f"\n6. Forward Pass Test:")
with torch.no_grad():
    logits = model(audio_features=audio_batch, text_ids=input_ids)

print(f"   Logits shape: {logits.shape}")
print(f"   Expected: (1, {len(input_tokens)}, {len(vocab)})")

# Check first prediction (after <s>)
first_logits = logits[0, 0, :]  # Prediction for first position
probs = torch.softmax(first_logits, dim=-1)
top_probs, top_indices = torch.topk(probs, 10)

print(f"\n   First token prediction (after <s>):")
for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
    word = vocab[idx.item()]
    marker = " ← EXPECTED" if idx.item() == token_ids[0] else ""
    print(f"     {i+1}. Token {idx.item():5d} ({word:20s}): {prob.item()*100:6.2f}%{marker}")

# 7. Test generation
print(f"\n7. Generation Test:")
with torch.no_grad():
    generated = model.generate(audio_batch, max_new_tokens=10, min_tokens=1)

generated_ids = generated[0].tolist()
print(f"   Generated IDs: {generated_ids}")
generated_words = [vocab[idx] for idx in generated_ids if idx != 1 and idx != 2]
print(f"   Generated text: {' '.join(generated_words)}")
print(f"   Expected text: {test_text}")

# 8. Check logits statistics
print(f"\n8. Logits Statistics:")
print(f"   Min: {logits.min().item():.4f}")
print(f"   Max: {logits.max().item():.4f}")
print(f"   Mean: {logits.mean().item():.4f}")
print(f"   Std: {logits.std().item():.4f}")

if abs(logits.mean().item()) > 10:
    print(f"   ⚠️ WARNING: Logits may be too large/small")

print(f"\n" + "="*60)
