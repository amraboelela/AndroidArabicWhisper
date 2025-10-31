#!/usr/bin/env python3
"""
Test the trained model on 002-01 segments
"""
import json
import torch
import torchaudio
from improved_transformer import ImprovedDecoderTransformer

# Use GPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 Using Metal GPU (Apple Silicon)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("🚀 Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("⚠️  Using CPU (slower)")

print(f"Device: {device}\n")


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

    return mel_features


def transcribe(model, audio_path, vocab, max_length=50):
    """Transcribe audio using the model"""
    model.eval()

    # Extract features
    audio_features = extract_mel_features(audio_path)
    audio_batch = audio_features.unsqueeze(0).to(device)

    # Create idx to word mapping
    idx_to_word = {idx: word for idx, word in enumerate(vocab)}

    # Start with <s> token (id=1)
    generated = [1]

    with torch.no_grad():
        for step in range(max_length):
            # Prepare input
            input_ids = torch.tensor([generated], dtype=torch.long, device=device)

            # Forward pass
            logits = model(
                audio_features=audio_batch,
                text_ids=input_ids,
                labels=None
            )

            # Get next token
            next_token = logits[0, -1, :].argmax().item()

            # Debug: print first few predictions
            if step < 5:
                print(f"    Step {step}: predicted token={next_token}, current_len={len(generated)}")

            # Stop if </s> token (id=2)
            if next_token == 2:
                break

            generated.append(next_token)

    # Convert to words (skip <s> token)
    words = [idx_to_word.get(idx, "<UNK>") for idx in generated[1:]]
    return " ".join(words)


def main():
    # Load vocabulary
    print("Loading vocabulary...")
    with open("vocabulary.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"✓ Vocabulary size: {len(vocab)}\n")

    # Load model
    print("Loading model...")
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
    print(f"✓ Model loaded\n")

    # Load expected transcriptions
    with open("segments/002-01.txt", "r", encoding="utf-8") as f:
        expected = [line.strip() for line in f.readlines()]

    # Test on a few segments
    test_indices = [0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60]

    print("="*80)
    print("Testing Model Transcription:")
    print("="*80)

    for idx in test_indices:
        if idx >= len(expected):
            continue

        segment_file = f"segments/002-01-{idx+1:03d}.wav"

        print(f"\nSegment {idx+1}:")
        print(f"  Audio: {segment_file}")

        # Transcribe
        transcription = transcribe(model, segment_file, vocab)

        print(f"  Expected:    {expected[idx]}")
        print(f"  Transcribed: {transcription}")

        # Calculate word-level accuracy
        expected_words = expected[idx].split()
        transcribed_words = transcription.split()

        if len(expected_words) > 0:
            matches = sum(1 for e, t in zip(expected_words, transcribed_words) if e == t)
            accuracy = matches / len(expected_words) * 100
            print(f"  Accuracy: {accuracy:.1f}% ({matches}/{len(expected_words)} words)")
        else:
            print(f"  Accuracy: N/A (empty expected)")


if __name__ == "__main__":
    main()
