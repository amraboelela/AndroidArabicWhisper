#!/usr/bin/env python3
import json
import torch
import torchaudio
from quran_transformer import DecoderOnlyTransformer

def extract_mel_features(audio_path, n_mels=1600, target_fps=10):
    """Extract mel spectrogram features from audio"""
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Calculate hop length for target fps
    hop_length = sample_rate // target_fps
    n_fft = 2048

    # Create mel spectrogram transform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=0,
        f_max=sample_rate // 2
    )

    # Extract mel spectrogram
    mel_spec = mel_transform(waveform)  # (1, n_mels, time)

    # Convert to log scale
    mel_spec = torch.log(mel_spec + 1e-9)

    # Transpose to (time, n_mels)
    mel_features = mel_spec.squeeze(0).transpose(0, 1)

    return mel_features


def test_model_inference():
    """Test the trained model on 001.wav"""

    # Paths
    audio_path = "/Users/amraboelela/develop/android/AndroidArabicWhisper/app/src/main/assets/001.wav"
    vocab_path = "vocabulary.json"
    model_path = "alfatiha_model.pt"

    print("="*60)
    print("Testing Trained Model on 001.wav (Al-Fatiha)")
    print("="*60)

    # Load vocabulary
    print("\n1. Loading vocabulary...")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"   Vocabulary size: {len(vocab)}")

    # Create model
    print("\n2. Creating model architecture...")
    model = DecoderOnlyTransformer(vocab_size=len(vocab), d_model=1600)

    # Load trained weights
    print(f"\n3. Loading trained weights from {model_path}...")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("   Model loaded successfully!")

    # Extract audio features
    print(f"\n4. Extracting audio features from 001.wav...")
    audio_features = extract_mel_features(audio_path)
    print(f"   Audio features: {audio_features.shape}")
    print(f"   Duration: {audio_features.shape[0] / 10:.2f} seconds")

    # Generate text
    print("\n5. Generating transcription...")
    audio_features = audio_features.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        generated_ids = model.generate(
            audio_features,
            max_new_tokens=30,
            temperature=0.1  # Lower temperature for more deterministic output
        )

    # Convert IDs to words
    generated_tokens = generated_ids[0].tolist()
    generated_words = [vocab[idx] for idx in generated_tokens]

    # Display results
    print("\n" + "="*60)
    print("TRANSCRIPTION RESULTS")
    print("="*60)

    print("\nGenerated Token IDs:")
    print(f"  {generated_tokens}")

    print("\nGenerated Text:")
    transcription = " ".join(generated_words)
    print(f"  {transcription}")

    print("\nExpected Text (Al-Fatiha):")
    expected = "اعوذ بالله من الشيطان الرجيم بسم الله الرحمن الرحيم الحمد لله رب العالمين الرحمن الرحيم مالك يوم الدين اياك نعبد واياك نستعين اهدنا الصراط المستقيم"
    print(f"  {expected}")

    print("\n" + "="*60)
    print("Analysis:")
    print("="*60)
    print(f"Total tokens generated: {len(generated_tokens)}")
    print(f"Unique words in output: {len(set(generated_words))}")

    # Count occurrences of key words
    key_words = ["الله", "الرحمن", "الرحيم", "بسم", "الحمد"]
    print("\nKey word occurrences:")
    for word in key_words:
        count = generated_words.count(word)
        print(f"  '{word}': {count} times")


if __name__ == "__main__":
    test_model_inference()
