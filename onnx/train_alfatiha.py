#!/usr/bin/env python3
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from quran_transformer import DecoderOnlyTransformer

def extract_mel_features(audio_path, n_mels=800, target_fps=10):
    """
    Extract mel spectrogram features from audio

    Args:
        audio_path: path to audio file
        n_mels: number of mel bins (should be 800)
        target_fps: target frames per second (10 fps)

    Returns:
        mel_features: (num_frames, n_mels) tensor
    """
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    print(f"Audio loaded: {waveform.shape}, sample_rate={sample_rate}")
    print(f"Duration: {waveform.shape[1] / sample_rate:.2f} seconds")

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

    print(f"Mel features extracted: {mel_features.shape}")
    print(f"Expected ~{int(waveform.shape[1] / sample_rate * target_fps)} frames at {target_fps} fps")

    return mel_features


def tokenize_text(text, vocab):
    """
    Tokenize text using word vocabulary

    Args:
        text: string of space-separated words
        vocab: list of vocabulary words

    Returns:
        token_ids: list of token indices
    """
    # Create word to index mapping
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}

    # Split text into words
    words = text.split()

    # Convert to token IDs
    token_ids = []
    for word in words:
        if word in word_to_idx:
            token_ids.append(word_to_idx[word])
        else:
            token_ids.append(0)  # <unk> token

    return token_ids


def prepare_alfatiha_data(audio_path, text_path, vocab):
    """
    Prepare Al-Fatiha audio and text data

    Args:
        audio_path: path to 001.wav
        text_path: path to 001.txt
        vocab: vocabulary list

    Returns:
        audio_features: (num_frames, 800) tensor
        text_tokens: list of token IDs
    """
    # Read Al-Fatiha text from file
    with open(text_path, "r", encoding="utf-8") as f:
        alfatiha_text = f.read().strip().replace('\n', ' ')

    print(f"\nAl-Fatiha text:")
    print(f"  {alfatiha_text}")

    # Extract audio features from 001.wav
    audio_features = extract_mel_features(audio_path)

    # Tokenize text
    text_tokens = tokenize_text(alfatiha_text, vocab)

    print(f"\nTokenized text:")
    print(f"  Tokens: {text_tokens}")
    print(f"  Words: {[vocab[idx] for idx in text_tokens]}")
    print(f"  Length: {len(text_tokens)} tokens")

    return audio_features, text_tokens


def train_on_alfatiha(model, audio_features, text_tokens, vocab, num_epochs=100, lr=1e-4):
    """
    Train the model on Al-Fatiha

    Args:
        model: DecoderOnlyTransformer
        audio_features: (num_frames, 800) tensor
        text_tokens: list of token IDs
        vocab: vocabulary list
        num_epochs: number of training epochs
        lr: learning rate
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Prepare data
    audio_features = audio_features.unsqueeze(0)  # (1, num_frames, 800)

    # Add <s> at beginning of text
    input_tokens = [1] + text_tokens  # <s> + text
    target_tokens = text_tokens + [2]  # text + </s>

    input_ids = torch.tensor([input_tokens], dtype=torch.long)
    labels = torch.tensor([target_tokens], dtype=torch.long)

    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"{'='*60}")
    print(f"Audio shape: {audio_features.shape}")
    print(f"Input text length: {len(input_tokens)}")
    print(f"Target text length: {len(target_tokens)}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {num_epochs}")
    print(f"Optimizer: AdamW")

    model.train()

    print(f"\n{'='*60}")
    print(f"Training Progress:")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass
        logits, loss = model(
            audio_features=audio_features,
            text_ids=input_ids,
            labels=labels
        )

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs}: Loss = {loss.item():.4f}")

        # Generate sample every 20 epochs
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                generated = model.generate(audio_features, max_new_tokens=len(text_tokens)+2)
                generated_words = [vocab[idx] for idx in generated[0].tolist()]
                print(f"  Generated: {' '.join(generated_words[:10])}...")
            model.train()

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")

    return model


def test_model(model, audio_features, text_tokens, vocab):
    """
    Test the trained model

    Args:
        model: trained model
        audio_features: audio features
        text_tokens: expected text tokens
        vocab: vocabulary
    """
    print(f"\n{'='*60}")
    print(f"Testing Trained Model:")
    print(f"{'='*60}")

    model.eval()
    audio_features = audio_features.unsqueeze(0)

    with torch.no_grad():
        generated = model.generate(audio_features, max_new_tokens=len(text_tokens)+5, temperature=0.1)

    generated_ids = generated[0].tolist()
    generated_words = [vocab[idx] for idx in generated_ids]

    expected_words = [vocab[idx] for idx in text_tokens]

    print(f"\nExpected text:")
    print(f"  {' '.join(expected_words)}")

    print(f"\nGenerated text:")
    print(f"  {' '.join(generated_words)}")

    print(f"\nToken comparison:")
    print(f"  Expected IDs:  {text_tokens}")
    print(f"  Generated IDs: {generated_ids[1:]}")  # Skip <s> token

    # Calculate accuracy
    min_len = min(len(text_tokens), len(generated_ids) - 1)
    correct = sum(1 for i in range(min_len) if text_tokens[i] == generated_ids[i+1])
    accuracy = correct / len(text_tokens) * 100 if text_tokens else 0

    print(f"\nAccuracy: {correct}/{len(text_tokens)} tokens ({accuracy:.1f}%)")


def main():
    """Main training function"""

    # Paths
    audio_path = "/Users/amraboelela/develop/android/AndroidArabicWhisper/app/src/main/assets/001.wav"
    text_path = "segments/001.txt"
    vocab_path = "vocabulary.json"

    # Load vocabulary
    print("Loading vocabulary...")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    print(f"Vocabulary size: {len(vocab)}")

    # Create model
    print("\nCreating model...")
    model = DecoderOnlyTransformer(vocab_size=len(vocab), d_model=800)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Prepare data
    print(f"\n{'='*60}")
    print(f"Preparing Al-Fatiha Data:")
    print(f"{'='*60}")
    audio_features, text_tokens = prepare_alfatiha_data(audio_path, text_path, vocab)

    # Train model
    model = train_on_alfatiha(
        model,
        audio_features,
        text_tokens,
        vocab,
        num_epochs=200,
        lr=1e-4
    )

    # Test model
    test_model(model, audio_features, text_tokens, vocab)

    # Save model
    save_path = "alfatiha_model.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\n{'='*60}")
    print(f"Model saved to: {save_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
