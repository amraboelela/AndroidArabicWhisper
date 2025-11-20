#!/usr/bin/env python3
"""
Train on ALL segments with pitch augmentation (±2 semitones)
Usage: python3 train_all_full_augmented.py

This script trains on the entire Quran-A dataset with pitch augmentation,
randomly pitch-shifting audio by ±2 semitones to make the model more robust
to pitch variations across different reciters.
"""
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Force unbuffered output for real-time logging
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import json
import torch
import torch.nn as nn
import torchaudio
import glob
import os
import random
import time
sys.path.append("..")
from tools.encoder_decoder_transformer import EncoderDecoderTransformer

# ==============================================================
# Device setup
# ==============================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# ==============================================================
# Pitch augmentation
# ==============================================================
def pitch_shift_audio(waveform, sample_rate, n_steps):
    """
    Pitch shift audio by n_steps semitones using resampling

    Args:
        waveform: (1, samples) audio tensor
        sample_rate: original sample rate
        n_steps: number of semitones to shift (+/- 2)

    Returns:
        pitch_shifted waveform at original sample rate
    """
    if n_steps == 0:
        return waveform

    # Calculate pitch shift ratio: 2^(n_steps/12)
    shift_ratio = 2.0 ** (n_steps / 12.0)

    # Resample to shift pitch
    # To shift pitch up, we speed up (higher rate), then resample back
    # To shift pitch down, we slow down (lower rate), then resample back
    shifted_rate = int(sample_rate * shift_ratio)

    # First resample: change speed (and pitch)
    resampler_shift = torchaudio.transforms.Resample(
        orig_freq=sample_rate,
        new_freq=shifted_rate
    )
    shifted = resampler_shift(waveform)

    # Second resample: bring back to original rate (preserves pitch shift)
    resampler_restore = torchaudio.transforms.Resample(
        orig_freq=shifted_rate,
        new_freq=sample_rate
    )
    restored = resampler_restore(shifted)

    # Ensure same length as original (trim or pad)
    if restored.shape[1] > waveform.shape[1]:
        restored = restored[:, :waveform.shape[1]]
    elif restored.shape[1] < waveform.shape[1]:
        padding = waveform.shape[1] - restored.shape[1]
        restored = torch.nn.functional.pad(restored, (0, padding))

    return restored

# ==============================================================
# Audio feature extraction with augmentation
# ==============================================================
def load_and_augment_audio(audio_path, augment=True):
    """
    Load audio from mic folder and optionally apply pitch augmentation

    Args:
        audio_path: path to .pt mel file (we derive .wav path from it)
        augment: whether to apply random pitch shift

    Returns:
        waveform, sample_rate
    """
    # Derive audio path from mel path: mels/ -> audio/mic/ and .pt -> .wav
    audio_wav_path = audio_path.replace('/mels/', '/audio/mic/').replace('.pt', '.wav')

    if not os.path.exists(audio_wav_path):
        raise FileNotFoundError(f"Audio file not found: {audio_wav_path}")

    waveform, sample_rate = torchaudio.load(audio_wav_path)

    # Convert stereo to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Apply pitch augmentation during training
    if augment:
        # Random pitch shift: -2, -1, 0, +1, or +2 semitones
        n_steps = random.randint(-2, 2)
        if n_steps != 0:
            waveform = pitch_shift_audio(waveform, sample_rate, n_steps)

    return waveform, sample_rate

def extract_mel_features_from_waveform(waveform, sample_rate, n_mels=40):
    """
    Extract mel features from waveform using Whisper-compatible settings

    Args:
        waveform: (1, samples) audio tensor
        sample_rate: audio sample rate
        n_mels: number of mel filterbanks (40)

    Returns:
        mel_features: (time, n_mels) tensor
    """
    # Resample to 16kHz (Whisper standard)
    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=target_sample_rate
        )
        waveform = resampler(waveform)
        sample_rate = target_sample_rate

    # Whisper parameters (100 fps: 16000 / 160 = 100)
    n_fft = 400
    hop_length = 160

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

    # Per-segment normalization (same as generate_mels.py)
    mel_mean = mel_features.mean()
    mel_std = mel_features.std()
    mel_features = (mel_features - mel_mean) / (mel_std + 1e-8)

    return mel_features

def load_mel_features(mel_path):
    """Load precomputed mel features from .pt file (for non-augmented baseline)"""
    if not os.path.exists(mel_path):
        raise FileNotFoundError(f"Precomputed mel features not found: {mel_path}")

    mel_features = torch.load(mel_path, map_location='cpu', weights_only=True)
    return mel_features

# ==============================================================
# Tokenization
# ==============================================================
def tokenize_text(text, vocab):
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    words = text.split()
    return [word_to_idx.get(word, 0) for word in words]

# ==============================================================
# Text normalization
# ==============================================================
def normalize_text(text):
    """Normalize Arabic text by removing diacritics and extra spacing"""
    normalized = text.replace("َ", "").replace("ً", "").replace("ُ", "").replace("ِ", "")
    normalized = normalized.replace("ّ", "").replace("ْ", "").replace("ٌ", "").replace("ٍ", "")
    return " ".join(normalized.split())

# ==============================================================
# Calculate accuracy
# ==============================================================
def calculate_accuracy(model, segment_files, transcriptions, vocab, device):
    """Calculate overall accuracy (without augmentation)"""
    model.eval()
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for seg_file, expected_text in zip(segment_files, transcriptions):
            # Load precomputed mel features (no augmentation for testing)
            mel_features = load_mel_features(seg_file)
            audio_batch = mel_features.transpose(0, 1).unsqueeze(0).to(device)

            # Get audio duration from original audio file
            audio_wav_path = seg_file.replace('/mels/', '/audio/mic/').replace('.pt', '.wav')
            waveform, sr = torchaudio.load(audio_wav_path)
            audio_duration = waveform.shape[1] / sr

            # Generate
            generated = model.generate(
                audio_batch,
                max_new_tokens=50,
                temperature=1.0,
                min_tokens=1,
                use_sampling=False,
                audio_duration_seconds=audio_duration
            )
            tokens = generated[0].tolist()

            # Clean tokens
            if tokens and tokens[0] == 1:
                tokens = tokens[1:]
            if 2 in tokens:
                tokens = tokens[:tokens.index(2)]

            generated_words = [vocab[idx] for idx in tokens if idx < len(vocab)]
            generated_text = " ".join(generated_words)

            # Token-level accuracy
            expected_words = expected_text.split()
            min_len = min(len(expected_words), len(generated_words))
            total_correct += sum(1 for i in range(min_len) if generated_words[i] == expected_words[i])
            total_tokens += len(expected_words)

    accuracy = (total_correct / total_tokens * 100) if total_tokens > 0 else 0.0
    return accuracy

# ==============================================================
# Main training
# ==============================================================
def main():
    dataset_name = "Quran-A"

    print(f"\n{'='*60}")
    print(f"TRAINING WITH PITCH AUGMENTATION (±2 semitones)")
    print(f"DATASET: {dataset_name} (entire dataset)")
    print(f"{'='*60}\n")

    # Paths
    vocab_path = "../models/vocabulary.json"
    model_path = "../models/muhaffez_whisper.pt"
    datasets_dir = f"../datasets/{dataset_name}"

    # Load vocabulary
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"Vocabulary size: {len(vocab)}")

    # Find ALL text files in dataset
    text_files = sorted(glob.glob(f"{datasets_dir}/text/*.txt"))
    if not text_files:
        print(f"❌ No text files found in {datasets_dir}/text/")
        sys.exit(1)

    # Collect all segments from all surah parts
    all_segment_files = []
    all_transcriptions = []

    for text_file in text_files:
        surah_part = os.path.splitext(os.path.basename(text_file))[0]
        surah_num = surah_part.split('-')[0]
        mels_dir = f"{datasets_dir}/mels/{surah_num}"

        # Load transcriptions
        with open(text_file, "r", encoding="utf-8") as f:
            transcriptions = [line.strip() for line in f if line.strip()]

        # Find mel feature files
        if '-' in surah_part and len(surah_part.split('-')) > 1 and surah_part.split('-')[1]:
            mel_files = sorted(glob.glob(f"{mels_dir}/{surah_part}/{surah_part}-*.pt"))
        else:
            mel_files = sorted(glob.glob(f"{mels_dir}/{surah_part}-*.pt"))

        if not mel_files:
            mel_files = sorted(glob.glob(f"{mels_dir}/{surah_part}/{surah_part}-*.pt"))

        if len(transcriptions) != len(mel_files):
            print(f"⚠️  Warning: Mismatch in {surah_part}: {len(transcriptions)} texts vs {len(mel_files)} mel files")
            continue

        all_segment_files.extend(mel_files)
        all_transcriptions.extend(transcriptions)
        print(f"  Loaded {len(mel_files)} segments from {surah_part}")

    total_segments = len(all_segment_files)
    print(f"\n✓ Total segments: {total_segments}")
    print(f"✓ Augmentation: Random pitch shift ±2 semitones per segment")

    # Initialize or load model
    model = EncoderDecoderTransformer(
        vocab_size=len(vocab),
        d_model=128,
        n_encoder_layers=4,
        n_decoder_layers=4,
        n_heads=4,
        d_ff=512,
        dropout=0.1,
        n_mels=40
    )

    if os.path.exists(model_path):
        print(f"\nLoading existing model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"✓ Model loaded successfully! Continuing training.")
    else:
        print(f"\n⚠️  No existing model found. Starting from scratch.")

    model = model.to(device)

    # Training setup
    learning_rate = 1e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

    print(f"\nStarting training for up to 5 epochs on {total_segments} segments...")
    print(f"Initial Learning Rate: {learning_rate:.1e}")

    # Calculate initial accuracy
    initial_acc = calculate_accuracy(model, all_segment_files, all_transcriptions, vocab, device)
    print(f"Initial accuracy: {initial_acc:.1f}%")

    if initial_acc >= 95.0:
        print(f"\n✓ Model already at {initial_acc:.1f}% accuracy. Skipping training.")
    else:
        # Training loop
        best_loss = float('inf')
        prev_loss = float('inf')
        start_time = time.time()

        for epoch in range(5):  # Max 5 epochs for augmented training
            model.train()
            total_loss = 0.0
            total_iterations = 0

            # Shuffle segments
            indices = list(range(len(all_segment_files)))
            random.shuffle(indices)

            for i in indices:
                seg_file = all_segment_files[i]
                text = all_transcriptions[i]

                # Load audio and apply pitch augmentation
                waveform, sample_rate = load_and_augment_audio(seg_file, augment=True)

                # Extract mel features from augmented audio
                mel_features = extract_mel_features_from_waveform(waveform, sample_rate, n_mels=40)
                audio_batch = mel_features.transpose(0, 1).unsqueeze(0).to(device)

                # Tokenize
                text_tokens = tokenize_text(text, vocab)
                full_sequence = [1] + text_tokens + [2]
                input_ids = torch.tensor([full_sequence[:-1]], dtype=torch.long, device=device)
                labels = torch.tensor([full_sequence[1:]], dtype=torch.long, device=device)

                # Forward
                logits = model(mel_features=audio_batch, text_ids=input_ids)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                total_iterations += 1

            avg_loss = total_loss / total_iterations
            elapsed = time.time() - start_time

            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), model_path)

            # Early stopping if loss plateaus (change < 0.001)
            if abs(avg_loss - prev_loss) < 0.001 and epoch > 0:
                print(f"  Loss plateaued (change < 0.001). Early stopping.")
                break

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1} | Loss={avg_loss:.4f} | LR={current_lr:.1e} | Time={elapsed:.0f}s", flush=True)

            # Update prev_loss
            prev_loss = avg_loss

            # Check accuracy every epoch
            current_acc = calculate_accuracy(model, all_segment_files, all_transcriptions, vocab, device)
            print(f"Accuracy: {current_acc:.1f}%", flush=True)

            if current_acc >= 95.0:
                print(f"✓ Early stopping: accuracy reached 95%", flush=True)
                break

    # Save final model
    torch.save(model.state_dict(), model_path)
    print(f"\nFinal model saved to: {model_path}")

    # Calculate and output final accuracy
    final_acc = calculate_accuracy(model, all_segment_files, all_transcriptions, vocab, device)
    print(f"FINAL_ACCURACY: {final_acc:.0f}%")


if __name__ == "__main__":
    main()
