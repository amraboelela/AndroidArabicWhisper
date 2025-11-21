#!/usr/bin/env python3
"""
Train on ALL segments including ALL augmented variations (pitch + speed)
Implements custom learning rate decay: start at 1e-3, decay by 10% when loss increases, stop at 1e-7

Usage: python3 train_all_augmented.py <dataset_name>
Example:
  python3 train_all_augmented.py Quran-A
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
# Audio feature extraction
# ==============================================================
def load_mel_features(mel_path):
    """Load precomputed mel features from .pt file"""
    if not os.path.exists(mel_path):
        raise FileNotFoundError(f"Precomputed mel features not found: {mel_path}\nPlease run generate_mels.py first")

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
    """Calculate overall accuracy on regular (non-augmented) segments only"""
    model.eval()
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for seg_file, expected_text in zip(segment_files, transcriptions):
            # Load precomputed mel features
            mel_features = load_mel_features(seg_file)
            audio_batch = mel_features.transpose(0, 1).unsqueeze(0).to(device)

            # Calculate audio duration from mel spectrogram
            # For 8kHz audio with hop_length=80: duration = (time_frames * hop_length) / sample_rate
            time_frames = mel_features.shape[0]
            sample_rate = 8000
            hop_length = 80
            audio_duration = (time_frames * hop_length) / sample_rate

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

            # Token-level accuracy (word-by-word comparison)
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
    if len(sys.argv) < 2:
        print("Usage: python3 train_all_augmented.py <dataset_name>")
        print("Example:")
        print("  python3 train_all_augmented.py Quran-A")
        sys.exit(1)

    dataset_name = sys.argv[1]

    print(f"\n{'='*60}")
    print(f"TRAINING WITH AUGMENTED DATA - DATASET: {dataset_name}")
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

    # Collect regular (non-augmented) segments for accuracy testing
    regular_segment_files = []
    regular_transcriptions = []

    # Collect ALL segments (regular + augmented) for training
    all_training_segments = []
    all_training_transcriptions = []

    for text_file in text_files:
        surah_part = os.path.splitext(os.path.basename(text_file))[0]
        surah_num = surah_part.split('-')[0]
        mels_dir = f"{datasets_dir}/mels/{surah_num}"
        mels_augmented_dir = f"{datasets_dir}/mels/augmented"

        # Load transcriptions
        with open(text_file, "r", encoding="utf-8") as f:
            transcriptions = [line.strip() for line in f if line.strip()]

        # Find regular mel feature files
        if '-' in surah_part and len(surah_part.split('-')) > 1 and surah_part.split('-')[1]:
            mel_files = sorted(glob.glob(f"{mels_dir}/{surah_part}/{surah_part}-*.pt"))
        else:
            mel_files = sorted(glob.glob(f"{mels_dir}/{surah_part}-*.pt"))

        if not mel_files:
            mel_files = sorted(glob.glob(f"{mels_dir}/{surah_part}/{surah_part}-*.pt"))

        if len(transcriptions) != len(mel_files):
            print(f"⚠️  Warning: Mismatch in {surah_part}: {len(transcriptions)} texts vs {len(mel_files)} mel files")
            continue

        # Add regular segments
        regular_segment_files.extend(mel_files)
        regular_transcriptions.extend(transcriptions)
        all_training_segments.extend(mel_files)
        all_training_transcriptions.extend(transcriptions)

        print(f"  Loaded {len(mel_files)} regular segments from {surah_part}")

        # Find augmented mel feature files (pitch and speed variations)
        augmented_variations = [
            'pitch/minus4', 'pitch/minus2', 'pitch/plus2', 'pitch/plus4',
            'speed/minus20', 'speed/minus10', 'speed/plus10', 'speed/plus20'
        ]

        augmented_count = 0
        has_augmented_data = False
        for aug_type in augmented_variations:
            aug_mel_files = sorted(glob.glob(f"{mels_augmented_dir}/{aug_type}/{surah_num}/{surah_part}-*.pt"))
            if aug_mel_files:
                # Each augmented file corresponds to the same transcription
                all_training_segments.extend(aug_mel_files)
                all_training_transcriptions.extend(transcriptions)
                augmented_count += len(aug_mel_files)
                has_augmented_data = True

        if augmented_count > 0:
            print(f"  Loaded {augmented_count} augmented segments from {surah_part}")

        # Only include regular segments if they have augmented data
        # This ensures we're training only on 40-mel format
        if not has_augmented_data:
            # Remove the regular segments we just added
            regular_segment_files = regular_segment_files[:-len(mel_files)]
            regular_transcriptions = regular_transcriptions[:-len(transcriptions)]
            all_training_segments = all_training_segments[:-len(mel_files)]
            all_training_transcriptions = all_training_transcriptions[:-len(transcriptions)]
            print(f"  Skipping {surah_part} - no augmented data available")

    regular_segments_count = len(regular_segment_files)
    total_training_segments = len(all_training_segments)
    augmented_segments_count = total_training_segments - regular_segments_count

    print(f"\n✓ Regular segments: {regular_segments_count}")
    print(f"✓ Augmented segments: {augmented_segments_count}")
    print(f"✓ Total training segments: {total_training_segments}")
    print(f"✓ Training on 40-mel spectrograms (8kHz audio)")

    # Initialize or load model
    model = EncoderDecoderTransformer(
        vocab_size=len(vocab),
        d_model=128,
        n_encoder_layers=4,
        n_decoder_layers=4,
        n_heads=4,
        d_ff=512,
        dropout=0.1
    )

    if os.path.exists(model_path):
        print(f"\nLoading existing model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"✓ Model loaded successfully! Continuing training.")
    else:
        print(f"\n⚠️  No existing model found. Starting from scratch.")

    model = model.to(device)

    # Training setup with custom LR decay strategy
    learning_rate = 1e-3
    min_lr = 1e-7
    lr_decay_factor = 0.9  # Decay by 10%

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

    print(f"\nTraining Configuration:")
    print(f"  Initial Learning Rate: {learning_rate:.1e}")
    print(f"  LR Decay Factor: {lr_decay_factor} (10% reduction)")
    print(f"  Minimum Learning Rate: {min_lr:.1e}")
    print(f"  Strategy: Decay LR when loss increases, stop at {min_lr:.1e}")

    # Calculate initial accuracy
    initial_acc = calculate_accuracy(model, regular_segment_files, regular_transcriptions, vocab, device)
    print(f"\nInitial accuracy: {initial_acc:.1f}%")

    if initial_acc >= 95.0:
        print(f"\n✓ Model already at {initial_acc:.1f}% accuracy. Skipping training.")
    else:
        # Training loop
        best_loss = float('inf')
        prev_loss = float('inf')
        start_time = time.time()
        epoch = 0

        while True:
            model.train()
            total_loss = 0.0
            total_iterations = 0

            # Shuffle segments
            indices = list(range(len(all_training_segments)))
            random.shuffle(indices)

            for i in indices:
                seg_file = all_training_segments[i]
                text = all_training_transcriptions[i]

                # Load precomputed mel features
                mel_features = load_mel_features(seg_file)
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
            current_lr = optimizer.param_groups[0]['lr']

            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), model_path)

            # Custom LR decay: reduce by 10% when loss increases
            if avg_loss > prev_loss:
                old_lr = current_lr
                new_lr = max(old_lr * lr_decay_factor, min_lr)
                if new_lr != old_lr:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                    print(f"  Loss increased: {prev_loss:.4f} → {avg_loss:.4f}")
                    print(f"  Learning rate reduced: {old_lr:.1e} → {new_lr:.1e}")
                    current_lr = new_lr

            print(f"Epoch {epoch+1} | Loss={avg_loss:.4f} | LR={current_lr:.1e} | Time={elapsed:.0f}s", flush=True)

            # Update prev_loss for next iteration
            prev_loss = avg_loss

            # Check accuracy every 5 epochs
            if (epoch + 1) % 5 == 0:
                current_acc = calculate_accuracy(model, regular_segment_files, regular_transcriptions, vocab, device)
                print(f"Accuracy: {current_acc:.1f}%", flush=True)

                if current_acc >= 95.0:
                    print(f"✓ Early stopping: accuracy reached 95%", flush=True)
                    break

            # Stop if learning rate reaches minimum
            if current_lr <= min_lr:
                print(f"\n✓ Stopping: Learning rate reached minimum ({min_lr:.1e})", flush=True)
                break

            epoch += 1

    # Save final model
    torch.save(model.state_dict(), model_path)
    print(f"\nFinal model saved to: {model_path}")

    # Calculate and output final accuracy
    final_acc = calculate_accuracy(model, regular_segment_files, regular_transcriptions, vocab, device)
    print(f"FINAL_ACCURACY: {final_acc:.0f}%")


if __name__ == "__main__":
    main()
