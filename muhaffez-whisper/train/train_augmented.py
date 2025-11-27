#!/usr/bin/env python3
"""
Train on segments including augmented variations (pitch + speed)
Uses PRECOMPUTED mel spectrograms from mels/normal/ and mels/augmented/ directories

Usage:
  python3 train_augmented.py <dataset_name> all                # Train all parts with augmentation
  python3 train_augmented.py <dataset_name> <surah_part>       # Train specific part with augmentation

Examples:
  python3 train_augmented.py Quran-A all            # Train all parts
  python3 train_augmented.py Quran-A 001            # Train on Al-Fatiha (001)
  python3 train_augmented.py Quran-A 002-04         # Train on Al-Baqara part 4
"""
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import json
import torch
import torch.nn as nn
import glob
import os
import time
sys.path.append("..")
from tools.encoder_decoder_transformer import EncoderDecoderTransformer

# Import common utilities
from common import (
    load_mel_features,
    tokenize_text,
    calculate_accuracy,
    run_training_epoch,
    update_learning_rate,
    format_time,
    collect_augmented_data,
    load_single_part_data,
    save_checkpoint,
    load_checkpoint
)

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
# Main
# ==============================================================
def main():
    if len(sys.argv) < 3:
        print("Usage: python3 train_augmented.py <dataset_name> <surah_part|all>")
        print("Examples:")
        print("  python3 train_augmented.py Quran-A all            # Train all parts")
        print("  python3 train_augmented.py Quran-A 001            # Train specific part")
        print("  python3 train_augmented.py Quran-A 002-04         # Train specific part")
        sys.exit(1)

    dataset_name = sys.argv[1]
    surah_part = sys.argv[2]

    if surah_part == "all":
        train_all_parts(dataset_name)
    else:
        train_single_part(dataset_name, surah_part)

def train_all_parts(dataset_name):
    """Train on ALL segments across ALL surah parts with augmentation"""

    print(f"\n{'='*60}")
    print(f"TRAINING WITH AUGMENTED DATA - DATASET: {dataset_name}")
    print(f"{'='*60}\n")

    vocab_path = "../models/vocabulary.json"
    model_path = "../models/muhaffez_whisper.pt"
    datasets_dir = f"../datasets/{dataset_name}"

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"Vocabulary size: {len(vocab)}")

    text_files = sorted(glob.glob(f"{datasets_dir}/text/*.txt"))
    if not text_files:
        print(f"❌ No text files found in {datasets_dir}/text/")
        sys.exit(1)

    # Collect regular and augmented segments
    regular_segment_files, regular_transcriptions, all_training_segments, all_training_transcriptions = collect_augmented_data(dataset_name, text_files)

    regular_segments_count = len(regular_segment_files)
    total_training_segments = len(all_training_segments)
    augmented_segments_count = total_training_segments - regular_segments_count

    print(f"\n✓ Regular segments: {regular_segments_count}")
    print(f"✓ Augmented segments: {augmented_segments_count}")
    print(f"✓ Training on 40-mel spectrograms (8kHz audio)")
    print(f"✓ Total training samples: {total_training_segments}\n")

    # Initialize model
    model = EncoderDecoderTransformer(
        vocab_size=len(vocab),
        d_model=128,
        n_encoder_layers=4,
        n_decoder_layers=4,
        n_heads=4,
        d_ff=512,
        dropout=0.1
    )

    model = model.to(device)

    # Setup optimizer and load checkpoint if exists
    min_lr = 1e-7
    lr_decay_factor = 0.5

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    checkpoint_info = load_checkpoint(model, optimizer, model_path, training_type="augmented", device=device)

    if checkpoint_info['restored']:
        print(f"✓ Checkpoint restored: Epoch {checkpoint_info['epoch']}, LR={checkpoint_info['lr']:.1e}")
    elif os.path.exists(model_path):
        print(f"✓ Model loaded (starting fresh with LR=1e-3)")
    else:
        print(f"⚠️  No existing model found. Starting from scratch.")

    learning_rate = checkpoint_info['lr']
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

    print(f"\nTraining Configuration:")
    print(f"  Initial Learning Rate: {learning_rate:.1e}")
    print(f"  LR Decay Factor: {lr_decay_factor} (50% reduction)")
    print(f"  Minimum Learning Rate: {min_lr:.1e}")
    print(f"  Strategy: Decay LR when loss increases, stop at {min_lr:.1e}\n")

    # Build training tuples
    all_training_tuples = [(seg_file, text, None, None) for seg_file, text in zip(all_training_segments, all_training_transcriptions)]

    # Training loop
    best_loss = float('inf')
    prev_loss = float('inf')
    start_time = time.time()
    checkpoint_time = start_time
    epoch = 0

    while True:
        avg_loss = run_training_epoch(model, all_training_tuples, vocab, criterion, optimizer, device)

        if avg_loss is None:
            print(f"⚠️  Warning: No valid training samples. Stopping.")
            break

        current_lr = optimizer.param_groups[0]['lr']

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_path)

        # Check accuracy every 10 epochs
        accuracy_str = ""
        current_acc = 0
        if epoch == 0 or (epoch + 1) % 10 == 0:
            current_acc = calculate_accuracy(model, regular_segment_files, regular_transcriptions, vocab, device)
            accuracy_str = f" | Accuracy={current_acc:.0f}%"

        time_str = format_time(time.time() - checkpoint_time)
        print(f"Epoch {epoch+1}/500 | Loss={avg_loss:.4f}{accuracy_str} | Time={time_str}", flush=True)
        checkpoint_time = time.time()

        # LR decay on loss increase
        new_lr, lr_changed = update_learning_rate(optimizer, avg_loss, prev_loss, min_lr, lr_decay_factor)
        if lr_changed:
            print(f"  Loss increased ({prev_loss:.4f} → {avg_loss:.4f}), reducing LR to: {new_lr:.1e}")
            save_checkpoint(model, optimizer, epoch, avg_loss, model_path, training_type="augmented")

        prev_loss = avg_loss

        if new_lr <= min_lr:
            print(f"\n✓ Stopping: Learning rate reached minimum ({min_lr:.1e})", flush=True)
            break

        if current_acc > 99.0:
            print(f"\n✓ Stopping: Accuracy > 99%", flush=True)
            break

        epoch += 1

    save_checkpoint(model, optimizer, epoch, avg_loss, model_path, training_type="augmented")
    print(f"\nFinal model saved to: {model_path}")

    final_acc = calculate_accuracy(model, regular_segment_files, regular_transcriptions, vocab, device)
    print(f"FINAL_ACCURACY: {final_acc:.0f}%")

def train_single_part(dataset_name, surah_part):
    """Train on a single surah part with augmentation"""

    print(f"\n{'='*60}")
    print(f"TRAINING WITH AUGMENTED DATA - PART: {surah_part}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}\n")

    vocab_path = "../models/vocabulary.json"
    model_path = "../models/muhaffez_whisper.pt"
    datasets_dir = f"../datasets/{dataset_name}"

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"Vocabulary size: {len(vocab)}")

    surah_num = surah_part.split('-')[0]
    mels_dir = f"{datasets_dir}/mels/normal/{surah_num}"
    mels_augmented_dir = f"{datasets_dir}/mels/augmented"

    text_path = f"{datasets_dir}/text/{surah_part}.txt"
    if not os.path.exists(text_path):
        print(f"❌ Error: Text file not found: {text_path}")
        sys.exit(1)

    with open(text_path, "r", encoding="utf-8") as f:
        transcriptions = [line.strip() for line in f if line.strip()]

    # Find regular mel files
    if '-' in surah_part and len(surah_part.split('-')) > 1 and surah_part.split('-')[1]:
        mel_files = sorted(glob.glob(f"{mels_dir}/{surah_part}/{surah_part}-*.pt"))
    else:
        mel_files = sorted(glob.glob(f"{mels_dir}/{surah_part}-*.pt"))

    if not mel_files:
        mel_files = sorted(glob.glob(f"{mels_dir}/{surah_part}/{surah_part}-*.pt"))

    if not mel_files:
        print(f"❌ Error: No mel files found for {surah_part}")
        sys.exit(1)

    regular_segment_files = list(mel_files)
    regular_transcriptions = list(transcriptions)
    all_training_segments = list(mel_files)
    all_training_transcriptions = list(transcriptions)

    print(f"  Loaded {len(mel_files)} regular segments from {surah_part}")

    # Find augmented mel files
    augmented_variations = [
        'pitch/minus4', 'pitch/minus2', 'pitch/plus2', 'pitch/plus4',
        'speed/minus20', 'speed/minus10', 'speed/plus10', 'speed/plus20'
    ]

    augmented_count = 0
    for aug_type in augmented_variations:
        if '-' in surah_part and len(surah_part.split('-')) > 1 and surah_part.split('-')[1]:
            aug_mel_files = sorted(glob.glob(f"{mels_augmented_dir}/{aug_type}/{surah_num}/{surah_part}/{surah_part}-*.pt"))
        else:
            aug_mel_files = sorted(glob.glob(f"{mels_augmented_dir}/{aug_type}/{surah_num}/{surah_part}-*.pt"))

        if aug_mel_files:
            all_training_segments.extend(aug_mel_files)
            all_training_transcriptions.extend(transcriptions)
            augmented_count += len(aug_mel_files)

    if augmented_count > 0:
        print(f"  Loaded {augmented_count} augmented segments from {surah_part}")
    else:
        print(f"  ⚠️  No augmented data found for {surah_part}")

    print(f"\n✓ Regular segments: {len(regular_segment_files)}")
    print(f"✓ Augmented segments: {augmented_count}")
    print(f"✓ Total training samples: {len(all_training_segments)}\n")

    # Initialize model
    model = EncoderDecoderTransformer(
        vocab_size=len(vocab),
        d_model=128,
        n_encoder_layers=4,
        n_decoder_layers=4,
        n_heads=4,
        d_ff=512,
        dropout=0.1
    )

    model = model.to(device)

    # Setup optimizer and load checkpoint if exists
    min_lr = 1e-7
    lr_decay = 0.5
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    checkpoint_info = load_checkpoint(model, optimizer, model_path, training_type="augmented", device=device)

    if checkpoint_info['restored']:
        print(f"✓ Checkpoint restored: Epoch {checkpoint_info['epoch']}, LR={checkpoint_info['lr']:.1e}")
    elif os.path.exists(model_path):
        print(f"✓ Model loaded (starting fresh with LR=1e-3)")
    else:
        print(f"⚠️  No existing model found. Starting from scratch.")

    initial_lr = checkpoint_info['lr']
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

    print(f"\nStarting training on {len(all_training_segments)} segments...")
    print(f"Initial Learning Rate: {initial_lr:.1e}")
    print(f"LR Decay: {lr_decay} (when loss increases)")
    print(f"Minimum LR: {min_lr:.1e}\n")

    # Build training tuples
    all_training_tuples = [(seg_file, text, None, None) for seg_file, text in zip(all_training_segments, all_training_transcriptions)]

    # Training loop
    best_loss = float('inf')
    prev_loss = float('inf')
    start_time = time.time()
    checkpoint_time = start_time
    epoch = 0
    max_epochs = 500

    while epoch < max_epochs:
        avg_loss = run_training_epoch(model, all_training_tuples, vocab, criterion, optimizer, device)

        if avg_loss is None:
            print(f"⚠️  Warning: No valid training samples. Stopping.")
            break

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_path)

        # Check accuracy every 10 epochs
        accuracy_str = ""
        current_acc = 0
        if epoch == 0 or (epoch + 1) % 10 == 0:
            current_acc = calculate_accuracy(model, regular_segment_files, regular_transcriptions, vocab, device)
            accuracy_str = f" | Accuracy={current_acc:.0f}%"

        time_str = format_time(time.time() - checkpoint_time)
        print(f"Epoch {epoch+1} | Loss={avg_loss:.4f}{accuracy_str} | Time={time_str}", flush=True)
        checkpoint_time = time.time()

        # LR decay on loss increase
        new_lr, lr_changed = update_learning_rate(optimizer, avg_loss, prev_loss, min_lr, lr_decay)
        if lr_changed:
            print(f"  Loss increased ({prev_loss:.4f} → {avg_loss:.4f}), reducing LR to: {new_lr:.1e}")
            save_checkpoint(model, optimizer, epoch, avg_loss, model_path, training_type="augmented")

        prev_loss = avg_loss

        if new_lr <= min_lr:
            print(f"\n✓ Stopping: Learning rate reached minimum ({min_lr:.1e})", flush=True)
            break

        if current_acc > 99.0:
            print(f"\n✓ Stopping: Accuracy > 99%", flush=True)
            break

        epoch += 1

    save_checkpoint(model, optimizer, epoch, avg_loss, model_path, training_type="augmented")
    print(f"\nFinal model saved to: {model_path}")

    final_acc = calculate_accuracy(model, regular_segment_files, regular_transcriptions, vocab, device)
    print(f"FINAL_ACCURACY: {final_acc:.0f}%")


if __name__ == "__main__":
    main()
