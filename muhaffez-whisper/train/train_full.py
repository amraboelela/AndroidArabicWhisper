#!/usr/bin/env python3
"""
Universal training script for encoder-decoder model
Usage:
  python3 train_full.py <dataset_name> all                # Train all parts in dataset
  python3 train_full.py <dataset_name> <surah_part>       # Train specific part

Examples:
  python3 train_full.py Quran-A all            # Train all parts
  python3 train_full.py Quran-A 001            # Train on Al-Fatiha (001)
  python3 train_full.py Quran-A 002-01         # Train on Al-Baqara part 1
  python3 train_full.py Quran-A 002-04         # Train on Al-Baqara part 4
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
import glob
import os
import time
import random
sys.path.append("../models")
from encoder_decoder_transformer import EncoderDecoderTransformer

# Import common utilities
from common import (
    load_mel_features,
    tokenize_text,
    calculate_comprehensive_accuracy,
    collect_replay_samples,
    collect_curriculum_replay_samples,
    run_training_epoch,
    update_learning_rate,
    format_time,
    collect_segment_files,
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
        print("Usage: python3 train_full.py <dataset_name> <surah_part|all>")
        print("Examples:")
        print("  python3 train_full.py Quran-A all            # Train all parts")
        print("  python3 train_full.py Quran-A 001            # Train specific part")
        print("  python3 train_full.py Quran-A 002-04         # Train specific part")
        sys.exit(1)

    dataset_name = sys.argv[1]
    surah_part = sys.argv[2]

    # Check if training all parts or single part
    if surah_part == "all":
        # Train all parts mode
        train_all_parts(dataset_name)
    else:
        # Train single part mode
        train_single_part(dataset_name, surah_part)

def train_all_parts(dataset_name):
    """Train on ALL segments across ALL surah parts in a dataset"""
    vocab_path = "../models/vocabulary.json"
    model_path = "../models/muhaffez_whisper.pt"
    datasets_dir = f"../datasets/{dataset_name}"

    print(f"\n{'='*60}")
    print(f"Full Training - Dataset: {dataset_name}")
    print(f"{'='*60}\n")

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
    all_segment_files, all_transcriptions = collect_segment_files(dataset_name, text_files)

    total_segments = len(all_segment_files)
    print(f"\n✓ Total segments: {total_segments}")
    print(f"✓ Training on full audio/text for all segments\n")

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

    model = model.to(device)

    # Setup optimizer and load checkpoint if exists
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    checkpoint_info = load_checkpoint(model, optimizer, model_path, training_type="full", device=device)

    if checkpoint_info['restored']:
        print(f"✓ Checkpoint restored (with optimizer state): Epoch {checkpoint_info['epoch']}, LR={checkpoint_info['lr']:.1e}")
    elif os.path.exists(model_path):
        print(f"✓ Model loaded (starting fresh with LR=1e-3)")
    else:
        print(f"⚠️  No existing model found. Starting from scratch.")

    learning_rate = checkpoint_info['lr']
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

    print(f"\nStarting training for up to 500 epochs on {total_segments} segments...")
    print(f"Initial Learning Rate: {learning_rate:.1e}\n")

    # Training loop
    best_loss = float('inf')
    best_accuracy = 0.0
    prev_loss = float('inf')
    start_time = time.time()
    checkpoint_time = start_time

    # Build training tuples
    all_training_tuples = [(seg_file, text, None, None) for seg_file, text in zip(all_segment_files, all_transcriptions)]

    for epoch in range(500):
        avg_loss = run_training_epoch(model, all_training_tuples, vocab, criterion, optimizer, device)

        if avg_loss is None:
            print(f"⚠️  Warning: No valid training samples. Stopping.")
            break

        # Check accuracy every 10 epochs (or every epoch if accuracy >= 95%)
        accuracy_str = ""
        current_acc = 0
        should_calc_accuracy = epoch == 0 or (epoch + 1) % 10 == 0 or best_accuracy >= 95
        if should_calc_accuracy:
            current_acc = calculate_comprehensive_accuracy(model, all_segment_files, all_transcriptions, vocab, None, None, device)[0]
            accuracy_str = f" | Accuracy={current_acc:.0f}%"
            # Update best accuracy
            if current_acc > best_accuracy:
                best_accuracy = current_acc

        # Track best loss and save when we get a new best
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save checkpoint when we achieve new best loss
            save_checkpoint(model, optimizer, epoch + 1, avg_loss, model_path, training_type="full", accuracy=best_accuracy)

        time_str = format_time(time.time() - checkpoint_time)
        print(f"Epoch {epoch+1} | Loss={avg_loss:.4f}{accuracy_str} | Time={time_str}", flush=True)

        # Decay learning rate if loss increases
        new_lr, lr_changed = update_learning_rate(optimizer, avg_loss, prev_loss, 1e-7, 0.5)
        if lr_changed:
            print(f"  Loss increased ({prev_loss:.4f} → {avg_loss:.4f}), reducing LR to: {new_lr:.1e}")

        checkpoint_time = time.time()
        prev_loss = avg_loss

        # Stop if learning rate reaches minimum
        if new_lr <= 1e-7:
            print(f"✓ Stopping: Learning rate reached minimum (1e-7)", flush=True)
            break

        # Stop if accuracy > 99%
        if current_acc > 99.0:
            print(f"✓ Stopping: Accuracy > 99%", flush=True)
            break

def train_single_part(dataset_name, surah_part):
    """Train on a single surah part"""
    vocab_path = "../models/vocabulary.json"
    model_path = "../models/muhaffez_whisper.pt"

    # Load vocab
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # Load transcriptions and segments
    segment_files, transcriptions = load_single_part_data(dataset_name, surah_part)

    print(f"Loaded {len(transcriptions)} transcriptions, {len(segment_files)} mel files")

    if len(transcriptions) != len(segment_files):
        print(f"⚠️  Warning: Mismatch between transcriptions and segments")

    print(f"\n{'='*60}")
    print(f"FULL-LENGTH TRAINING - PART: {surah_part}")
    print(f"{'='*60}\n")

    print(f"   Training samples: {len(segment_files)} segments\n")

    # Create model
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    checkpoint_info = load_checkpoint(model, optimizer, model_path, training_type="full", device=device)

    if checkpoint_info['restored']:
        print(f"✓ Checkpoint restored (with optimizer state): Epoch {checkpoint_info['epoch']}, LR={checkpoint_info['lr']:.1e}")
    elif os.path.exists(model_path):
        print(f"✓ Model loaded (starting fresh with LR=1e-3)")
    else:
        print(f"⚠️  No existing model found. Starting from scratch.")

    learning_rate = checkpoint_info['lr']
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

    print(f"Initial Learning Rate: {learning_rate:.1e}\n")

    # Build training tuples
    all_training_tuples = [(seg_file, text, None, None) for seg_file, text in zip(segment_files, transcriptions)]

    best_loss = float('inf')
    best_accuracy = 0.0
    prev_loss = float('inf')
    start_time = time.time()
    checkpoint_time = start_time
    epoch = 0
    max_epochs = 500

    while epoch < max_epochs:
        model.train()
        total_loss = 0.0
        total_iterations = 0

        random.shuffle(all_training_tuples)

        for seg_file, text, target_sec, target_wrd in all_training_tuples:
            mel_features = load_mel_features(seg_file, target_seconds=target_sec)
            audio_batch = mel_features.transpose(0, 1).unsqueeze(0).to(device)

            if target_wrd:
                words = text.split()
                if len(words) < target_wrd:
                    continue
                target_text = " ".join(words[:target_wrd])
            else:
                target_text = text

            if not target_text:
                continue

            text_tokens = tokenize_text(target_text, vocab)
            full_sequence = [1] + text_tokens + [2]
            input_ids = torch.tensor([full_sequence[:-1]], dtype=torch.long, device=device)
            labels = torch.tensor([full_sequence[1:]], dtype=torch.long, device=device)

            logits = model(mel_features=audio_batch, text_ids=input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_iterations += 1

        if total_iterations == 0:
            print(f"⚠️  Warning: No valid training samples. Stopping.")
            break

        avg_loss = total_loss / total_iterations

        # Check accuracy every 10 epochs (or every epoch if accuracy >= 95%)
        accuracy_str = ""
        current_acc = 0
        should_calc_accuracy = epoch == 0 or (epoch + 1) % 10 == 0 or best_accuracy >= 95
        if should_calc_accuracy:
            current_acc = calculate_comprehensive_accuracy(model, segment_files, transcriptions, vocab, None, None, device)[0]
            accuracy_str = f" | Accuracy={current_acc:.0f}%"
            # Update best accuracy
            if current_acc > best_accuracy:
                best_accuracy = current_acc

        # Track best loss and save when we get a new best
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save checkpoint when we achieve new best loss
            save_checkpoint(model, optimizer, epoch + 1, avg_loss, model_path, training_type="full", accuracy=best_accuracy)

        # Format time
        elapsed_from_checkpoint = time.time() - checkpoint_time
        if elapsed_from_checkpoint >= 3600:
            # Round to nearest hour
            hours = int(round(elapsed_from_checkpoint / 3600))
            time_str = f"{hours}h"
        elif elapsed_from_checkpoint >= 60:
            time_str = f"{int(round(elapsed_from_checkpoint / 60))}m"
        else:
            time_str = f"{int(round(elapsed_from_checkpoint))}s"

        print(f"Epoch {epoch+1} | Loss={avg_loss:.4f}{accuracy_str} | Time={time_str}", flush=True)

        checkpoint_time = time.time()

        # Dynamic learning rate
        if avg_loss > prev_loss:
            current_lr = optimizer.param_groups[0]['lr']
            new_lr = max(current_lr * 0.5, min_lr)
            if new_lr != current_lr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                if new_lr > min_lr:
                    print(f"  Loss increased ({prev_loss:.4f} → {avg_loss:.4f}), reducing LR to: {new_lr:.1e}", flush=True)

        prev_loss = avg_loss
        current_lr = optimizer.param_groups[0]['lr']

        if current_lr <= min_lr:
            print(f"\n✓ Stopping: Learning rate reached minimum ({min_lr:.1e})", flush=True)
            break

        if current_acc >= 100.0:
            print(f"\n✓ Stopping: Accuracy reached 100%", flush=True)
            break

        epoch += 1


if __name__ == "__main__":
    main()
