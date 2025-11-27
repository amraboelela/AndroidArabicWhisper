#!/usr/bin/env python3
"""
Curriculum training script - trains incrementally on chunks
Usage:
  python3 train_curriculum.py <dataset_name> all         # Train all parts in dataset
  python3 train_curriculum.py <dataset_name> <surah_part> # Train specific part

Examples:
  python3 train_curriculum.py Quran-A all          # Train all parts
  python3 train_curriculum.py Quran-A 002-04       # Train only part 002-04
  python3 train_curriculum.py Quran-A 001          # Train only part 001

This script trains the model using curriculum learning:
- Stage 1: First 1.3s of each segment → first 1 word
- Stage 2: First 2.6s of each segment → first 2 words
- Stage 3: First 3.9s of each segment → first 3 words
- ... and so on until full segment audio → full segment transcription
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
import subprocess
sys.path.append("../models")
from encoder_decoder_transformer import EncoderDecoderTransformer

# Import common utilities
from common import (
    load_mel_features,
    tokenize_text,
    calculate_comprehensive_accuracy,
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
# Configuration
# ==============================================================
CHUNK_DURATION = 1.3  # seconds per chunk
WORDS_PER_CHUNK = 1   # words per chunk

# ==============================================================
# Main
# ==============================================================
def main():
    if len(sys.argv) < 3:
        print("Usage: python3 train_curriculum.py <dataset_name> <surah_part|all>")
        print("Examples:")
        print("  python3 train_curriculum.py Quran-A all          # Train all parts")
        print("  python3 train_curriculum.py Quran-A 002-04       # Train specific part")
        print("  python3 train_curriculum.py Quran-A 001          # Train specific part")
        sys.exit(1)

    dataset_name = sys.argv[1]  # e.g., "Quran-A"
    surah_part = sys.argv[2]  # e.g., "all", "001", "002-01", "002-04"

    # Check if training all parts or single part
    if surah_part == "all":
        train_all_parts(dataset_name)
    else:
        # Call internal script for single part training
        script_path = os.path.join(os.path.dirname(__file__), "internal", "train_curriculum_single.py")
        result = subprocess.run([sys.executable, script_path, dataset_name, surah_part])
        sys.exit(result.returncode)

def train_all_parts(dataset_name):
    """Train on ALL segments across ALL surah parts with curriculum learning"""
    vocab_path = "../models/vocabulary.json"
    model_path = "../models/muhaffez_whisper.pt"
    datasets_dir = f"../datasets/{dataset_name}"

    print(f"\n{'='*60}")
    print(f"CURRICULUM TRAINING - DATASET: {dataset_name}")
    print(f"{'='*60}\n")

    # Load vocabulary
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"Vocabulary size: {len(vocab)}")

    # Find ALL text files
    text_files = sorted(glob.glob(f"{datasets_dir}/text/*.txt"))
    if not text_files:
        print(f"❌ No text files found in {datasets_dir}/text/")
        sys.exit(1)

    # Collect all segments from all parts
    all_segment_files, all_transcriptions = collect_segment_files(dataset_name, text_files)

    if not all_segment_files:
        print(f"❌ No mel files found for any part!")
        sys.exit(1)

    print(f"Total segments: {len(all_segment_files)} across {len(text_files)} parts\n")

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

    # Collect segment info for curriculum stages
    segment_info = []
    for segment_file, transcription in zip(all_segment_files, all_transcriptions):
        segment_name = os.path.basename(segment_file)
        words = transcription.split()
        num_words = len(words)

        # Get audio duration from mel features (precomputed at 100 fps)
        mel_features = torch.load(segment_file, map_location='cpu', weights_only=True)
        audio_duration = mel_features.shape[0] / 100.0

        # Calculate how many chunks fit in this audio
        num_chunks = int(audio_duration / CHUNK_DURATION)
        max_chunks = min(num_chunks, num_words)

        segment_info.append({
            'file': segment_file,
            'name': segment_name,
            'transcription': transcription,
            'audio_duration': audio_duration,
            'num_words': num_words,
            'max_chunks': max_chunks
        })

    # Find the maximum number of chunks across all segments
    global_max_chunks = max(info['max_chunks'] for info in segment_info)

    print(f"Total segments: {len(segment_info)}")
    print(f"Maximum curriculum stages: {global_max_chunks}")
    print(f"Chunk size: {CHUNK_DURATION}s → {WORDS_PER_CHUNK} word(s)\n")

    # Collect ALL curriculum samples (all stages mixed together)
    all_curriculum_files = []
    all_curriculum_transcriptions = []
    all_curriculum_target_seconds = []
    all_curriculum_target_words = []

    print("Collecting all curriculum stages...")
    for chunk_count in range(1, global_max_chunks + 1):
        target_seconds = chunk_count * CHUNK_DURATION
        target_words = chunk_count * WORDS_PER_CHUNK

        for info in segment_info:
            # Skip if segment is too short for this chunk count
            if chunk_count > info['max_chunks']:
                continue

            # Add this curriculum sample
            all_curriculum_files.append(info['file'])
            all_curriculum_transcriptions.append(info['transcription'])
            all_curriculum_target_seconds.append(target_seconds)
            all_curriculum_target_words.append(target_words)

    print(f"Total curriculum samples: {len(all_curriculum_files)}")

    # Train on all mixed curriculum samples in one big stage
    print(f"{'='*60}")
    print(f"TRAINING ALL CURRICULUM STAGES MIXED")
    print(f"{'='*60}\n")

    # Setup optimizer and load checkpoint if exists
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    checkpoint_info = load_checkpoint(model, optimizer, model_path, training_type="curriculum", device=device)

    if checkpoint_info['restored']:
        print(f"✓ Checkpoint restored (with optimizer state): Epoch {checkpoint_info['epoch']}, LR={checkpoint_info['lr']:.1e}")
    elif os.path.exists(model_path):
        print(f"✓ Model loaded (starting fresh with LR=1e-3)")
    else:
        print(f"⚠️  No existing model found. Starting from scratch.")

    learning_rate = checkpoint_info['lr']
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

    print(f"Initial Learning Rate: {learning_rate:.1e}\n")

    best_loss = float('inf')
    best_accuracy = 0.0
    prev_loss = float('inf')
    start_time = time.time()
    checkpoint_time = start_time

    for epoch in range(500):
        model.train()
        total_loss = 0.0
        total_iterations = 0

        # Shuffle all curriculum samples
        indices = list(range(len(all_curriculum_files)))
        random.shuffle(indices)

        for i in indices:
            seg_file = all_curriculum_files[i]
            text = all_curriculum_transcriptions[i]
            target_sec = all_curriculum_target_seconds[i]
            target_wrd = all_curriculum_target_words[i]

            # Load mel features (with optional truncation)
            audio_features = load_mel_features(seg_file, target_seconds=target_sec)
            audio_batch = audio_features.transpose(0, 1).unsqueeze(0).to(device)

            # Extract target text
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
            print(f"⚠️  Warning: No valid training samples. Skipping.")
            break

        avg_loss = total_loss / total_iterations

        # Calculate accuracy every 10 epochs
        accuracy_str = ""
        current_acc = 0
        if epoch == 0 or (epoch + 1) % 10 == 0:
            overall_acc = calculate_comprehensive_accuracy(model, all_segment_files, all_transcriptions, vocab, None, None, device)[0]
            current_acc = overall_acc
            accuracy_str = f" | Accuracy={overall_acc:.0f}%"
            # Update best accuracy
            if current_acc > best_accuracy:
                best_accuracy = current_acc

        # Track best loss and save when we get a new best
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save checkpoint when we achieve new best loss
            save_checkpoint(model, optimizer, epoch + 1, avg_loss, model_path, training_type="curriculum", accuracy=best_accuracy)

        # Format time
        elapsed_from_checkpoint = time.time() - checkpoint_time
        if elapsed_from_checkpoint >= 3600:
            hours = int(elapsed_from_checkpoint // 3600)
            minutes = int((elapsed_from_checkpoint % 3600) // 60)
            time_str = f"{hours}h {minutes}m" if minutes > 0 else f"{hours}h"
        elif elapsed_from_checkpoint >= 60:
            time_str = f"{int(round(elapsed_from_checkpoint / 60))}m"
        else:
            time_str = f"{int(round(elapsed_from_checkpoint))}s"

        current_lr = optimizer.param_groups[0]['lr']

        # Print every epoch in "all" mode
        print(f"Epoch {epoch+1} | Loss={avg_loss:.4f}{accuracy_str} | Time={time_str}")

        # Dynamic learning rate: reduce by 50% if loss increases
        if avg_loss > prev_loss:
            current_lr = optimizer.param_groups[0]['lr']
            new_lr = max(current_lr * 0.5, 1e-7)
            if new_lr != current_lr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                if new_lr > 1e-7:
                    print(f"  Loss increased ({prev_loss:.4f} → {avg_loss:.4f}), reducing LR to: {new_lr:.1e}")

        checkpoint_time = time.time()

        prev_loss = avg_loss

        # Stop if learning rate reaches minimum
        if current_lr <= 1e-7:
            print(f"\n✓ Stopping: Learning rate reached minimum (1e-7)")
            break

        # Stop if accuracy > 99%
        if current_acc > 99.0:
            print(f"\n✓ Stopping: Accuracy > 99%")
            break

    print(f"\nTraining complete. Best model already saved to: {model_path}")




if __name__ == "__main__":
    main()
