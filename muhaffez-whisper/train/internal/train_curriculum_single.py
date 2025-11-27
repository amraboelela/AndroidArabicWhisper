#!/usr/bin/env python3
"""
Curriculum training for a single surah part
Usage:
  python3 train_curriculum_single.py <dataset_name> <surah_part>

Examples:
  python3 train_curriculum_single.py Quran-A 002-04
  python3 train_curriculum_single.py Quran-A 001
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
sys.path.append("../../models")
from encoder_decoder_transformer import EncoderDecoderTransformer

# Import common utilities
from common import (
    load_mel_features,
    tokenize_text,
    calculate_comprehensive_accuracy,
    calculate_curriculum_accuracy,
    collect_augmented_replay_samples,
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
        print("Usage: python3 train_curriculum_single.py <dataset_name> <surah_part>")
        print("Examples:")
        print("  python3 train_curriculum_single.py Quran-A 002-04")
        print("  python3 train_curriculum_single.py Quran-A 001")
        sys.exit(1)

    dataset_name = sys.argv[1]
    surah_part = sys.argv[2]

    train_single_part(dataset_name, surah_part)

def train_single_part(dataset_name, surah_part):
    """Train on a single surah part with curriculum learning"""

    datasets_dir = f"../datasets/{dataset_name}"
    mels_dir = f"{datasets_dir}/mels/normal"
    vocab_path = "../models/vocabulary.json"
    model_path = "../models/muhaffez_whisper.pt"

    # Load vocab
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    print(f"\n{'='*60}")
    print(f"CURRICULUM TRAINING - PART: {surah_part}")
    print(f"Dataset: {dataset_name}")
    print(f"Vocabulary: {len(vocab)} words")
    print(f"Chunk size: {CHUNK_DURATION}s → {WORDS_PER_CHUNK} word(s)")
    print(f"{'='*60}")

    # Parse surah part name to determine surah number
    surah_num = surah_part.split('-')[0]

    # Load transcriptions and segments
    text_path = f"{datasets_dir}/text/{surah_part}.txt"
    if not os.path.exists(text_path):
        print(f"❌ Error: Text file not found: {text_path}")
        sys.exit(1)

    with open(text_path, "r", encoding="utf-8") as f:
        transcriptions = [line.strip() for line in f if line.strip()]

    # Determine mel directory based on segment structure
    if '-' in surah_part and len(surah_part.split('-')) > 1 and surah_part.split('-')[1]:
        segment_files = sorted(glob.glob(os.path.join(mels_dir, surah_num, surah_part, f"{surah_part}-*.pt")))
    else:
        segment_files = sorted(glob.glob(os.path.join(mels_dir, surah_num, f"{surah_part}-*.pt")))

    if not segment_files:
        segment_files = sorted(glob.glob(os.path.join(mels_dir, surah_num, surah_part, f"{surah_part}-*.pt")))
        if not segment_files:
            print(f"❌ Error: No mel files found")
            sys.exit(1)

    print(f"Loaded {len(transcriptions)} transcriptions, {len(segment_files)} mel files")

    if len(transcriptions) != len(segment_files):
        print(f"⚠️  Warning: Mismatch between transcriptions and segments")

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

    # Calculate max chunks across all segments
    segment_info = []
    for segment_file, transcription in zip(segment_files, transcriptions):
        mel_features = torch.load(segment_file, map_location='cpu', weights_only=True)
        audio_duration = mel_features.shape[0] / 100.0
        num_words = len(transcription.split())
        num_chunks = int(audio_duration / CHUNK_DURATION)
        max_chunks = min(num_chunks, num_words)

        segment_info.append({
            'file': segment_file,
            'transcription': transcription,
            'max_chunks': max_chunks
        })

    global_max_chunks = max(info['max_chunks'] for info in segment_info)

    print(f"\nTotal segments: {len(segment_info)}")
    print(f"Maximum curriculum stages: {global_max_chunks}")
    print(f"Chunk size: {CHUNK_DURATION}s → {WORDS_PER_CHUNK} word(s)\n")

    # Collect ALL curriculum samples
    all_curriculum_files = []
    all_curriculum_transcriptions = []
    all_curriculum_target_seconds = []
    all_curriculum_target_words = []

    print("Collecting all curriculum stages...")
    for chunk_count in range(1, global_max_chunks + 1):
        target_seconds = chunk_count * CHUNK_DURATION
        target_words = chunk_count * WORDS_PER_CHUNK

        for info in segment_info:
            if chunk_count > info['max_chunks']:
                continue

            all_curriculum_files.append(info['file'])
            all_curriculum_transcriptions.append(info['transcription'])
            all_curriculum_target_seconds.append(target_seconds)
            all_curriculum_target_words.append(target_words)

    print(f"Total curriculum samples: {len(all_curriculum_files)}\n")

    # Collect replay buffer from augmented data (10%)
    print("Collecting replay buffer from augmented data...")
    replay_samples = collect_augmented_replay_samples(dataset_name, len(all_curriculum_files))

    # Add replay samples to curriculum training data
    for replay_file, replay_text, target_sec, target_wrd in replay_samples:
        all_curriculum_files.append(replay_file)
        all_curriculum_transcriptions.append(replay_text)
        all_curriculum_target_seconds.append(target_sec)
        all_curriculum_target_words.append(target_wrd)

    print(f"Total training samples (curriculum + replay): {len(all_curriculum_files)}\n")

    # Train
    print(f"{'='*60}")
    print(f"TRAINING ALL CURRICULUM STAGES MIXED")
    print(f"{'='*60}\n")

    model = model.to(device)

    # Setup optimizer and load checkpoint if exists
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    checkpoint_info = load_checkpoint(model, optimizer, model_path, training_type="curriculum", device=device)

    if checkpoint_info['restored']:
        print(f"✓ Checkpoint restored: Epoch {checkpoint_info['epoch']}, LR={checkpoint_info['lr']:.1e}")
    elif os.path.exists(model_path):
        print(f"✓ Model loaded (starting fresh with LR=1e-3)")
    else:
        print(f"⚠️  No existing model found. Starting from scratch.")

    learning_rate = checkpoint_info['lr']
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

    print(f"Initial Learning Rate: {learning_rate:.1e}")

    best_loss = float('inf')
    prev_loss = float('inf')
    checkpoint_time = time.time()

    for epoch in range(500):
        model.train()
        total_loss = 0.0
        total_iterations = 0

        indices = list(range(len(all_curriculum_files)))
        random.shuffle(indices)

        for i in indices:
            seg_file = all_curriculum_files[i]
            text = all_curriculum_transcriptions[i]
            target_sec = all_curriculum_target_seconds[i]
            target_wrd = all_curriculum_target_words[i]

            audio_features = load_mel_features(seg_file, target_seconds=target_sec)
            audio_batch = audio_features.transpose(0, 1).unsqueeze(0).to(device)

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
            print(f"⚠️  Warning: No valid training samples.")
            break

        avg_loss = total_loss / total_iterations

        if avg_loss > prev_loss:
            current_lr = optimizer.param_groups[0]['lr']
            new_lr = max(current_lr * 0.5, 1e-7)
            if new_lr != current_lr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                if new_lr > 1e-7:
                    print(f"  LR reduced to: {new_lr:.1e}")
                save_checkpoint(model, optimizer, epoch + 1, avg_loss, model_path, training_type="curriculum")

        # Track best loss (checkpoint saved by save_checkpoint when LR reduces)
        if avg_loss < best_loss:
            best_loss = avg_loss

        elapsed = time.time() - checkpoint_time
        time_str = f"{int(elapsed//60)}m" if elapsed >= 60 else f"{int(elapsed)}s"

        current_lr = optimizer.param_groups[0]['lr']
        accuracy_str = ""
        current_acc = 0
        if epoch == 0 or (epoch + 1) % 10 == 0:
            # Test on curriculum-appropriate samples (mixed stages)
            current_acc = calculate_curriculum_accuracy(
                model,
                all_curriculum_files,
                all_curriculum_transcriptions,
                all_curriculum_target_seconds,
                all_curriculum_target_words,
                vocab,
                device,
                sample_rate=8
            )
            accuracy_str = f" | Accuracy={current_acc:.0f}%"

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == 499:
            print(f"Epoch {epoch+1}/500 | Loss={avg_loss:.4f}{accuracy_str} | Time={time_str}")
            checkpoint_time = time.time()

        prev_loss = avg_loss

        if current_lr <= 1e-7:
            print(f"\n✓ Stopping: LR reached minimum")
            break

        if current_acc > 99.0:
            print(f"\n✓ Stopping: Accuracy > 99%")
            break

    save_checkpoint(model, optimizer, epoch + 1, avg_loss, model_path, training_type="curriculum")
    print(f"\nFinal model saved to: {model_path}")



if __name__ == "__main__":
    main()
