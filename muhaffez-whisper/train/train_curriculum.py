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
# Configuration
# ==============================================================
CHUNK_DURATION = 1.3  # seconds per chunk
WORDS_PER_CHUNK = 1   # words per chunk

# ==============================================================
# Mel feature loading
# ==============================================================
def load_mel_features(mel_path, target_seconds=None):
    """Load precomputed mel features from .pt file, optionally trimming to target_seconds"""
    if not os.path.exists(mel_path):
        raise FileNotFoundError(f"Precomputed mel features not found: {mel_path}\nPlease run precompute_mel_features.py first")

    mel_features = torch.load(mel_path, map_location='cpu', weights_only=True)

    # Trim to target seconds if specified
    # Mel features are at 100 fps (frames per second)
    if target_seconds is not None:
        target_frames = int(target_seconds * 100)
        if mel_features.shape[0] > target_frames:
            mel_features = mel_features[:target_frames, :]

    return mel_features

# ==============================================================
# Tokenization
# ==============================================================
def tokenize_text(text, vocab):
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    words = text.split()
    return [word_to_idx.get(word, 0) for word in words]  # 0 = unknown

# ==============================================================
# Comprehensive accuracy calculation for all segments
# ==============================================================
def calculate_comprehensive_accuracy(model, segment_files, transcriptions, vocab, target_seconds, target_words, device):
    """Calculate accuracy across all segments"""
    model.eval()
    total_correct = 0
    total_expected = 0
    segment_accuracies = []

    with torch.no_grad():
        for idx, (seg_file, transcription) in enumerate(zip(segment_files, transcriptions)):
            # Extract audio features
            audio_features = load_mel_features(seg_file, target_seconds=target_seconds)
            audio_batch = audio_features.transpose(0, 1).unsqueeze(0).to(device)

            # Get expected text
            expected_words = transcription.split()[:target_words] if target_words else transcription.split()
            expected_text = " ".join(expected_words)

            if not expected_text:
                continue

            # Generate with timeout protection
            max_tokens = min((target_words * 10) if target_words else 50, 100)  # Cap at 100 tokens

            try:
                generated = model.generate(audio_batch, max_new_tokens=max_tokens, audio_duration_seconds=target_seconds, use_sampling=False)
                generated_ids = generated[0].tolist()
            except Exception as e:
                print(f"    Warning: Generation failed for segment {idx}: {e}", flush=True)
                continue

            # Clean up generated IDs
            if generated_ids and generated_ids[0] == 1:
                generated_ids = generated_ids[1:]
            if 2 in generated_ids:
                generated_ids = generated_ids[:generated_ids.index(2)]

            generated_words = [vocab[idx] for idx in generated_ids if idx < len(vocab)]

            # Calculate confidence and filter low confidence words
            if len(generated_ids) > 0:
                encoder_output = model.encode(audio_batch)
                text_ids = torch.tensor([[1] + generated_ids[:len(generated_words)]], dtype=torch.long, device=device)
                logits, _ = model.decode(text_ids, encoder_output)
                probs = torch.softmax(logits, dim=-1)

                # Get confident words only (>= 20% threshold)
                confident_words = []
                for i, token_id in enumerate(generated_ids[:len(generated_words)]):
                    if i < logits.shape[1] - 1:
                        token_prob = probs[0, i, token_id].item()
                        if token_prob >= 0.2:  # 20% threshold
                            confident_words.append(generated_words[i])

                # Count correct confident words
                correct = sum(1 for i, word in enumerate(confident_words) if i < len(expected_words) and word == expected_words[i])
            else:
                correct = 0

            # Calculate segment accuracy
            segment_acc = (correct / len(expected_words) * 100) if expected_words else 0
            segment_accuracies.append(segment_acc)
            total_correct += correct
            total_expected += len(expected_words)

    # Calculate overall accuracy
    overall_accuracy = (total_correct / total_expected * 100) if total_expected > 0 else 0
    avg_segment_accuracy = sum(segment_accuracies) / len(segment_accuracies) if segment_accuracies else 0

    return overall_accuracy, avg_segment_accuracy, segment_accuracies

# ==============================================================
# Training for curriculum stage (all segments at specific chunk count)
# ==============================================================
def train_curriculum_stage(model, segment_files, transcriptions, vocab, surah_part,
                           stage_num, target_seconds, target_words, num_epochs=500, learning_rate=1e-3,
                           full_length_indices=None):
    """
    Train model on all segments for a specific curriculum stage

    Args:
        stage_num: The curriculum stage number (for logging)
        target_seconds: Audio duration to use for this stage
        target_words: Number of words to predict for this stage
        full_length_indices: Set of indices that should use full audio/text (for replay buffer)
    """
    if full_length_indices is None:
        full_length_indices = set()

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

    print(f"  Initial Learning Rate: {learning_rate:.1e}", flush=True)

    best_loss = float('inf')
    best_epoch = -1  # Track which epoch had the best loss
    best_accuracy = 0.0  # Track accuracy at best epoch
    best_model_state = None  # Track best model state
    prev_loss = float('inf')
    start_time = time.time()
    checkpoint_time = start_time  # Time of last checkpoint (for relative timing)

    # Calculate initial accuracy before training
    model.eval()
    overall_acc, avg_acc, seg_accuracies = calculate_comprehensive_accuracy(
        model, segment_files, transcriptions, vocab,
        target_seconds, target_words, device
    )
    print(f"  Initial accuracy: {overall_acc:.1f}%", flush=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_iterations = 0
        indices = list(range(len(segment_files)))
        random.shuffle(indices)

        for i in indices:
            seg_file = segment_files[i]
            text = transcriptions[i]

            # Determine if this is a full-length sample (from replay buffer)
            is_full_length = i in full_length_indices

            # Extract audio features (full or chunked)
            if is_full_length:
                # Full-length sample: use entire audio
                audio_features = load_mel_features(seg_file, target_seconds=None)
            else:
                # Chunked sample: use target_seconds
                audio_features = load_mel_features(seg_file, target_seconds=target_seconds)

            # Extract target text (full or chunked)
            words = text.split()
            if is_full_length:
                # Full-length sample: use entire transcription
                target_text = text
            else:
                # Chunked sample: use first target_words
                if len(words) < target_words:
                    continue  # Skip if not enough words
                target_text = " ".join(words[:target_words])

            if not target_text:
                continue

            text_tokens = tokenize_text(target_text, vocab)
            audio_batch = audio_features.transpose(0, 1).unsqueeze(0).to(device)

            full_sequence = [1] + text_tokens + [2]  # <s> + tokens + </s>
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
            print(f"  ⚠️  Warning: No valid training samples in this stage. Skipping.")
            break

        avg_loss = total_loss / total_iterations

        # Dynamic learning rate: reduce by 10% if loss increases
        if avg_loss > prev_loss:
            current_lr = optimizer.param_groups[0]['lr']
            new_lr = max(current_lr * 0.5, 1e-7)  # Minimum LR = 1e-7
            if new_lr != current_lr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                # Only print if LR actually changed
                if new_lr > 1e-7:
                    print(f"  Loss increased ({prev_loss:.4f} → {avg_loss:.4f}), reducing LR to: {new_lr:.1e}", flush=True)

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Calculate relative elapsed time since last checkpoint
        elapsed = time.time() - checkpoint_time
        current_lr = optimizer.param_groups[0]['lr']

        # Format elapsed time: seconds if < 60s, minutes if >= 60s
        if elapsed >= 60:
            time_str = f"{int(round(elapsed / 60))}m"
        else:
            time_str = f"{int(round(elapsed))}s"

        # Calculate accuracy after epoch 1 and every 10 epochs for display and early stopping
        accuracy_str = ""
        if epoch == 0 or (epoch + 1) % 10 == 0:
            # Save current model state
            current_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            # Load best model state for accuracy evaluation
            if best_model_state is not None:
                model.load_state_dict(best_model_state)

            # Calculate accuracy
            model.eval()
            overall_acc, avg_acc, seg_accuracies = calculate_comprehensive_accuracy(
                model, segment_files, transcriptions, vocab,
                target_seconds, target_words, device
            )

            # Build accuracy string for display
            accuracy_str = f" | Accuracy={overall_acc:.0f}%"

            # Update best accuracy when we have best model loaded
            if best_model_state is not None:
                # We just evaluated the best model, so this is the accuracy for best epoch
                best_accuracy = overall_acc

            # Restore current model to continue training
            model.load_state_dict({k: v.to(device) for k, v in current_model_state.items()})
            model.train()

        # Print epoch info: epoch 1, every 10 epochs, or last epoch
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            print(f"  Epoch {epoch+1}/{num_epochs} | Loss={avg_loss:.4f}{accuracy_str} | Time={time_str}")
            # Reset checkpoint time after printing
            checkpoint_time = time.time()

        prev_loss = avg_loss

        # Stop if learning rate reaches minimum
        if current_lr <= 1e-7:
            print(f"  ✓ Stopping: Learning rate reached minimum (1e-7)", flush=True)
            break

    total_time = time.time() - start_time

    # Format time: seconds if < 60s, minutes if >= 60s
    if total_time >= 60:
        print(f"  ✓ Stage {stage_num} completed in {int(round(total_time / 60))}m")
    else:
        print(f"  ✓ Stage {stage_num} completed in {int(round(total_time))}s")

    # Restore best model state before returning (only if not already loaded from early stopping)
    if best_model_state is not None:
        # Check if we need to restore (we might have already loaded it during early stopping)
        try:
            current_state = model.state_dict()
            # Only restore if current model is not the best model
            needs_restore = any(
                not torch.equal(current_state[k].cpu(), best_model_state[k])
                for k in best_model_state.keys()
            )
            if needs_restore:
                model.load_state_dict(best_model_state)
                print(f"  ✓ Restored best model state")
        except:
            # If comparison fails, just restore to be safe
            model.load_state_dict(best_model_state)
            print(f"  ✓ Restored best model state")

    return model

# ==============================================================
# Replay Buffer - Prevent Catastrophic Forgetting
# ==============================================================
def collect_replay_samples(dataset_name, current_surah_part, datasets_dir, current_set_size):
    """
    Collect a small sample from all previously trained surahs to prevent catastrophic forgetting.

    Args:
        dataset_name: Name of dataset (e.g., "Quran-A")
        current_surah_part: Current surah being trained (e.g., "002-01")
        datasets_dir: Path to datasets directory
        current_set_size: Size of current training set (to calculate 10% replay buffer with minimum 20)

    Returns:
        (replay_segment_files, replay_transcriptions): Lists of replay samples
    """
    current_surah_num = current_surah_part.split('-')[0]

    replay_segment_files = []
    replay_transcriptions = []

    # Find all text files for previous surah parts (including previous parts from same surah)
    text_dir = f"../datasets/{dataset_name}/text"
    all_text_files = sorted(glob.glob(os.path.join(text_dir, "*.txt")))

    # Count previous surah parts and total available samples
    # Only include parts that are < current_surah_part (not <=, we don't replay current)
    previous_surah_parts = []
    total_previous_samples = 0
    for text_file in all_text_files:
        basename = os.path.basename(text_file)
        surah_part = basename.replace('.txt', '')

        # Compare surah parts properly: "002-01" < "002-02" is TRUE
        if surah_part < current_surah_part:
            # Count samples in this surah part
            with open(text_file, "r", encoding="utf-8") as f:
                num_samples = len([line for line in f if line.strip()])
            previous_surah_parts.append((text_file, surah_part))
            total_previous_samples += num_samples

    if not previous_surah_parts:
        return replay_segment_files, replay_transcriptions

    # Calculate total replay buffer size as min(max(10% of current set, 30), total previous samples)
    total_replay_size = min(max(int(current_set_size * 0.1), 30), total_previous_samples)

    # Distribute replay budget evenly across previous surahs
    samples_per_surah = max(1, total_replay_size // len(previous_surah_parts))

    for text_file, surah_part in previous_surah_parts:
        # Load transcriptions
        with open(text_file, "r", encoding="utf-8") as f:
            transcriptions = [line.strip() for line in f if line.strip()]

        # Extract surah number for mels path
        surah_num = surah_part.split('-')[0]

        # Load corresponding mel files
        # Check if surah_part has multiple parts (e.g., "002-04")
        if '-' in surah_part and len(surah_part.split('-')) > 1 and surah_part.split('-')[1]:
            # Multi-part surah (e.g., "002-04") - look in subdirectory
            mel_files = sorted(glob.glob(os.path.join(f"../datasets/{dataset_name}/mels/normal", surah_num, surah_part, f"{surah_part}-*.pt")))
        else:
            # Single surah (e.g., "001") - look directly in surah folder
            mel_files = sorted(glob.glob(os.path.join(f"../datasets/{dataset_name}/mels/normal", surah_num, f"{surah_part}-*.pt")))

        # Fallback: try subdirectory if not found
        if not mel_files:
            mel_files = sorted(glob.glob(os.path.join(f"../datasets/{dataset_name}/mels/normal", surah_num, surah_part, f"{surah_part}-*.pt")))

        if len(mel_files) > 0 and len(mel_files) == len(transcriptions):
            # Sample up to samples_per_surah, but not more than available
            num_samples = min(samples_per_surah, len(mel_files))

            # Randomly sample indices
            indices = random.sample(range(len(mel_files)), num_samples)

            for idx in indices:
                replay_segment_files.append(mel_files[idx])
                replay_transcriptions.append(transcriptions[idx])

    if len(replay_segment_files) > 0:
        # Extract segment names and format as comma-separated list
        segment_names = [os.path.basename(f).replace('.pt', '') for f in replay_segment_files]
        segments_str = ', '.join(segment_names)
        print(f"  Replay buffer segments: ({segments_str})")
        print(f"  Replay buffer size: {len(replay_segment_files)}\n")

    return replay_segment_files, replay_transcriptions

def collect_full_length_replay_samples(dataset_name, current_surah_part, datasets_dir, current_set_size):
    """
    Collect full-length samples from PREVIOUS surahs for curriculum training.
    This helps the model not forget full-length patterns while learning chunked patterns.

    Args:
        dataset_name: Name of dataset (e.g., "Quran-A")
        current_surah_part: Current surah being trained (e.g., "002-01")
        datasets_dir: Path to datasets directory
        current_set_size: Size of current training set (to calculate 10% full-length replay buffer with minimum 20)

    Returns:
        (full_replay_segment_files, full_replay_transcriptions): Lists of full-length replay samples from previous parts only
    """
    full_replay_segment_files = []
    full_replay_transcriptions = []

    # Find all text files for previous surah parts only
    text_dir = f"../datasets/{dataset_name}/text"
    all_text_files = sorted(glob.glob(os.path.join(text_dir, "*.txt")))

    # Count previous surah parts and total available samples
    # Only include parts that are < current_surah_part (not <=, exclude current)
    relevant_surah_parts = []
    total_available_samples = 0
    for text_file in all_text_files:
        basename = os.path.basename(text_file)
        surah_part = basename.replace('.txt', '')

        # Compare surah parts properly: only include parts BEFORE current (not including current)
        if surah_part < current_surah_part:
            # Count samples in this surah part
            with open(text_file, "r", encoding="utf-8") as f:
                num_samples = len([line for line in f if line.strip()])
            relevant_surah_parts.append((text_file, surah_part))
            total_available_samples += num_samples

    if not relevant_surah_parts:
        return full_replay_segment_files, full_replay_transcriptions

    # Calculate full-length replay buffer size as min(max(10% of current set, 30), total available samples)
    total_full_replay_size = min(max(int(current_set_size * 0.1), 30), total_available_samples)

    # Distribute replay budget evenly
    samples_per_surah = max(1, total_full_replay_size // len(relevant_surah_parts))

    for text_file, surah_part in relevant_surah_parts:
        # Load transcriptions
        with open(text_file, "r", encoding="utf-8") as f:
            transcriptions = [line.strip() for line in f if line.strip()]

        # Extract surah number for mels path
        surah_num = surah_part.split('-')[0]

        # Load corresponding mel files
        # Check if surah_part has multiple parts (e.g., "002-04")
        if '-' in surah_part and len(surah_part.split('-')) > 1 and surah_part.split('-')[1]:
            # Multi-part surah (e.g., "002-04") - look in subdirectory
            mel_files = sorted(glob.glob(os.path.join(f"../datasets/{dataset_name}/mels/normal", surah_num, surah_part, f"{surah_part}-*.pt")))
        else:
            # Single surah (e.g., "001") - look directly in surah folder
            mel_files = sorted(glob.glob(os.path.join(f"../datasets/{dataset_name}/mels/normal", surah_num, f"{surah_part}-*.pt")))

        # Fallback: try subdirectory if not found
        if not mel_files:
            mel_files = sorted(glob.glob(os.path.join(f"../datasets/{dataset_name}/mels/normal", surah_num, surah_part, f"{surah_part}-*.pt")))

        if len(mel_files) > 0 and len(mel_files) == len(transcriptions):
            # Sample up to samples_per_surah, but not more than available
            num_samples = min(samples_per_surah, len(mel_files))

            # Randomly sample indices
            indices = random.sample(range(len(mel_files)), num_samples)

            for idx in indices:
                full_replay_segment_files.append(mel_files[idx])
                full_replay_transcriptions.append(transcriptions[idx])

    if len(full_replay_segment_files) > 0:
        # Extract segment names and format as comma-separated list
        segment_names = [os.path.basename(f).replace('.pt', '') for f in full_replay_segment_files]
        segments_str = ', '.join(segment_names)
        print(f"  Replay buffer segments: ({segments_str})")
        print(f"  Replay buffer size: {len(full_replay_segment_files)}\n")

    return full_replay_segment_files, full_replay_transcriptions

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
        # Train all parts mode
        train_all_parts(dataset_name)
    else:
        # Train single part mode
        train_single_part(dataset_name, surah_part)

def train_all_parts(dataset_name):
    """Train on ALL segments across ALL surah parts with curriculum learning"""
    vocab_path = "../models/vocabulary.json"
    model_path = "../models/muhaffez_whisper.pt"
    datasets_dir = f"../datasets/{dataset_name}"

    print(f"\n{'='*60}")
    print(f"CURRICULUM TRAINING ON ALL SEGMENTS - DATASET: {dataset_name}")
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
    all_segment_files = []
    all_transcriptions = []

    for text_file in text_files:
        surah_part = os.path.basename(text_file).replace('.txt', '')
        surah_num = surah_part.split('-')[0]

        # Load transcriptions
        with open(text_file, 'r', encoding='utf-8') as f:
            transcriptions = [line.strip() for line in f if line.strip()]

        # Find mel files (using normal mels, not augmented)
        if '-' in surah_part and len(surah_part.split('-')) > 1 and surah_part.split('-')[1]:
            segment_files = sorted(glob.glob(f"{datasets_dir}/mels/normal/{surah_num}/{surah_part}/{surah_part}-*.pt"))
        else:
            segment_files = sorted(glob.glob(f"{datasets_dir}/mels/normal/{surah_num}/{surah_part}-*.pt"))

        if not segment_files:
            print(f"⚠️  Skipping {surah_part}: no mel files found")
            continue

        all_segment_files.extend(segment_files)
        all_transcriptions.extend(transcriptions)

    if not all_segment_files:
        print(f"❌ No mel files found for any part!")
        sys.exit(1)

    print(f"Total segments: {len(all_segment_files)} across {len(text_files)} parts\n")

    # Calculate max words needed
    max_words = max(len(t.split()) for t in all_transcriptions)
    total_segments = len(all_segment_files)

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

    if os.path.exists(model_path):
        print(f"\nLoading existing model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"✓ Model loaded successfully!")
    else:
        print(f"\n⚠️  No existing model found. Starting from scratch.")

    model = model.to(device)

    total_start_time = time.time()

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

    print(f"  Initial Learning Rate: 1.0e-03")

    best_loss = float('inf')
    prev_loss = float('inf')
    start_time = time.time()
    checkpoint_time = start_time

    # Calculate initial accuracy (on full segments only)
    model.eval()
    overall_acc, avg_acc, seg_accuracies = calculate_comprehensive_accuracy(
        model, all_segment_files, all_transcriptions, vocab,
        None, None, device  # Full length
    )
    print(f"  Initial accuracy: {overall_acc:.1f}%", flush=True)

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
            print(f"  ⚠️  Warning: No valid training samples. Skipping.")
            break

        avg_loss = total_loss / total_iterations

        # Dynamic learning rate: reduce by 50% if loss increases
        if avg_loss > prev_loss:
            current_lr = optimizer.param_groups[0]['lr']
            new_lr = max(current_lr * 0.5, 1e-7)
            if new_lr != current_lr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                if new_lr > 1e-7:
                    lr_str = f"{new_lr:.0e}" if new_lr >= 1e-6 else f"{new_lr:.1e}"
                    print(f"  Loss increased ({prev_loss:.4f} → {avg_loss:.4f}), reducing LR to: {lr_str}", flush=True)

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_path)

        elapsed_from_checkpoint = time.time() - checkpoint_time
        if elapsed_from_checkpoint >= 60:
            time_str = f"{int(round(elapsed_from_checkpoint / 60))}m"
        else:
            time_str = f"{int(round(elapsed_from_checkpoint))}s"

        current_lr = optimizer.param_groups[0]['lr']

        # Calculate accuracy after epoch 1 and every 10 epochs
        accuracy_str = ""
        if epoch == 0 or (epoch + 1) % 10 == 0:
            overall_acc, avg_acc, seg_accuracies = calculate_comprehensive_accuracy(
                model, all_segment_files, all_transcriptions, vocab,
                None, None, device  # Full length
            )
            accuracy_str = f" | Accuracy={overall_acc:.0f}%"

        # Print epoch info: epoch 1, every 10 epochs, or last epoch
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == 499:
            print(f"  Epoch {epoch+1}/500 | Loss={avg_loss:.4f}{accuracy_str} | Time={time_str}")
            checkpoint_time = time.time()

        prev_loss = avg_loss

        # Stop if learning rate reaches minimum
        if current_lr <= 1e-7:
            print(f"  ✓ Stopping: Learning rate reached minimum (1e-7)", flush=True)
            break

    total_time = time.time() - start_time
    if total_time >= 60:
        print(f"  ✓ Training completed in {int(round(total_time / 60))}m")
    else:
        print(f"  ✓ Training completed in {int(round(total_time))}s")

    # Save best model
    torch.save(model.state_dict(), model_path)

    total_time = time.time() - total_start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)

    print(f"\n{'='*60}")
    print(f"✓ CURRICULUM TRAINING COMPLETED!")
    print(f"Total time: {minutes}m {seconds}s")
    print(f"Best model saved to: {model_path}")
    print(f"{'='*60}\n")

    # Load the saved model for final evaluation
    print("Loading saved model for final evaluation...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    final_acc = calculate_comprehensive_accuracy(model, all_segment_files, all_transcriptions, vocab, None, None, device)[0]
    print(f"\n📊 Final Evaluation (full audio):")
    print(f"   Accuracy: {final_acc:.0f}%\n")
    print(f"FINAL_ACCURACY: {final_acc:.0f}%")

def train_single_part(dataset_name, surah_part):
    """Train on a single surah part with curriculum learning"""

    datasets_dir = f"../datasets/{dataset_name}/audio"
    mels_dir = f"../datasets/{dataset_name}/mels/normal"
    vocab_path = "../models/vocabulary.json"
    model_path = "../models/muhaffez_whisper.pt"

    # Load vocab
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    print(f"\n{'='*60}")
    print(f"CURRICULUM LEARNING - SURAH PART: {surah_part}")
    print(f"Dataset: {dataset_name}")
    print(f"Vocabulary: {len(vocab)} words")
    print(f"Chunk size: {CHUNK_DURATION}s → {WORDS_PER_CHUNK} word(s)")
    print(f"{'='*60}")

    # Parse surah part name to determine surah number
    surah_num = surah_part.split('-')[0]  # "001" or "002"

    # Load transcriptions and segments
    text_path = f"../datasets/{dataset_name}/text/{surah_part}.txt"
    if not os.path.exists(text_path):
        print(f"❌ Error: Text file not found: {text_path}")
        sys.exit(1)

    with open(text_path, "r", encoding="utf-8") as f:
        transcriptions = [line.strip() for line in f if line.strip()]

    # Determine mel directory based on segment structure
    # If surah_part has parts (e.g., "002-04"), look in subdirectory
    if '-' in surah_part and len(surah_part.split('-')) > 1 and surah_part.split('-')[1]:
        # Multi-part surah (e.g., "002-04")
        segment_files = sorted(glob.glob(os.path.join(mels_dir, surah_num, surah_part, f"{surah_part}-*.pt")))
    else:
        # Single surah (e.g., "001")
        segment_files = sorted(glob.glob(os.path.join(mels_dir, surah_num, f"{surah_part}-*.pt")))

    if not segment_files:
        # Try the subdirectory path as fallback
        segment_files = sorted(glob.glob(os.path.join(mels_dir, surah_num, surah_part, f"{surah_part}-*.pt")))
        if not segment_files:
            print(f"❌ Error: No mel files found in {mels_dir}/{surah_num}/{surah_part}-*.pt")
            print(f"       or {mels_dir}/{surah_num}/{surah_part}/{surah_part}-*.pt")
            sys.exit(1)

    print(f"Loaded {len(transcriptions)} transcriptions, {len(segment_files)} mel files")

    if len(transcriptions) != len(segment_files):
        print(f"⚠️  Warning: Mismatch between transcriptions ({len(transcriptions)}) and segments ({len(segment_files)})")

    # Note: Replay buffer samples will be collected fresh for each curriculum stage
    # This ensures variety and prevents overfitting to the same replay samples

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

    # Load existing model if available (no backup - already done by train_full.py)
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✓ Model loaded successfully! Starting curriculum training on {surah_part}.")
    else:
        print(f"\nNo existing model found. Starting with fresh weights for curriculum training.")

    # Train through curriculum: train all segments at each stage
    total_start_time = time.time()

    print(f"\n{'='*60}")
    print(f"CURRICULUM TRAINING")
    print(f"{'='*60}\n")

    # Calculate max chunks across all segments
    segment_info = []
    for segment_file, transcription in zip(segment_files, transcriptions):
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

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

    print(f"  Initial Learning Rate: 1.0e-03")

    best_loss = float('inf')
    prev_loss = float('inf')
    start_time = time.time()
    checkpoint_time = start_time

    # Calculate initial accuracy (on full segments only)
    model.eval()
    overall_acc, avg_acc, seg_accuracies = calculate_comprehensive_accuracy(
        model, segment_files, transcriptions, vocab,
        None, None, device  # Full length
    )
    print(f"  Initial accuracy: {overall_acc:.1f}%", flush=True)

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
            print(f"  ⚠️  Warning: No valid training samples. Skipping.")
            break

        avg_loss = total_loss / total_iterations

        # Dynamic learning rate: reduce by 50% if loss increases
        if avg_loss > prev_loss:
            current_lr = optimizer.param_groups[0]['lr']
            new_lr = max(current_lr * 0.5, 1e-7)
            if new_lr != current_lr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                if new_lr > 1e-7:
                    lr_str = f"{new_lr:.0e}" if new_lr >= 1e-6 else f"{new_lr:.1e}"
                    print(f"  Loss increased ({prev_loss:.4f} → {avg_loss:.4f}), reducing LR to: {lr_str}", flush=True)

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_path)

        elapsed_from_checkpoint = time.time() - checkpoint_time
        if elapsed_from_checkpoint >= 60:
            time_str = f"{int(round(elapsed_from_checkpoint / 60))}m"
        else:
            time_str = f"{int(round(elapsed_from_checkpoint))}s"

        current_lr = optimizer.param_groups[0]['lr']

        # Calculate accuracy after epoch 1 and every 10 epochs
        accuracy_str = ""
        if epoch == 0 or (epoch + 1) % 10 == 0:
            overall_acc, avg_acc, seg_accuracies = calculate_comprehensive_accuracy(
                model, segment_files, transcriptions, vocab,
                None, None, device  # Full length
            )
            accuracy_str = f" | Accuracy={overall_acc:.0f}%"

        # Print epoch info: epoch 1, every 10 epochs, or last epoch
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == 499:
            print(f"  Epoch {epoch+1}/500 | Loss={avg_loss:.4f}{accuracy_str} | Time={time_str}")
            checkpoint_time = time.time()

        prev_loss = avg_loss

        # Stop if learning rate reaches minimum
        if current_lr <= 1e-7:
            print(f"  ✓ Stopping: Learning rate reached minimum (1e-7)", flush=True)
            break

    total_time = time.time() - start_time
    if total_time >= 60:
        print(f"  ✓ Training completed in {int(round(total_time / 60))}m")
    else:
        print(f"  ✓ Training completed in {int(round(total_time))}s")

    # Save best model
    torch.save(model.state_dict(), model_path)

    total_time = time.time() - total_start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)

    print(f"\n{'='*60}")
    print(f"✓ CURRICULUM TRAINING COMPLETED!")
    print(f"Total time: {minutes}m {seconds}s")
    print(f"Best model saved to: {model_path}")
    print(f"{'='*60}\n")

    # Calculate final accuracy for train.sh to capture
    model.eval()
    overall_acc = calculate_comprehensive_accuracy(
        model, segment_files, transcriptions, vocab,
        target_seconds=None, target_words=None, device=device
    )[0]
    print(f"FINAL_ACCURACY: {overall_acc:.0f}%")


if __name__ == "__main__":
    main()
