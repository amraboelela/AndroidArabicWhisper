#!/usr/bin/env python3
"""
Curriculum training script - trains incrementally on chunks
Usage: python3 train_curriculum.py <dataset_name> <surah_part>

Examples:
  python3 train_curriculum.py Quran-A 002-04
  python3 train_curriculum.py Quran-A 001

This script trains the model using curriculum learning on all segments in a surah part:
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
                logits = model.decode(text_ids, encoder_output)
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

    # If already perfect, no need to train
    if overall_acc > 90.0:
        print(f"  ✓ Model already at {overall_acc:.1f}% accuracy. Skipping training.", flush=True)
        return model

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
            new_lr = max(current_lr * 0.9, 1e-7)  # Minimum LR = 1e-7
            if new_lr != current_lr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                # Only print if LR actually changed
                if new_lr > 1e-7:
                    # Format LR without trailing zero (e.g., 9e-04 instead of 9.0e-04)
                    lr_str = f"{new_lr:.0e}" if new_lr >= 1e-6 else f"{new_lr:.1e}"
                    print(f"  Loss increased ({prev_loss:.4f} → {avg_loss:.4f}), reducing LR to: {lr_str}", flush=True)

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

            # Early stopping if accuracy > 90%
            if overall_acc > 90.0:
                # Print current epoch info with accuracy before stopping
                print(f"  Epoch {epoch+1}/{num_epochs} | Loss={avg_loss:.4f}{accuracy_str} | Time={time_str}", flush=True)
                print(f"  ✓ Early stopping: accuracy {overall_acc:.1f}% at epoch {best_epoch}", flush=True)
                # Keep best model loaded, don't restore
                break

            # Restore current model to continue training
            model.load_state_dict({k: v.to(device) for k, v in current_model_state.items()})
            model.train()

        # Print epoch info: epoch 1, every 10 epochs, or last epoch
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            print(f"  Epoch {epoch+1}/{num_epochs} | Loss={avg_loss:.4f}{accuracy_str} | Time={time_str}")
            # Reset checkpoint time after printing
            checkpoint_time = time.time()

        prev_loss = avg_loss

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
# Old train_stage function (kept for compatibility, but not used)
# ==============================================================
def train_stage(model, segment_files, transcriptions, vocab, surah_part,
                stage_num, target_seconds, target_words, num_epochs=500, learning_rate=1e-5):
    """
    Train model on one curriculum stage

    Args:
        stage_num: The curriculum stage number (for logging)
        target_seconds: Audio duration to use (None = full)
        target_words: Number of words to predict (None = all)
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

    best_loss = float('inf')
    best_epoch = -1
    prev_loss = float('inf')
    start_time = time.time()

    # Build description
    audio_desc = f"{target_seconds:.1f}s" if target_seconds else "full"
    text_desc = f"{target_words} word(s)" if target_words else "full"
    checkpoint_name = "checkpoint_best.pt"

    print(f"\n{'='*60}")
    print(f"CURRICULUM STAGE {stage_num}")
    print(f"Audio: {audio_desc} → Text: {text_desc}")
    print(f"{'='*60}\n")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_iterations = 0
        indices = list(range(len(segment_files)))
        random.shuffle(indices)

        for i in indices:
            seg_file = segment_files[i]
            text = transcriptions[i]

            # Extract audio features
            audio_features = load_mel_features(seg_file, target_seconds=target_seconds)

            # Extract target text
            if target_words:
                words = text.split()
                # Only process if we have enough words
                if len(words) < target_words:
                    continue
                target_text = " ".join(words[:target_words])
            else:
                target_text = text

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
            print(f"⚠️  Warning: No valid training samples in this stage. Skipping.")
            break

        avg_loss = total_loss / total_iterations

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "stage": stage_num
            }, checkpoint_name)
            best_marker = " ⭐ NEW BEST!"
        else:
            best_marker = ""

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} | Loss={avg_loss:.4f} | LR={current_lr:.6f} | Time={elapsed:.1f}s{best_marker}")

        # Calculate accuracy every 50 epochs and check for early stopping
        if (epoch + 1) % 50 == 0:
            # Load best checkpoint for accuracy evaluation
            if os.path.exists(checkpoint_name):
                checkpoint = torch.load(checkpoint_name, map_location=device)
                model.load_state_dict(checkpoint["model"])

            # Calculate accuracy
            overall_acc, avg_acc, seg_accuracies = calculate_comprehensive_accuracy(
                model, segment_files, transcriptions, vocab,
                target_seconds, target_words, device
            )
            print(f"  Accuracy at epoch {epoch+1}: {overall_acc:.1f}%")

            # Early stopping if accuracy > 90%
            if overall_acc > 90.0:
                print(f"✓ Early stopping: accuracy {overall_acc:.1f}% exceeds 90% threshold")
                break

        prev_loss = avg_loss

        # Sample generation every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            model.eval()
            test_audio = load_mel_features(segment_files[0], target_seconds=target_seconds)
            test_audio = test_audio.transpose(0, 1).unsqueeze(0).to(device)

            # Get expected text
            if target_words:
                words = transcriptions[0].split()
                expected_text = " ".join(words[:target_words]) if len(words) >= target_words else transcriptions[0]
            else:
                expected_text = transcriptions[0]

            # Calculate audio duration for generation
            if target_seconds:
                audio_duration = target_seconds
            else:
                waveform, sr = torchaudio.load(segment_files[0])
                audio_duration = waveform.shape[1] / sr

            with torch.no_grad():
                max_tokens = (target_words * 10) if target_words else 50
                generated = model.generate(test_audio, max_new_tokens=max_tokens, audio_duration_seconds=audio_duration, use_sampling=False)
                generated_ids = generated[0].tolist()
                if generated_ids and generated_ids[0] == 1:
                    generated_ids = generated_ids[1:]
                if 2 in generated_ids:
                    generated_ids = generated_ids[:generated_ids.index(2)]
                generated_words = [vocab[idx] for idx in generated_ids if idx < len(vocab)]

                # Show only target words if specified
                if target_words:
                    display_words = generated_words[:target_words] if len(generated_words) >= target_words else generated_words
                else:
                    display_words = generated_words

                print(f"  🔸 Expected: {expected_text}")
                print(f"  🔹 Generated: {' '.join(display_words)}")
            model.train()

    total_time = time.time() - start_time
    print(f"\n✓ Stage {stage_num} complete in {total_time:.1f}s")

    # Load best checkpoint
    if os.path.exists(checkpoint_name):
        print(f"✓ Loading best checkpoint from {checkpoint_name}...")
        checkpoint = torch.load(checkpoint_name, map_location=device)
        model.load_state_dict(checkpoint["model"])
        print(f"✓ Loaded best model from epoch {checkpoint['epoch'] + 1}")

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
            mel_files = sorted(glob.glob(os.path.join(f"../datasets/{dataset_name}/mels", surah_num, surah_part, f"{surah_part}-*.pt")))
        else:
            # Single surah (e.g., "001") - look directly in surah folder
            mel_files = sorted(glob.glob(os.path.join(f"../datasets/{dataset_name}/mels", surah_num, f"{surah_part}-*.pt")))

        # Fallback: try subdirectory if not found
        if not mel_files:
            mel_files = sorted(glob.glob(os.path.join(f"../datasets/{dataset_name}/mels", surah_num, surah_part, f"{surah_part}-*.pt")))

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
    Collect full-length samples from current and previous surahs for curriculum training.
    This helps the model not forget full-length patterns while learning chunked patterns.

    Args:
        dataset_name: Name of dataset (e.g., "Quran-A")
        current_surah_part: Current surah being trained (e.g., "002-01")
        datasets_dir: Path to datasets directory
        current_set_size: Size of current training set (to calculate 10% full-length replay buffer with minimum 20)

    Returns:
        (full_replay_segment_files, full_replay_transcriptions): Lists of full-length replay samples
    """
    full_replay_segment_files = []
    full_replay_transcriptions = []

    # Find all text files for current and previous surah parts
    text_dir = f"../datasets/{dataset_name}/text"
    all_text_files = sorted(glob.glob(os.path.join(text_dir, "*.txt")))

    # Count current and previous surah parts and total available samples
    # Only include parts that are <= current_surah_part
    relevant_surah_parts = []
    total_available_samples = 0
    for text_file in all_text_files:
        basename = os.path.basename(text_file)
        surah_part = basename.replace('.txt', '')

        # Compare surah parts properly: "002-01" <= "002-01" but "002-02" > "002-01"
        if surah_part <= current_surah_part:
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
            mel_files = sorted(glob.glob(os.path.join(f"../datasets/{dataset_name}/mels", surah_num, surah_part, f"{surah_part}-*.pt")))
        else:
            # Single surah (e.g., "001") - look directly in surah folder
            mel_files = sorted(glob.glob(os.path.join(f"../datasets/{dataset_name}/mels", surah_num, f"{surah_part}-*.pt")))

        # Fallback: try subdirectory if not found
        if not mel_files:
            mel_files = sorted(glob.glob(os.path.join(f"../datasets/{dataset_name}/mels", surah_num, surah_part, f"{surah_part}-*.pt")))

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
        print("Usage: python3 train_curriculum.py <dataset_name> <surah_part>")
        print("Examples:")
        print("  python3 train_curriculum.py Quran-A 002-04")
        print("  python3 train_curriculum.py Quran-A 001")
        sys.exit(1)

    dataset_name = sys.argv[1]  # e.g., "Quran-A"
    surah_part = sys.argv[2]  # e.g., "001", "002-01", "002-04"

    datasets_dir = f"../datasets/{dataset_name}/audio"
    mels_dir = f"../datasets/{dataset_name}/mels"
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

        # Get audio duration
        waveform, sample_rate = torchaudio.load(segment_file)
        audio_duration = waveform.shape[1] / sample_rate

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

    # Train stage by stage: all segments at 1 chunk, then all segments at 2 chunks, etc.
    # Include the last stage (full audio) to complete the curriculum
    for chunk_count in range(1, global_max_chunks + 1):
        target_seconds = chunk_count * CHUNK_DURATION
        target_words = chunk_count * WORDS_PER_CHUNK

        # Print stage header
        print(f"\n{'='*60}")
        print(f"Part: {surah_part}, Curriculum Stage: {chunk_count}/{global_max_chunks}")
        print(f"Training all segments: {target_seconds:.1f}s → {target_words} word(s)")
        print(f"{'='*60}\n")

        # Include ALL segments: chunk long ones, use full-length for short ones
        stage_segment_files = []
        stage_transcriptions = []
        short_segment_indices = []  # Indices of segments shorter than current stage

        for info in segment_info:
            stage_segment_files.append(info['file'])
            stage_transcriptions.append(info['transcription'])
            # Mark segments that are shorter than current chunk_count as full-length
            if chunk_count > info['max_chunks']:
                short_segment_indices.append(len(stage_segment_files) - 1)

        if not stage_segment_files:
            print(f"  ⚠️  No segments available for this stage. Skipping.")
            continue

        # Collect fresh random replay samples for this stage (all full-length)
        # This ensures variety and prevents overfitting to the same replay samples
        # Replay buffer = 10% of current stage size, all samples are full-length
        stage_full_replay_files, stage_full_replay_transcriptions = collect_full_length_replay_samples(
            dataset_name, surah_part, datasets_dir, len(stage_segment_files)
        )

        # Add replay buffer to this stage (current segments + full-length replay)
        all_stage_files = stage_segment_files + stage_full_replay_files
        all_stage_transcriptions = stage_transcriptions + stage_full_replay_transcriptions

        # Mark which indices are full-length samples:
        # 1. Short segments (shorter than current stage)
        # 2. Replay buffer samples (start after all stage segments)
        full_length_indices = set(short_segment_indices)
        replay_buffer_start = len(stage_segment_files)
        full_length_indices.update(range(replay_buffer_start, len(all_stage_files)))

        # Count how many segments are being chunked vs full-length
        num_chunked = len(stage_segment_files) - len(short_segment_indices)
        num_short = len(short_segment_indices)

        # Build training status message
        if num_short > 0:
            print(f"  Training on {num_chunked} chunked segments + {num_short} short segments (full) + {len(stage_full_replay_files)} replay buffer segments", flush=True)
        else:
            print(f"  Training on {num_chunked} chunked segments + {len(stage_full_replay_files)} replay buffer segments", flush=True)

        model = train_curriculum_stage(
            model,
            all_stage_files,
            all_stage_transcriptions,
            vocab,
            surah_part,
            chunk_count,
            target_seconds,
            target_words,
            num_epochs=500,
            learning_rate=1e-3,
            full_length_indices=full_length_indices
        )

        # Calculate comprehensive accuracy for this stage on all segments
        model.eval()
        overall_acc, avg_acc, seg_accuracies = calculate_comprehensive_accuracy(
            model, stage_segment_files, stage_transcriptions, vocab,
            target_seconds, target_words, device
        )

        print(f"  Accuracy: {overall_acc:.0f}%", flush=True)

        # Show one sample for visualization
        test_idx = random.randint(0, len(stage_segment_files) - 1)
        test_audio_features = load_mel_features(stage_segment_files[test_idx], target_seconds=target_seconds)
        test_audio_batch = test_audio_features.transpose(0, 1).unsqueeze(0).to(device)

        # Get expected text (first target_words words)
        words = stage_transcriptions[test_idx].split()
        expected_text = " ".join(words[:target_words]) if len(words) >= target_words else stage_transcriptions[test_idx]

        with torch.no_grad():
            max_tokens = target_words * 10
            generated = model.generate(test_audio_batch, max_new_tokens=max_tokens, audio_duration_seconds=target_seconds, use_sampling=False)
            generated_ids = generated[0].tolist()
            if generated_ids and generated_ids[0] == 1:
                generated_ids = generated_ids[1:]
            if 2 in generated_ids:
                generated_ids = generated_ids[:generated_ids.index(2)]
            generated_words = [vocab[idx] for idx in generated_ids if idx < len(vocab)]
            display_words = generated_words[:target_words] if len(generated_words) >= target_words else generated_words

            # Calculate confidence for each token
            # Re-run forward pass to get probabilities
            if len(generated_ids[:target_words]) == 0:
                # No tokens generated
                display_text = ""
                print(f"  🔸 Sample Expected: {expected_text}", flush=True)
                print(f"  🔹 Sample Generated: {display_text}", flush=True)
                print(f"     Sample Confidence: N/A\n", flush=True)
                model.train()
                continue

            encoder_output = model.encode(test_audio_batch)
            text_ids = torch.tensor([[1] + generated_ids[:target_words]], dtype=torch.long, device=device)
            logits = model.decode(text_ids, encoder_output)
            probs = torch.softmax(logits, dim=-1)

            # Get probability of each generated token
            min_confidence = 1.0
            token_confidences = []
            for i, token_id in enumerate(generated_ids[:len(display_words)]):  # Only check generated words
                if i < logits.shape[1] - 1:  # -1 because we prepended <s>
                    token_prob = probs[0, i, token_id].item()
                    token_confidences.append(token_prob)
                    min_confidence = min(min_confidence, token_prob)

            # Calculate accuracy (percentage of correct words)
            expected_words = expected_text.split()
            correct_words = sum(1 for i, word in enumerate(display_words) if i < len(expected_words) and word == expected_words[i])
            accuracy = (correct_words / len(expected_words) * 100) if expected_words else 0

            # Build display text and confidence text
            confidence_threshold = 0.2  # 20% threshold
            if len(display_words) == len(token_confidences):
                # Show words (mark low confidence with brackets, hide 0% or very low confidence)
                display_text_parts = []
                confidence_list = []
                correct_confident_words = 0
                total_confident_words = 0
                for i, (word, conf) in enumerate(zip(display_words, token_confidences)):
                    if conf < 0.01:  # Hide words with < 1% confidence (rounds to 0%)
                        # Skip words with very low confidence
                        continue
                    elif conf >= confidence_threshold:
                        display_text_parts.append(word)
                        confidence_list.append(f"{conf:.0%}")
                        # Count for accuracy only if confidence >= threshold
                        total_confident_words += 1
                        if i < len(expected_words) and word == expected_words[i]:
                            correct_confident_words += 1
                    else:
                        display_text_parts.append(f"[{word}]")  # Mark low confidence with brackets
                        confidence_list.append(f"{conf:.0%}")

                display_text = ' '.join(display_text_parts) if display_text_parts else ""
                confidence_text = ', '.join(confidence_list) if confidence_list else "N/A"

                # Accuracy: correct confident words out of total EXPECTED words
                accuracy = (correct_confident_words / len(expected_words) * 100) if expected_words else 0
            else:
                display_text = ' '.join(display_words)
                confidence_text = "N/A"
                # Original accuracy calculation if confidences don't match
                accuracy = (correct_words / len(expected_words) * 100) if expected_words else 0

            print(f"  🔸 Sample Expected: {expected_text}", flush=True)
            print(f"  🔹 Sample Generated: {display_text}", flush=True)
            print(f"     Sample Confidence: {confidence_text}\n", flush=True)
        model.train()

    # Save best model (restored from best checkpoint in train_segment_curriculum)
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

    # Calculate comprehensive accuracy on all segments (full audio)
    print(f"\n📊 Final Evaluation (full audio):")
    overall_acc, avg_acc, seg_accuracies = calculate_comprehensive_accuracy(
        model, segment_files, transcriptions, vocab,
        target_seconds=None, target_words=None, device=device
    )
    print(f"   Accuracy: {overall_acc:.0f}%\n", flush=True)

    # Output final accuracy for train.sh to capture
    print(f"FINAL_ACCURACY: {overall_acc:.0f}%")

    # Sample generation at the end (first segment for consistency)
    test_audio_features = load_mel_features(segment_files[0])
    test_audio_batch = test_audio_features.transpose(0, 1).unsqueeze(0).to(device)

    waveform, sr = torchaudio.load(segment_files[0])
    audio_duration = waveform.shape[1] / sr

    expected_text = transcriptions[0]

    with torch.no_grad():
        generated = model.generate(test_audio_batch, max_new_tokens=50, audio_duration_seconds=audio_duration, use_sampling=False)
        generated_ids = generated[0].tolist()
        if generated_ids and generated_ids[0] == 1:
            generated_ids = generated_ids[1:]
        if 2 in generated_ids:
            generated_ids = generated_ids[:generated_ids.index(2)]
        generated_words = [vocab[idx] for idx in generated_ids if idx < len(vocab)]

        # Calculate confidence for each token
        encoder_output = model.encode(test_audio_batch)
        text_ids = torch.tensor([[1] + generated_ids], dtype=torch.long, device=device)
        logits = model.decode(text_ids, encoder_output)
        probs = torch.softmax(logits, dim=-1)

        # Get probability of each generated token
        token_confidences = []
        for i, token_id in enumerate(generated_ids):
            if i < logits.shape[1] - 1:  # -1 because we prepended <s>
                token_prob = probs[0, i, token_id].item()
                token_confidences.append(token_prob)

        # Calculate accuracy (percentage of correct words)
        expected_words = expected_text.split()
        correct_words = sum(1 for i, word in enumerate(generated_words) if i < len(expected_words) and word == expected_words[i])
        accuracy = (correct_words / len(expected_words) * 100) if expected_words else 0

        # Filter out words with 0% confidence and calculate accuracy based on confident words
        filtered_words = []
        filtered_confidences = []
        correct_confident_words = 0
        total_confident_words = 0

        if len(generated_words) == len(token_confidences):
            for i, (word, conf) in enumerate(zip(generated_words, token_confidences)):
                if conf < 0.01:  # Hide words with < 1% confidence (rounds to 0%)
                    continue
                elif conf >= 0.2:  # 20% threshold
                    filtered_words.append(word)
                    filtered_confidences.append(f'{conf:.0%}')
                    total_confident_words += 1
                    if i < len(expected_words) and word == expected_words[i]:
                        correct_confident_words += 1
                else:
                    filtered_words.append(f"[{word}]")  # Mark low confidence with brackets
                    filtered_confidences.append(f'{conf:.0%}')

            # Accuracy: correct confident words out of total EXPECTED words
            accuracy = (correct_confident_words / len(expected_words) * 100) if expected_words else 0
        else:
            filtered_words = generated_words
            filtered_confidences = [f'{c:.0%}' for c in token_confidences] if token_confidences else []
            # Original accuracy calculation if confidences don't match
            accuracy = (correct_words / len(expected_words) * 100) if expected_words else 0

        # Build confidence text
        if filtered_confidences:
            confidence_text = ', '.join(filtered_confidences)
        else:
            confidence_text = "N/A"

        display_text = ' '.join(filtered_words) if filtered_words else ""

        print(f"🔸 Expected: {expected_text}")
        print(f"🔹 Generated: {display_text}")
        print(f"   Confidence: {confidence_text}")


if __name__ == "__main__":
    main()
