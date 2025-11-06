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
import json
import torch
import torch.nn as nn
import torchaudio
import glob
import os
import random
import time
import sys
sys.path.append("..")
from custom_scripts.encoder_decoder_transformer import EncoderDecoderTransformer

# ==============================================================
# Device setup
# ==============================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 Using Metal GPU (Apple Silicon)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("🚀 Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("⚠️  Using CPU (slower)")

print(f"Device: {device}")

# ==============================================================
# Configuration
# ==============================================================
CHUNK_DURATION = 1.3  # seconds per chunk
WORDS_PER_CHUNK = 1   # words per chunk

# ==============================================================
# Audio feature extraction
# ==============================================================
def extract_mel_features(audio_path, n_mels=80, target_seconds=None):
    """Extract mel features from audio, optionally trimming to target_seconds"""
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert stereo to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Trim to target seconds if specified
    if target_seconds is not None:
        num_samples = int(sample_rate * target_seconds)
        if waveform.shape[1] > num_samples:
            waveform = waveform[:, :num_samples]

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

    # Per-segment normalization
    mel_mean = mel_features.mean()
    mel_std = mel_features.std()
    mel_features = (mel_features - mel_mean) / (mel_std + 1e-8)

    return mel_features, sample_rate

# ==============================================================
# Tokenization
# ==============================================================
def tokenize_text(text, vocab):
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    words = text.split()
    return [word_to_idx.get(word, 0) for word in words]  # 0 = unknown

# ==============================================================
# Training for one segment with specific word count
# ==============================================================
def train_segment_curriculum(model, segment_file, transcription, vocab, word_count, target_seconds, num_epochs=100, learning_rate=1e-5):
    """
    Train model on a single segment for a specific number of words

    Args:
        segment_file: Path to audio segment
        transcription: Full transcription text
        word_count: Number of words to train on (from beginning)
        target_seconds: Audio duration to use
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    best_loss = float('inf')
    prev_loss = float('inf')
    patience_counter = 0
    min_delta = 1e-3
    patience = 3
    start_time = time.time()

    # Extract target text (first word_count words)
    words = transcription.split()
    target_text = " ".join(words[:word_count])

    for epoch in range(num_epochs):
        model.train()

        # Extract audio features
        audio_features, sample_rate = extract_mel_features(segment_file, target_seconds=target_seconds)

        # Tokenize target text
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

        avg_loss = loss.item()

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_marker = " ⭐"
        else:
            best_marker = ""

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"  Epoch {epoch+1}/{num_epochs} | Loss={avg_loss:.4f} | LR={current_lr:.6f} | Time={elapsed:.1f}s{best_marker}")

        # Early stopping check
        loss_change = prev_loss - avg_loss
        if loss_change < min_delta:  # Includes small improvement, no change, or getting worse
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  ⚠️  Early stopping: loss not improving for {patience} consecutive epochs")
                break
        else:
            patience_counter = 0
        prev_loss = avg_loss

        # Step scheduler
        scheduler.step()

    total_time = time.time() - start_time
    print(f"  ✓ Completed in {total_time:.1f}s | Best loss: {best_loss:.4f}")

    return model

# ==============================================================
# Old train_stage function (kept for compatibility, but not used)
# ==============================================================
def train_stage(model, segment_files, transcriptions, vocab, surah_part,
                stage_num, target_seconds, target_words, num_epochs=100, learning_rate=1e-5):
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

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    best_loss = float('inf')
    prev_loss = float('inf')
    patience_counter = 0  # Track consecutive epochs with low loss change
    min_delta = 1e-3  # Minimum change to consider as improvement
    patience = 3  # Number of epochs to wait before stopping
    start_time = time.time()

    # Build description
    audio_desc = f"{target_seconds:.1f}s" if target_seconds else "full"
    text_desc = f"{target_words} word(s)" if target_words else "full"
    checkpoint_name = f"checkpoint_curriculum_{surah_part}_stage{stage_num:02d}.pt"

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
            audio_features, sample_rate = extract_mel_features(seg_file, target_seconds=target_seconds)

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

        # Early stopping check
        loss_change = prev_loss - avg_loss
        if loss_change < min_delta:  # Includes small improvement, no change, or getting worse
            patience_counter += 1
            print(f"⚠️  Low/negative loss change ({loss_change:.6f}) | Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"⚠️  Early stopping: loss not improving for {patience} consecutive epochs")
                break
        else:
            patience_counter = 0  # Reset counter if loss change is significant
        prev_loss = avg_loss

        # Step the learning rate scheduler
        scheduler.step()

        # Sample generation every 5 epochs
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            model.eval()
            test_audio, sample_rate = extract_mel_features(segment_files[0], target_seconds=target_seconds)
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
                generated = model.generate(test_audio, max_new_tokens=max_tokens, audio_duration_seconds=audio_duration)
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

                print(f"  🔹 Generated: {' '.join(display_words)}")
                print(f"  🔸 Expected: {expected_text}")
            model.train()

    total_time = time.time() - start_time
    print(f"\n✓ Stage {stage_num} complete in {total_time:.1f}s | Best loss: {best_loss:.4f}")

    # Load best checkpoint
    if os.path.exists(checkpoint_name):
        print(f"✓ Loading best checkpoint from {checkpoint_name}...")
        checkpoint = torch.load(checkpoint_name, map_location=device)
        model.load_state_dict(checkpoint["model"])
        print(f"✓ Loaded best model from epoch {checkpoint['epoch'] + 1}")

    return model

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
    vocab_path = "../models/vocabulary.json"
    model_path = "../models/encoder_decoder_model.pt"

    print(f"\n{'='*60}")
    print(f"CURRICULUM LEARNING - SURAH PART: {surah_part}")
    print(f"Dataset: {dataset_name}")
    print(f"Chunk size: {CHUNK_DURATION}s → {WORDS_PER_CHUNK} word(s)")
    print(f"{'='*60}\n")

    # Load vocab
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"Vocabulary size: {len(vocab)}")

    # Parse surah part name to determine surah number
    surah_num = surah_part.split('-')[0]  # "001" or "002"

    # Load transcriptions and segments
    text_path = f"../datasets/{dataset_name}/text/{surah_part}.txt"
    if not os.path.exists(text_path):
        print(f"❌ Error: Text file not found: {text_path}")
        sys.exit(1)

    with open(text_path, "r", encoding="utf-8") as f:
        transcriptions = [line.strip() for line in f if line.strip()]

    segment_files = sorted(glob.glob(os.path.join(datasets_dir, surah_num, f"{surah_part}-*.wav")))

    if not segment_files:
        print(f"❌ Error: No audio segments found in {datasets_dir}/{surah_num}/{surah_part}-*.wav")
        sys.exit(1)

    print(f"Loaded {len(transcriptions)} transcriptions, {len(segment_files)} audio segments")

    if len(transcriptions) != len(segment_files):
        print(f"⚠️  Warning: Mismatch between transcriptions ({len(transcriptions)}) and segments ({len(segment_files)})")

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

    # Load existing model if available
    import shutil
    if os.path.exists(model_path):
        backup_path = model_path.replace(".pt", f"_backup_curriculum_{surah_part}.pt")
        shutil.copy2(model_path, backup_path)
        print(f"\n✓ Backup created: {backup_path}")

        print(f"Loading existing model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✓ Model loaded successfully! Starting curriculum training on {surah_part}.")
    else:
        print(f"\nNo existing model found. Starting with fresh weights for curriculum training.")

    # Train through curriculum: for each segment, train progressively on 1 word, 2 words, ..., all words
    total_start_time = time.time()

    print(f"\n{'='*60}")
    print(f"CURRICULUM TRAINING")
    print(f"{'='*60}\n")

    # Train on each segment individually with curriculum approach
    for seg_idx, (segment_file, transcription) in enumerate(zip(segment_files, transcriptions), 1):
        segment_name = os.path.basename(segment_file)
        words = transcription.split()
        num_words = len(words)

        # Get audio duration
        waveform, sample_rate = torchaudio.load(segment_file)
        audio_duration = waveform.shape[1] / sample_rate

        # Calculate how many chunks fit in this audio
        num_chunks = int(audio_duration / CHUNK_DURATION)

        # Don't exceed the number of words in the transcription
        max_chunks = min(num_chunks, num_words)

        print(f"\n{'='*60}")
        print(f"SEGMENT {seg_idx}/{len(segment_files)}: {segment_name}")
        print(f"Transcription: {transcription}")
        print(f"Audio duration: {audio_duration:.2f}s")
        print(f"Chunks that fit: {num_chunks} (1.3s each)")
        print(f"Words available: {num_words}")
        print(f"Training chunks: {max_chunks}")
        print(f"{'='*60}\n")

        # Train progressively: 1 chunk, 2 chunks, ..., max_chunks
        for chunk_count in range(1, max_chunks + 1):
            target_seconds = chunk_count * CHUNK_DURATION
            target_words = chunk_count * WORDS_PER_CHUNK

            print(f"\n--- Training on {chunk_count} chunk(s) = {target_seconds:.1f}s → {target_words} word(s) ---")

            model = train_segment_curriculum(
                model,
                segment_file,
                transcription,
                vocab,
                target_words,
                target_seconds,
                num_epochs=100,
                learning_rate=1e-5
            )

        print(f"\n✓ Completed curriculum training for {segment_name}\n")

    # Save final model
    torch.save(model.state_dict(), model_path)

    total_time = time.time() - total_start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)

    print(f"\n{'='*60}")
    print(f"✓ CURRICULUM TRAINING COMPLETED!")
    print(f"Total time: {minutes}m {seconds}s")
    print(f"Final model saved to: {model_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
