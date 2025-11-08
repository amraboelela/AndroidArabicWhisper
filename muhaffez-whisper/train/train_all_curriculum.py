#!/usr/bin/env python3
"""
Train on ALL segments across ALL surah parts with curriculum learning
Usage: python3 train_all_curriculum.py <dataset_name>
Example:
  python3 train_all_curriculum.py Quran-A
"""
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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
from custom_scripts.encoder_decoder_transformer import EncoderDecoderTransformer

# Device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

CHUNK_DURATION = 1.3
WORDS_PER_CHUNK = 1

def extract_mel_features(audio_path, n_mels=80, target_seconds=None):
    """Extract mel features from audio"""
    waveform, sample_rate = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
        sample_rate = target_sample_rate

    if target_seconds is not None:
        num_samples = int(sample_rate * target_seconds)
        if waveform.shape[1] > num_samples:
            waveform = waveform[:, :num_samples]

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

    mel_mean = -4.2677
    mel_std = 4.5689
    mel_features = (mel_features - mel_mean) / (mel_std + 1e-8)

    return mel_features, sample_rate

def tokenize_text(text, vocab):
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    words = text.split()
    return [word_to_idx.get(word, 0) for word in words]

def calculate_accuracy(model, segment_files, transcriptions, vocab, target_seconds, target_words, device):
    """Calculate accuracy for current curriculum stage"""
    model.eval()
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for seg_file, full_text in zip(segment_files, transcriptions):
            # Get expected text for this stage
            expected_words = full_text.split()[:target_words] if target_words else full_text.split()
            if not expected_words:
                continue

            # Extract features
            mel_features, _ = extract_mel_features(seg_file, target_seconds=target_seconds)
            audio_batch = mel_features.transpose(0, 1).unsqueeze(0).to(device)

            # Generate
            max_tokens = (target_words * 10) if target_words else 50
            generated = model.generate(audio_batch, max_new_tokens=max_tokens, audio_duration_seconds=target_seconds, use_sampling=False)
            tokens = generated[0].tolist()

            if tokens and tokens[0] == 1:
                tokens = tokens[1:]
            if 2 in tokens:
                tokens = tokens[:tokens.index(2)]

            generated_words = [vocab[idx] for idx in tokens if idx < len(vocab)]

            # Compare only the words for this stage
            if target_words:
                generated_words = generated_words[:target_words]

            min_len = min(len(generated_words), len(expected_words))
            total_correct += sum(1 for i in range(min_len) if generated_words[i] == expected_words[i])
            total_tokens += len(expected_words)

    accuracy = (total_correct / total_tokens * 100) if total_tokens > 0 else 0.0
    return accuracy

def train_curriculum_stage(model, segment_files, transcriptions, vocab, stage_num, target_seconds, target_words, replay_files, replay_texts, device):
    """Train one curriculum stage with replay buffer"""
    learning_rate = 1e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

    # Combine current stage data with replay buffer (10%)
    all_files = segment_files + replay_files
    all_texts = transcriptions + replay_texts

    print(f"\n{'='*60}")
    print(f"CURRICULUM STAGE {stage_num}")
    print(f"Audio: {target_seconds:.1f}s → Text: {target_words} word(s)")
    print(f"Training on {len(segment_files)} current + {len(replay_files)} replay = {len(all_files)} total segments")
    print(f"{'='*60}")

    best_loss = float('inf')
    prev_loss = float('inf')
    start_time = time.time()

    for epoch in range(500):
        model.train()
        total_loss = 0.0
        total_iterations = 0

        indices = list(range(len(all_files)))
        random.shuffle(indices)

        for i in indices:
            seg_file = all_files[i]
            text = all_texts[i]

            # Extract features
            mel_features, _ = extract_mel_features(seg_file, target_seconds=target_seconds)
            audio_batch = mel_features.transpose(0, 1).unsqueeze(0).to(device)

            # Get target text for this stage
            words = text.split()
            if target_words and len(words) < target_words:
                continue
            target_text = " ".join(words[:target_words]) if target_words else text

            if not target_text:
                continue

            # Tokenize
            text_tokens = tokenize_text(target_text, vocab)
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

        if total_iterations == 0:
            print(f"⚠️  No valid training samples in this stage. Skipping.")
            break

        avg_loss = total_loss / total_iterations
        elapsed = time.time() - start_time

        if avg_loss < best_loss:
            best_loss = avg_loss

        # Decay LR if loss increases
        if avg_loss > prev_loss:
            old_lr = optimizer.param_groups[0]['lr']
            new_lr = max(old_lr * 0.5, 1e-7)
            if new_lr != old_lr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f"  Learning rate reduced: {old_lr:.1e} → {new_lr:.1e}")

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} | Loss={avg_loss:.4f} | LR={current_lr:.1e} | Time={elapsed:.0f}s", flush=True)

        prev_loss = avg_loss

        # Check accuracy every epoch
        current_acc = calculate_accuracy(model, segment_files, transcriptions, vocab, target_seconds, target_words, device)
        print(f"Accuracy: {current_acc:.1f}%", flush=True)

        if current_acc >= 90.0:
            print(f"✓ Early stopping: accuracy reached 90%")
            break

    return model

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 train_all_curriculum.py <dataset_name>")
        print("Example:")
        print("  python3 train_all_curriculum.py Quran-A")
        sys.exit(1)

    dataset_name = sys.argv[1]

    print(f"\n{'='*60}")
    print(f"CURRICULUM TRAINING ON ALL SEGMENTS - DATASET: {dataset_name}")
    print(f"{'='*60}\n")

    # Paths
    vocab_path = "../models/vocabulary.json"
    model_path = "../models/muhaffez_whisper.pt"
    datasets_dir = f"../datasets/{dataset_name}"

    # Load vocabulary
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"Vocabulary size: {len(vocab)}")

    # Find ALL text files
    text_files = sorted(glob.glob(f"{datasets_dir}/text/*.txt"))
    if not text_files:
        print(f"❌ No text files found in {datasets_dir}/text/")
        sys.exit(1)

    # Collect all segments
    all_segment_files = []
    all_transcriptions = []

    for text_file in text_files:
        surah_part = os.path.splitext(os.path.basename(text_file))[0]
        surah_num = surah_part.split('-')[0]
        audio_dir = f"{datasets_dir}/audio/{surah_num}"

        with open(text_file, "r", encoding="utf-8") as f:
            transcriptions = [line.strip() for line in f if line.strip()]

        segment_files = sorted(glob.glob(f"{audio_dir}/{surah_part}-*.wav"))

        if len(transcriptions) != len(segment_files):
            print(f"⚠️  Warning: Mismatch in {surah_part}")
            continue

        all_segment_files.extend(segment_files)
        all_transcriptions.extend(transcriptions)
        print(f"  Loaded {len(segment_files)} segments from {surah_part}")

    total_segments = len(all_segment_files)
    print(f"\n✓ Total segments: {total_segments}")

    # Find max words needed for curriculum
    max_words = max(len(text.split()) for text in all_transcriptions)
    print(f"✓ Maximum words in any segment: {max_words}")

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
        print(f"✓ Model loaded successfully!")
    else:
        print(f"\n⚠️  No existing model found. Starting from scratch.")

    model = model.to(device)

    print(f"\n✓ Starting curriculum training with {max_words} stages")
    print(f"✓ Replay buffer size: 10% of segments\n")

    # Curriculum training with 10% replay buffer
    for stage_num in range(1, max_words + 1):
        target_seconds = stage_num * CHUNK_DURATION
        target_words = stage_num * WORDS_PER_CHUNK

        # Create replay buffer (10% of current stage segments)
        replay_size = max(int(total_segments * 0.1), 1)
        replay_indices = random.sample(range(total_segments), min(replay_size, total_segments))
        replay_files = [all_segment_files[i] for i in replay_indices]
        replay_texts = [all_transcriptions[i] for i in replay_indices]

        # Train this stage
        model = train_curriculum_stage(
            model, all_segment_files, all_transcriptions, vocab,
            stage_num, target_seconds, target_words,
            replay_files, replay_texts, device
        )

        # Save after each stage
        torch.save(model.state_dict(), model_path)

    # Final save and accuracy
    torch.save(model.state_dict(), model_path)
    print(f"\nFinal model saved to: {model_path}")

    final_acc = calculate_accuracy(model, all_segment_files, all_transcriptions, vocab, None, None, device)
    print(f"FINAL_ACCURACY: {final_acc:.0f}%")


if __name__ == "__main__":
    main()
