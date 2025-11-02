#!/usr/bin/env python3
"""
Train encoder-decoder model on Al-Baqara full segments → full transcriptions
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
sys.path.append("../..")
from encoder_decoder_transformer import EncoderDecoderTransformer

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
# Audio feature extraction
# ==============================================================
def extract_mel_features(audio_path, n_mels=80):
    """Extract Whisper-compatible mel spectrogram features"""
    waveform, sample_rate = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

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
    mel_features = (mel_features - mel_features.mean()) / (mel_features.std() + 1e-5)
    return mel_features, sample_rate

# ==============================================================
# Tokenization
# ==============================================================
def tokenize_text(text, vocab):
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    words = text.split()
    return [word_to_idx.get(word, 0) for word in words]  # 0 = unknown

# ==============================================================
# Training
# ==============================================================
def train_full_segments(model, segment_files, transcriptions, vocab, num_epochs=5, learning_rate=1e-5):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

    best_loss = float('inf')
    prev_loss = float('inf')
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_iterations = 0
        indices = list(range(len(segment_files)))
        random.shuffle(indices)

        for i in indices:
            seg_file = segment_files[i]
            text = transcriptions[i]

            # Train on FULL SEGMENT -> full transcription
            audio_features, sample_rate = extract_mel_features(seg_file)
            text_tokens = tokenize_text(text, vocab)
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

        avg_loss = total_loss / total_iterations

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch
            }, "checkpoint_best_full.pt")
            best_marker = " ⭐ NEW BEST!"
        else:
            best_marker = ""

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} | Loss={avg_loss:.4f} | LR={learning_rate:.6f} | Time={elapsed:.1f}s{best_marker}")

        # Early stopping check
        loss_change = prev_loss - avg_loss
        if loss_change < 0.001 and epoch > 0:
            print(f"⚠️ Early stopping: loss change ({loss_change:.6f}) < 0.001")
            break
        prev_loss = avg_loss

        # Sample generation
        model.eval()
        test_audio, sample_rate = extract_mel_features(segment_files[0])
        waveform, sr = torchaudio.load(segment_files[0])
        audio_duration = waveform.shape[1] / sr
        test_audio = test_audio.transpose(0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            generated = model.generate(test_audio, max_new_tokens=50, audio_duration_seconds=audio_duration)
            generated_ids = generated[0].tolist()
            if generated_ids and generated_ids[0] == 1:
                generated_ids = generated_ids[1:]
            if 2 in generated_ids:
                generated_ids = generated_ids[:generated_ids.index(2)]
            generated_words = [vocab[idx] for idx in generated_ids if idx < len(vocab)]
            print(f"  🔹 Generated: {' '.join(generated_words)}")
            print(f"  🔸 Expected: {transcriptions[0]}")
        model.train()

    total_time = time.time() - start_time
    print(f"Training complete in {total_time:.1f}s | Best loss: {best_loss:.4f}")
    return model

# ==============================================================
# Main
# ==============================================================
def main():
    datasets_dir = "audio"
    vocab_path = "../../vocabulary.json"
    model_path = "../../encoder_decoder_model.pt"

    # Load vocab
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"Vocabulary size: {len(vocab)}")

    # Load Al-Baqara segments
    all_transcriptions = []
    all_segment_files = []

    # Load Al-Baqara part 1 (002-01)
    baqara_01_text_path = "002-01.txt"
    with open(baqara_01_text_path, "r", encoding="utf-8") as f:
        baqara_01_transcriptions = [line.strip() for line in f if line.strip()]
    baqara_01_segments = sorted(glob.glob(os.path.join(datasets_dir, "002-01-*.wav")))
    print(f"Loaded {len(baqara_01_transcriptions)} Al-Baqara part 1 transcriptions, {len(baqara_01_segments)} segments")

    # Load Al-Baqara part 2 (002-02)
    baqara_02_text_path = "002-02.txt"
    with open(baqara_02_text_path, "r", encoding="utf-8") as f:
        baqara_02_transcriptions = [line.strip() for line in f if line.strip()]
    baqara_02_segments = sorted(glob.glob(os.path.join(datasets_dir, "002-02-*.wav")))
    print(f"Loaded {len(baqara_02_transcriptions)} Al-Baqara part 2 transcriptions, {len(baqara_02_segments)} segments")

    # Load Al-Baqara part 3 (002-03)
    baqara_03_text_path = "002-03.txt"
    with open(baqara_03_text_path, "r", encoding="utf-8") as f:
        baqara_03_transcriptions = [line.strip() for line in f if line.strip()]
    baqara_03_segments = sorted(glob.glob(os.path.join(datasets_dir, "002-03-*.wav")))
    print(f"Loaded {len(baqara_03_transcriptions)} Al-Baqara part 3 transcriptions, {len(baqara_03_segments)} segments")

    # Combine all Baqara datasets
    all_transcriptions = baqara_01_transcriptions + baqara_02_transcriptions + baqara_03_transcriptions
    all_segment_files = baqara_01_segments + baqara_02_segments + baqara_03_segments
    print(f"\n✓ Total Al-Baqara: {len(all_transcriptions)} transcriptions, {len(all_segment_files)} segments")
    print("Training on: Al-Baqara full segments → full transcriptions")

    # Create smaller 128-dimension encoder-decoder
    model = EncoderDecoderTransformer(
        vocab_size=len(vocab),
        d_model=128,           # Smaller dimension
        n_encoder_layers=4,
        n_decoder_layers=4,
        n_heads=4,             # 128/4 = 32 dim per head
        d_ff=512,              # 4x d_model
        dropout=0.1
    )

    # Load existing model and continue training
    import shutil
    if os.path.exists(model_path):
        backup_path = model_path.replace(".pt", "_backup_full.pt")
        shutil.copy2(model_path, backup_path)
        print(f"✓ Backup created: {backup_path}")

        print(f"Loading existing model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✓ Model loaded successfully! Training on full segments.")
    else:
        print("No existing model found. Starting with fresh weights for full segments training.")

    # Train
    print(f"\nStarting training for up to 5 epochs on {len(all_segment_files)} segments (full → full)...\n")
    model = train_full_segments(
        model,
        all_segment_files,
        all_transcriptions,
        vocab,
        num_epochs=5,
        learning_rate=1e-5
    )

    # Save final model
    torch.save(model.state_dict(), model_path)
    print(f"Final model saved to: {model_path}")


if __name__ == "__main__":
    main()
