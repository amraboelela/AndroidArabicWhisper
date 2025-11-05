#!/usr/bin/env python3
"""
Train encoder-decoder model on Al-Fatiha first 3 seconds → first 2 words
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
def extract_first_3_seconds_mel(audio_path, n_mels=80, target_seconds=3.0):
    """Extract mel features from only the first 3 seconds of the audio"""
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert stereo to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Trim to first N seconds
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
    # Global Whisper normalization (more robust than per-sample)

    WHISPER_MEL_MEAN = -4.2677393

    WHISPER_MEL_STD = 4.5689974

    mel_features = (mel_features - WHISPER_MEL_MEAN) / WHISPER_MEL_STD
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
def train_first_3_seconds(model, segment_files, transcriptions, vocab, dataset_name, num_epochs=5, learning_rate=1e-5):
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

            # Train on FIRST 3 SECONDS -> first 2 words
            audio_features, sample_rate = extract_first_3_seconds_mel(seg_file)
            words = text.split()
            first_two_words = " ".join(words[:2]) if len(words) >= 2 else text
            if not first_two_words:
                continue
            text_tokens = tokenize_text(first_two_words, vocab)
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
            }, "../../models/checkpoint_best_first_3_seconds.pt")
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
        test_audio, sample_rate = extract_first_3_seconds_mel(segment_files[0])
        test_audio = test_audio.transpose(0, 1).unsqueeze(0).to(device)
        words = transcriptions[0].split()
        first_two_words = " ".join(words[:2]) if len(words) >= 2 else transcriptions[0]
        with torch.no_grad():
            generated = model.generate(test_audio, max_new_tokens=30, audio_duration_seconds=3.0)
            generated_ids = generated[0].tolist()
            if generated_ids and generated_ids[0] == 1:
                generated_ids = generated_ids[1:]
            if 2 in generated_ids:
                generated_ids = generated_ids[:generated_ids.index(2)]
            generated_words = [vocab[idx] for idx in generated_ids if idx < len(vocab)]
            # Only show first 2 words since we're testing first 3 seconds
            display_words = generated_words[:2] if len(generated_words) >= 2 else generated_words
            print(f"  🔹 Generated: {' '.join(display_words)}")
            print(f"  🔸 Expected: {first_two_words}")
        model.train()

    total_time = time.time() - start_time
    print(f"Training complete in {total_time:.1f}s | Best loss: {best_loss:.4f}")
    return model

# ==============================================================
# Main
# ==============================================================
def main():
    import sys
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "base"
    datasets_dir = f"../{dataset_name}/audio"
    vocab_path = "../../vocabulary.json"
    model_path = "../../models/encoder_decoder_model.pt"

    # Load vocab
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"Vocabulary size: {len(vocab)}")

    # Load Al-Fatiha segments
    all_transcriptions = []
    all_segment_files = []

    # Load Al-Fatiha (001)
    fatiha_text_path = f"../{dataset_name}/text/001.txt"
    with open(fatiha_text_path, "r", encoding="utf-8") as f:
        fatiha_transcriptions = [line.strip() for line in f if line.strip()]
    fatiha_segments = sorted(glob.glob(os.path.join(datasets_dir, "001-*.wav")))
    print(f"Loaded {len(fatiha_transcriptions)} Al-Fatiha transcriptions, {len(fatiha_segments)} segments")

    # Use Al-Fatiha data
    all_transcriptions = fatiha_transcriptions
    all_segment_files = fatiha_segments
    print(f"\n✓ Total Al-Fatiha: {len(all_transcriptions)} transcriptions, {len(all_segment_files)} segments")
    print("Training on: Al-Fatiha first 3 seconds → first 2 words")

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
        backup_path = model_path.replace(".pt", "_backup_first_3_seconds.pt")
        shutil.copy2(model_path, backup_path)
        print(f"✓ Backup created: {backup_path}")

        print(f"Loading existing model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✓ Model loaded successfully! Training on first 3 seconds.")
    else:
        print("No existing model found. Starting with fresh weights for first 3 seconds training.")

    # Train
    print(f"\nStarting training for up to 5 epochs on {len(all_segment_files)} segments (first 3 seconds → first 2 words)...\n")
    model = train_first_3_seconds(
        model,
        all_segment_files,
        all_transcriptions,
        vocab,
        dataset_name,
        num_epochs=5,
        learning_rate=1e-5
    )

    # Save final model
    torch.save(model.state_dict(), model_path)
    print(f"Final model saved to: {model_path}")


if __name__ == "__main__":
    main()
