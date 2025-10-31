#!/usr/bin/env python3
import json
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from improved_transformer import ImprovedDecoderTransformer

# Check for Metal GPU support
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


def extract_mel_features(audio_path, n_mels=800, target_fps=10):
    """Extract mel spectrogram features"""
    waveform, sample_rate = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    hop_length = sample_rate // target_fps
    n_fft = 2048

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

    return mel_features, sample_rate


def split_into_chunks(audio_features, text_tokens, chunk_duration=5, fps=10):
    """Split audio and text into 5-second chunks"""
    frames_per_chunk = chunk_duration * fps
    total_frames = audio_features.shape[0]
    num_chunks = (total_frames + frames_per_chunk - 1) // frames_per_chunk

    tokens_per_chunk = len(text_tokens) / num_chunks

    chunks = []

    for i in range(num_chunks):
        start_frame = i * frames_per_chunk
        end_frame = min((i + 1) * frames_per_chunk, total_frames)
        audio_chunk = audio_features[start_frame:end_frame]

        start_token = int(i * tokens_per_chunk)
        end_token = int((i + 1) * tokens_per_chunk) if i < num_chunks - 1 else len(text_tokens)
        text_chunk = text_tokens[start_token:end_token]

        if len(text_chunk) > 0:
            chunks.append((audio_chunk, text_chunk))
            print(f"  Chunk {i+1}: Audio frames={audio_chunk.shape[0]}, Text tokens={len(text_chunk)}")

    return chunks


def tokenize_text(text, vocab):
    """Tokenize text"""
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    words = text.split()
    return [word_to_idx.get(word, 0) for word in words]


def train_on_chunks_gpu(model, chunks, vocab, num_epochs=300, lr=1e-4, use_fp16=True):
    """Train model on GPU with optional FP16"""

    # Move model to device
    model = model.to(device)

    # Use FP16 if requested and supported
    use_amp = use_fp16 and device.type in ['mps', 'cuda']
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None

    if use_fp16:
        print(f"✓ Using FP16 mixed precision training")
        if device.type == 'mps':
            # For Metal, convert model to float16
            model = model.half()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Precision: {'FP16' if use_fp16 else 'FP32'}")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {num_epochs}")
    print(f"Optimizer: AdamW")

    model.train()

    print(f"\n{'='*60}")
    print(f"Training Progress:")
    print(f"{'='*60}")

    import time
    start_time = time.time()

    for epoch in range(num_epochs):
        total_loss = 0

        for chunk_idx, (audio_chunk, text_chunk) in enumerate(chunks):
            optimizer.zero_grad()

            # Prepare data and move to device
            audio_batch = audio_chunk.unsqueeze(0).to(device)
            if use_fp16 and device.type == 'mps':
                audio_batch = audio_batch.half()

            input_tokens = [1] + text_chunk
            target_tokens = text_chunk + [2]

            input_ids = torch.tensor([input_tokens], dtype=torch.long, device=device)
            labels = torch.tensor([target_tokens], dtype=torch.long, device=device)

            # Forward pass with optional AMP
            if use_amp and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    logits, loss = model(
                        audio_features=audio_batch,
                        text_ids=input_ids,
                        labels=labels
                    )
            else:
                logits, loss = model(
                    audio_features=audio_batch,
                    text_ids=input_ids,
                    labels=labels
                )

            # Backward pass
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(chunks)

        # Print progress
        if (epoch + 1) % 20 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:3d}/{num_epochs}: Avg Loss = {avg_loss:.4f} | Time: {elapsed:.1f}s")

        # Generate sample every 50 epochs
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                audio_test = chunks[0][0].unsqueeze(0).to(device)
                if use_fp16 and device.type == 'mps':
                    audio_test = audio_test.half()
                generated = model.generate(audio_test, max_new_tokens=10, temperature=0.5)
                generated_words = [vocab[idx] for idx in generated[0].cpu().tolist()]
                print(f"  Sample: {' '.join(generated_words[:8])}...")
            model.train()

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Total time: {total_time:.1f}s ({total_time/num_epochs:.2f}s per epoch)")
    print(f"{'='*60}")

    return model


def main():
    """Main training function with GPU acceleration"""

    # Paths
    audio_path = "/Users/amraboelela/develop/android/AndroidArabicWhisper/app/src/main/assets/001.wav"
    vocab_path = "vocabulary.json"

    # Load vocabulary
    print("Loading vocabulary...")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"Vocabulary size: {len(vocab)}")

    # Create improved model
    print("\nCreating improved model...")
    print("  - Dimension: 800")
    print("  - 5 transformer layers")
    print("  - 10 attention heads")
    print("  - FP16 precision for efficiency")

    model = ImprovedDecoderTransformer(
        vocab_size=len(vocab),
        d_model=800,
        n_layers=5,
        n_heads=10,
        d_ff=3200,
        dropout=0.1
    )

    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 2 / (1024**2)  # FP16 = 2 bytes per param
    print(f"Model parameters: {total_params:,}")
    print(f"Model size (FP16): ~{model_size_mb:.1f} MB")

    # Prepare data
    print(f"\n{'='*60}")
    print(f"Preparing Al-Fatiha Data (5-second chunks):")
    print(f"{'='*60}")

    alfatiha_text = "اعوذ بالله من الشيطان الرجيم بسم الله الرحمن الرحيم الحمد لله رب العالمين الرحمن الرحيم مالك يوم الدين اياك نعبد واياك نستعين اهدنا الصراط المستقيم"

    print(f"Al-Fatiha text:")
    print(f"  {alfatiha_text}")

    print(f"\nExtracting audio features...")
    audio_features, sample_rate = extract_mel_features(audio_path)
    print(f"Total audio: {audio_features.shape[0]} frames ({audio_features.shape[0]/10:.1f} seconds)")

    text_tokens = tokenize_text(alfatiha_text, vocab)
    print(f"Total text: {len(text_tokens)} tokens")

    print(f"\nSplitting into 5-second chunks:")
    chunks = split_into_chunks(audio_features, text_tokens, chunk_duration=5, fps=10)
    print(f"\nTotal chunks created: {len(chunks)}")

    # Train model with GPU acceleration (FP32 for stability)
    model = train_on_chunks_gpu(
        model,
        chunks,
        vocab,
        num_epochs=300,
        lr=5e-5,
        use_fp16=False  # Use FP32 for numerical stability
    )

    # Test on full audio
    print(f"\n{'='*60}")
    print(f"Testing on First Chunk:")
    print(f"{'='*60}")

    model.eval()
    with torch.no_grad():
        test_audio = chunks[0][0].unsqueeze(0).to(device)
        if device.type == 'mps':
            test_audio = test_audio.half()
        generated = model.generate(test_audio, max_new_tokens=15, temperature=0.1)
        generated_words = [vocab[idx] for idx in generated[0].cpu().tolist()]

        expected_words = [vocab[idx] for idx in chunks[0][1]]

        print(f"\nFirst chunk (5 seconds):")
        print(f"Expected: {' '.join(expected_words)}")
        print(f"Generated: {' '.join(generated_words)}")

    # Save model (convert back to FP32 for compatibility)
    save_path = "alfatiha_model_gpu.pt"
    model_cpu = model.cpu().float()  # Convert to FP32 for saving
    torch.save(model_cpu.state_dict(), save_path)

    save_size_mb = total_params * 4 / (1024**2)  # FP32 = 4 bytes
    print(f"\n{'='*60}")
    print(f"Model saved to: {save_path}")
    print(f"Saved size (FP32): ~{save_size_mb:.1f} MB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
