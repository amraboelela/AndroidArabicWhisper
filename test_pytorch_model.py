#!/usr/bin/env python3
"""
Test the PyTorch encoder_decoder_model.pt directly on segmented audio files
"""

import sys
import torch
import torch.nn.functional as F
import torchaudio
import json
from pathlib import Path

# Add onnx directory to path to import the model
sys.path.insert(0, 'onnx')
from encoder_decoder_transformer import EncoderDecoderTransformer


def load_model():
    """Load the PyTorch model"""
    model_path = "onnx/models/encoder_decoder_model.pt"

    print("=" * 70)
    print("📦 Loading PyTorch Model")
    print("=" * 70)
    print(f"Model: {model_path}")

    checkpoint = torch.load(model_path, map_location='cpu')

    # Get model parameters
    d_model = checkpoint['positional_embedding'].shape[1]
    max_seq_len = checkpoint['positional_embedding'].shape[0]
    n_encoder_layers = max([int(k.split('.')[1]) for k in checkpoint.keys()
                             if k.startswith('blocks.') and k.split('.')[1].isdigit()], default=-1) + 1
    n_decoder_layers = max([int(k.split('.')[1]) for k in checkpoint.keys()
                             if k.startswith('decoder_layers.') and k.split('.')[1].isdigit()], default=-1) + 1
    vocab_size = checkpoint['token_embedding.weight'].shape[0]

    print(f"   d_model: {d_model}")
    print(f"   vocab_size: {vocab_size}")
    print(f"   encoder_layers: {n_encoder_layers}")
    print(f"   decoder_layers: {n_decoder_layers}")
    print(f"   max_seq_len: {max_seq_len}")
    print()

    # Create model
    n_heads = 8 if d_model == 128 else 6
    d_ff = d_model * 4

    model = EncoderDecoderTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=0.0,
        max_seq_len=max_seq_len,
        n_mels=80
    )

    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    print("✅ Model loaded successfully")
    print()

    return model, vocab_size


def load_vocabulary():
    """Load vocabulary"""
    vocab_path = "onnx/vocabulary.json"

    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_list = json.load(f)

    vocab_dict = {token: idx for idx, token in enumerate(vocab_list)}
    reverse_vocab = {idx: token for idx, token in enumerate(vocab_list)}

    print(f"📖 Loaded vocabulary: {len(vocab_list)} tokens")
    print(f"   Special tokens: {vocab_list[:3]}")
    print()

    return vocab_dict, reverse_vocab


def extract_mel_features(audio_path):
    """Extract mel spectrogram features (simplified - just load and resample)"""
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)

    # Pad or truncate to 30 seconds (480000 samples at 16kHz)
    target_length = 480000
    if waveform.shape[1] < target_length:
        waveform = F.pad(waveform, (0, target_length - waveform.shape[1]))
    else:
        waveform = waveform[:, :target_length]

    # Compute mel spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        n_mels=80,
        f_min=0,
        f_max=8000
    )

    mel_spec = mel_transform(waveform)

    # Apply log
    mel_spec = torch.log10(torch.clamp(mel_spec, min=1e-10))

    # Normalize (Whisper normalization)
    mean = -4.2677393
    std = 4.5689974
    mel_spec = (mel_spec - mean) / std

    # Take first 3000 frames (30 seconds)
    mel_spec = mel_spec[:, :, :3000]

    return mel_spec.squeeze(0)  # Remove batch dimension -> (80, 3000)


def decode_tokens(token_ids, reverse_vocab):
    """Decode token IDs to text"""
    # Skip special tokens (0=<unk>, 1=<s>, 2=</s>)
    text_tokens = [tid for tid in token_ids if tid > 2]

    # Convert to strings
    words = [reverse_vocab.get(tid, f"<UNK_{tid}>") for tid in text_tokens]

    return ' '.join(words)


def transcribe_segment(model, mel_features, reverse_vocab, vocab_size, max_length=50):
    """Transcribe a single audio segment"""
    with torch.no_grad():
        # Encode
        encoder_output = model.encode(mel_features.unsqueeze(0))  # Add batch dim

        # Decode autoregressively
        sos_token = 1  # <s>
        eos_token = 2  # </s>

        generated = [sos_token]

        for step in range(max_length):
            # Prepare input
            input_ids = torch.tensor([generated], dtype=torch.long)

            # Decode
            logits = model.decode(input_ids, encoder_output)

            # Get next token (greedy)
            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()

            # Check for EOS
            if next_token == eos_token:
                print(f"      → EOS at step {step}")
                break

            generated.append(next_token)

            # Safety check
            if next_token >= vocab_size:
                print(f"      → Invalid token {next_token} >= {vocab_size}, stopping")
                break

        return generated


def main():
    print("=" * 70)
    print("🧪 Testing PyTorch Model on Segmented Audio")
    print("=" * 70)
    print()

    # Load model and vocabulary
    model, vocab_size = load_model()
    vocab_dict, reverse_vocab = load_vocabulary()

    # Test segments
    audio_dir = Path("onnx/datasets/base/audio")
    segment_files = sorted(audio_dir.glob("001-00*.wav"))

    if not segment_files:
        print(f"❌ No segment files found in {audio_dir}")
        return 1

    print(f"📁 Found {len(segment_files)} segment files")
    print()

    # Transcribe each segment
    all_transcriptions = []

    for i, audio_path in enumerate(segment_files, 1):
        print("=" * 70)
        print(f"🎤 Segment {i}/{len(segment_files)}: {audio_path.name}")
        print("=" * 70)

        # Extract features
        print("   Extracting mel features...")
        mel_features = extract_mel_features(str(audio_path))
        print(f"   ✅ Mel shape: {mel_features.shape}")

        # Transcribe
        print("   Transcribing...")
        generated_tokens = transcribe_segment(model, mel_features, reverse_vocab, vocab_size)

        # Decode
        transcription = decode_tokens(generated_tokens, reverse_vocab)

        print(f"   Generated tokens: {generated_tokens}")
        print(f"   📝 Transcription: '{transcription}'")
        print()

        all_transcriptions.append(transcription)

    # Final result
    print("=" * 70)
    print("✅ All Segments Transcribed")
    print("=" * 70)
    print()

    for i, trans in enumerate(all_transcriptions, 1):
        print(f"   Segment {i}: '{trans}'")

    print()
    final_text = ' '.join(all_transcriptions)
    print(f"🏁 Complete transcription:")
    print(f"   {final_text}")
    print()

    # Expected text (Al-Fatiha)
    expected = "بسم الله الرحمن الرحيم الحمد لله رب العالمين الرحمن الرحيم مالك يوم الدين إياك نعبد وإياك نستعين اهدنا الصراط المستقيم صراط الذين أنعمت عليهم غير المغضوب عليهم ولا الضالين"
    print(f"📖 Expected (Al-Fatiha):")
    print(f"   {expected}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
