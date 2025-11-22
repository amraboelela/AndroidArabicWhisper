#!/usr/bin/env python3
"""
Export muhaffez_whisper.pt to ONNX format for Android app
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
import json

sys.path.append("..")
from encoder_decoder_transformer import EncoderDecoderTransformer

def export_muhaffez_to_onnx():
    model_path = "../models/muhaffez_whisper.pt"
    vocab_path = "../models/vocabulary.json"
    output_dir = Path("../../app/src/main/assets/muhaffez_whisper")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🚀 Exporting Muhaffez Whisper Model to ONNX")
    print("=" * 70)
    print(f"Source: {model_path}")
    print(f"Vocab: {vocab_path}")
    print(f"Output: {output_dir}")
    print()

    # Load vocabulary
    print("📖 Loading vocabulary...")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    print(f"   ✅ Vocabulary loaded: {vocab_size} words")
    print()

    # Create model with same parameters as training
    print("🏗️  Creating model architecture...")
    model = EncoderDecoderTransformer(
        vocab_size=vocab_size,
        d_model=128,
        n_encoder_layers=4,
        n_decoder_layers=4,
        n_heads=4,
        d_ff=512,
        dropout=0.0,  # No dropout for inference
        max_seq_len=1500,
        n_mels=40
    )

    # Load weights
    print("📥 Loading weights...")
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    print("   ✅ Weights loaded successfully")
    print()

    # Export encoder
    print("=" * 70)
    print("📦 Exporting Encoder to ONNX")
    print("=" * 70)

    encoder_path = output_dir / "encoder_model.onnx"

    # Dummy input: mel features (batch, n_mels=40, time=3000)
    dummy_mel = torch.randn(1, 40, 3000)

    print("   Input shape: (batch=1, n_mels=40, time=3000)")
    print("   Exporting...")

    # Create encoder wrapper
    class EncoderWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_features):
            return self.model.encode(input_features)

    encoder_wrapper = EncoderWrapper(model)

    torch.onnx.export(
        encoder_wrapper,
        dummy_mel,
        encoder_path,
        input_names=["input_features"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_features": {0: "batch", 2: "time"},
            "last_hidden_state": {0: "batch", 1: "sequence"}
        },
        opset_version=14,
    )

    print(f"   ✅ Encoder exported: {encoder_path.name}")
    print(f"   Size: {encoder_path.stat().st_size / (1024*1024):.2f} MB")
    print()

    # Export decoder WITHOUT KV-cache (simpler, works immediately)
    print("=" * 70)
    print("📦 Exporting Decoder to ONNX (Without KV-cache)")
    print("=" * 70)

    decoder_path = output_dir / "decoder_model.onnx"

    # Dummy inputs for decoder
    dummy_text_ids = torch.tensor([[1, 2, 3, 4]])  # Start tokens
    dummy_encoder_output = torch.randn(1, 1500, 128)  # encoder output

    print(f"   Input shapes:")
    print(f"      input_ids: {dummy_text_ids.shape}")
    print(f"      encoder_hidden_states: {dummy_encoder_output.shape}")
    print("   Exporting...")

    # Create a wrapper for decoder
    class DecoderWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, encoder_hidden_states):
            logits, _ = self.model.decode(input_ids, encoder_hidden_states, use_cache=False)
            return logits

    decoder_wrapper = DecoderWrapper(model)

    torch.onnx.export(
        decoder_wrapper,
        (dummy_text_ids, dummy_encoder_output),
        decoder_path,
        input_names=["input_ids", "encoder_hidden_states"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "encoder_hidden_states": {0: "batch", 1: "encoder_sequence"},
            "logits": {0: "batch", 1: "sequence"}
        },
        opset_version=14,
    )

    print(f"   ✅ Decoder exported: {decoder_path.name}")
    print(f"   Size: {decoder_path.stat().st_size / (1024*1024):.2f} MB")
    print()

    # Copy vocabulary file
    print("📋 Copying vocabulary file...")
    import shutil
    vocab_dest = output_dir / "vocabulary.json"
    shutil.copy2(vocab_path, vocab_dest)
    print(f"   ✅ Vocabulary copied: {vocab_dest.name}")
    print()

    # Calculate total size
    total_size = (encoder_path.stat().st_size + decoder_path.stat().st_size) / (1024*1024)

    print("=" * 70)
    print("✅ Export Complete!")
    print("=" * 70)
    print()
    print(f"📊 Total model size: {total_size:.2f} MB")
    print()
    print("📁 Exported files:")
    print(f"   {encoder_path}")
    print(f"   {decoder_path}")
    print(f"   {vocab_dest}")
    print()
    print("=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print()
    print("1. The models are ready in: " + str(output_dir))
    print()
    print("2. Update Android app to:")
    print("   - Load ONNX models from muhaffez_whisper directory")
    print("   - Use global normalization: mel_mean=-4.2677, mel_std=4.5689")
    print("   - Segment audio into chunks before transcription")
    print("   - Concatenate segment transcriptions")
    print()

    return 0

if __name__ == "__main__":
    try:
        sys.exit(export_muhaffez_to_onnx())
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
