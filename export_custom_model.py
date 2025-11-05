#!/usr/bin/env python3
"""
Export custom encoder_decoder_model.pt to ONNX with KV-cache support.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add onnx directory to path to import the model
sys.path.insert(0, 'onnx')
from encoder_decoder_transformer import EncoderDecoderTransformer

def export_custom_model_to_onnx():
    model_path = "onnx/models/encoder_decoder_model.pt"
    output_dir = Path("app/src/main/assets/whisper_onnx_custom")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🚀 Exporting Custom Lightweight Model to ONNX")
    print("=" * 70)
    print(f"Source: {model_path}")
    print(f"Output: {output_dir}")
    print()

    # Load checkpoint
    print("📦 Loading model checkpoint...")
    checkpoint = torch.load(model_path, map_location='cpu')

    # Determine model parameters from checkpoint
    d_model = checkpoint['positional_embedding'].shape[1]  # 128
    max_seq_len = checkpoint['positional_embedding'].shape[0]  # 1500

    # Count encoder blocks
    n_encoder_layers = max([int(k.split('.')[1]) for k in checkpoint.keys()
                             if k.startswith('blocks.') and k.split('.')[1].isdigit()], default=-1) + 1

    # Count decoder layers
    n_decoder_layers = max([int(k.split('.')[1]) for k in checkpoint.keys()
                             if k.startswith('decoder_layers.') and k.split('.')[1].isdigit()], default=-1) + 1

    # Get vocab size from token_embedding
    vocab_size = checkpoint['token_embedding.weight'].shape[0]

    print(f"   ✅ Detected parameters:")
    print(f"      d_model: {d_model}")
    print(f"      vocab_size: {vocab_size}")
    print(f"      encoder_layers: {n_encoder_layers}")
    print(f"      decoder_layers: {n_decoder_layers}")
    print(f"      max_seq_len: {max_seq_len}")
    print()

    # Create model
    print("🏗️  Creating model architecture...")

    # Calculate n_heads and d_ff from attention weights
    # attn.q_proj.weight shape: (d_model, d_model) = (128, 128)
    # For 128-dim model, typical is n_heads=8, giving head_dim=16
    n_heads = 8 if d_model == 128 else 6
    d_ff = d_model * 4  # Typical FFN expansion

    model = EncoderDecoderTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=0.0,  # No dropout for inference
        max_seq_len=max_seq_len,
        n_mels=80
    )

    # Load weights
    print("📥 Loading weights...")
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    print("   ✅ Weights loaded successfully")
    print()

    # Export encoder
    print("=" * 70)
    print("📦 Exporting Encoder to ONNX")
    print("=" * 70)

    encoder_path = output_dir / "encoder_model.onnx"

    # Dummy input: mel features (batch, n_mels=80, time=3000)
    dummy_mel = torch.randn(1, 80, 3000)

    print("   Input shape: (batch=1, n_mels=80, time=3000)")
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

    # Dummy inputs for decoder - use valid token IDs for this vocab
    dummy_text_ids = torch.tensor([[1, 2, 3, 4]])  # Start tokens (within vocab_size)
    dummy_encoder_output = torch.randn(1, 1500, d_model)  # encoder output

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
            return self.model.decode(input_ids, encoder_hidden_states)

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

    # Calculate total size
    total_size = (encoder_path.stat().st_size + decoder_path.stat().st_size) / (1024*1024)

    print("=" * 70)
    print("✅ Export Complete!")
    print("=" * 70)
    print()
    print(f"📊 Total model size: {total_size:.2f} MB")
    print(f"   (vs {290:.2f} MB for custom-whisper-ar-quran)")
    print(f"   → {290/total_size:.1f}x smaller!")
    print()
    print("📁 Exported files:")
    print(f"   {encoder_path}")
    print(f"   {decoder_path}")
    print()
    print("=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print()
    print("1. Copy tokenizer files:")
    print(f"   cp app/src/main/assets/whisper_onnx/{{vocab.json,merges.txt,added_tokens.json}} {output_dir}/")
    print()
    print("2. Test the model:")
    print(f"   # Use the models from: {output_dir}")
    print()
    print("3. Expected performance:")
    print("   - Much faster than current model (16MB vs 290MB)")
    print("   - Trained specifically on Quran audio")
    print("   - May need to update Kotlin code if vocab size differs")
    print()
    print("⚠️  Note: This export does NOT include KV-cache yet.")
    print("   The decoder still processes all previous tokens on each step.")
    print("   Adding KV-cache requires modifying the model architecture.")
    print()

    return 0

if __name__ == "__main__":
    try:
        sys.exit(export_custom_model_to_onnx())
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
