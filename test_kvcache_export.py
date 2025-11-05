#!/usr/bin/env python3
"""
Manual ONNX export with KV-cache support using PyTorch directly.
"""

import sys
import torch
from pathlib import Path
from transformers import WhisperForConditionalGeneration, WhisperProcessor

def export_decoder_with_kvcache():
    """
    Export Whisper decoder with KV-cache using PyTorch ONNX export.
    """

    model_path = "onnx/models/custom-whisper-ar-quran"
    output_dir = Path("onnx/models/custom-whisper-ar-quran-onnx-kvcache")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🚀 Exporting Whisper Decoder with KV-Cache (Manual Method)")
    print("=" * 70)
    print(f"Source model: {model_path}")
    print(f"Output directory: {output_dir}")
    print()

    # Load model
    print("📦 Loading model...")
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.eval()

    # Enable KV-cache in config
    model.config.use_cache = True

    print(f"   ✅ Model loaded")
    print(f"   Model has {model.config.decoder_layers} decoder layers")
    print()

    # Export encoder (simple, no KV-cache needed)
    print("📦 Exporting encoder...")
    encoder_output_path = output_dir / "encoder_model.onnx"

    # Dummy input for encoder
    mel_features = torch.randn(1, 80, 3000)  # batch, n_mels, time

    torch.onnx.export(
        model.model.encoder,
        mel_features,
        encoder_output_path,
        input_names=["input_features"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_features": {0: "batch", 2: "time"},
            "last_hidden_state": {0: "batch", 1: "time"}
        },
        opset_version=14,
    )

    print(f"   ✅ Encoder exported to {encoder_output_path.name}")
    print(f"   Size: {encoder_output_path.stat().st_size / (1024*1024):.2f} MB")
    print()

    # For decoder with KV-cache, we need a more complex setup
    print("📦 Exporting decoder with KV-cache...")
    print("   ⚠️  Note: Full KV-cache decoder export is complex")
    print("   Let me check if we can use a simpler approach...")
    print()

    # Check if model supports past_key_values
    print("   Testing decoder with past_key_values...")
    decoder = model.model.decoder

    # Create dummy inputs
    batch_size = 1
    seq_len = 4
    encoder_seq_len = 1500
    hidden_size = model.config.d_model

    input_ids = torch.tensor([[50258, 50272, 50359, 50363]])  # Start tokens
    encoder_hidden_states = torch.randn(batch_size, encoder_seq_len, hidden_size)

    # Run forward pass to see structure
    with torch.no_grad():
        outputs = decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            use_cache=True,
        )

    print(f"   ✅ Decoder supports KV-cache!")
    print(f"   - Output logits shape: {outputs.last_hidden_state.shape}")
    print(f"   - Past key values: {len(outputs.past_key_values)} layers")
    print(f"   - Each layer has: {len(outputs.past_key_values[0])} key/value tensors")
    print()

    print("=" * 70)
    print("⚠️  Complex KV-Cache Export Limitation")
    print("=" * 70)
    print()
    print("Exporting a decoder with full KV-cache support requires:")
    print("1. Multiple ONNX files (decoder_model.onnx, decoder_with_past.onnx)")
    print("2. Complex input/output management for past_key_values")
    print("3. The optimum[onnxruntime] package (which failed to install)")
    print()
    print("=" * 70)
    print("Alternative Approach - Simpler Solution:")
    print("=" * 70)
    print()
    print("Instead of full KV-cache, we can:")
    print("1. Use the current decoder as-is")
    print("2. Optimize at the Kotlin level by reducing redundant computation")
    print("3. Or try using a pre-exported model from Hugging Face")
    print()
    print("Would you like me to:")
    print("A) Try downloading a pre-exported model with KV-cache from HF")
    print("B) Optimize the Kotlin code without changing the ONNX model")
    print("C) Try a minimal single-file decoder export (partial KV-cache)")
    print()

    return 0

if __name__ == "__main__":
    try:
        sys.exit(export_decoder_with_kvcache())
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
