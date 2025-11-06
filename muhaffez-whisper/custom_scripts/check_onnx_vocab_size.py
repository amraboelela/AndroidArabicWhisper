#!/usr/bin/env python3
"""
Check the actual vocabulary size embedded in the ONNX decoder model.
"""

import onnx
from pathlib import Path

def check_onnx_vocab_size():
    print("=" * 70)
    print("🔍 Checking ONNX Model Vocabulary Size")
    print("=" * 70)
    print()

    decoder_path = Path("app/src/main/assets/whisper_onnx_custom/decoder_model.onnx")

    if not decoder_path.exists():
        print(f"❌ Decoder not found: {decoder_path}")
        return 1

    print(f"📦 Loading decoder model: {decoder_path.name}")
    model = onnx.load(str(decoder_path))

    # Find the output shape (logits dimension should be vocab_size)
    print()
    print("📊 Model Outputs:")
    for output in model.graph.output:
        print(f"   {output.name}:")
        if output.type.tensor_type.shape.dim:
            dims = []
            for dim in output.type.tensor_type.shape.dim:
                if dim.dim_value:
                    dims.append(str(dim.dim_value))
                elif dim.dim_param:
                    dims.append(dim.dim_param)
                else:
                    dims.append("?")
            print(f"      Shape: ({', '.join(dims)})")
            # Last dimension should be vocab_size
            if dims:
                last_dim = dims[-1]
                print(f"      → Vocab size: {last_dim}")

    # Find initializers with embeddings
    print()
    print("📊 Looking for Embedding/Output layers:")
    for init in model.graph.initializer:
        if 'embedding' in init.name.lower() or 'output' in init.name.lower() or 'lm_head' in init.name.lower():
            shape = init.dims
            print(f"   {init.name}: {shape}")

    print()
    print("=" * 70)
    print("✅ Check Complete")
    print("=" * 70)
    print()
    print("⚠️  Important:")
    print("   The decoder output dimension should match your vocabulary size (14754).")
    print("   If it shows 50257 or 50000, the model was exported with wrong vocab!")
    print()

    return 0

if __name__ == "__main__":
    import sys
    try:
        sys.exit(check_onnx_vocab_size())
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
