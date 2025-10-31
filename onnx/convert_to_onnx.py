#!/usr/bin/env python3
"""
Convert trained PyTorch model to ONNX format with FP16 optimization
"""
import json
import torch
import onnx
from onnxconverter_common import float16
from improved_transformer import ImprovedDecoderTransformer

def convert_to_fp16(model_path, output_path):
    """
    Convert FP32 model to FP16

    Args:
        model_path: Path to FP32 model (.pt)
        output_path: Path to save FP16 model (.pt)
    """
    print(f"Loading FP32 model from: {model_path}")

    # Load vocabulary to get vocab size
    with open("vocabulary.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # Create model
    model = ImprovedDecoderTransformer(
        vocab_size=len(vocab),
        d_model=800,
        n_layers=5,
        n_heads=10,
        d_ff=3200,
        dropout=0.1
    )

    # Load weights
    model.load_state_dict(torch.load(model_path))

    # Convert to FP16
    model_fp16 = model.half()

    # Save FP16 model
    torch.save(model_fp16.state_dict(), output_path)

    # Calculate sizes
    fp32_size = sum(p.numel() * 4 for p in model.parameters()) / (1024**2)
    fp16_size = sum(p.numel() * 2 for p in model_fp16.parameters()) / (1024**2)

    print(f"\n✓ Model converted to FP16")
    print(f"  FP32 size: {fp32_size:.1f} MB")
    print(f"  FP16 size: {fp16_size:.1f} MB")
    print(f"  Saved to: {output_path}")

    return model_fp16


def export_to_onnx(model, output_path, vocab_size):
    """
    Export PyTorch model to ONNX format

    Args:
        model: PyTorch model
        output_path: Path to save ONNX model (.onnx)
        vocab_size: Vocabulary size
    """
    print(f"\nExporting model to ONNX...")

    model.eval()
    model = model.float()  # ONNX export needs FP32

    # Create dummy inputs for tracing
    batch_size = 1
    audio_len = 50  # 5 seconds at 10 fps
    text_len = 5

    dummy_audio = torch.randn(batch_size, audio_len, 800)
    dummy_text = torch.randint(0, vocab_size, (batch_size, text_len))

    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_audio, dummy_text, None),  # (audio_features, text_ids, labels)
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['audio_features', 'text_ids'],
        output_names=['logits'],
        dynamic_axes={
            'audio_features': {0: 'batch_size', 1: 'audio_len'},
            'text_ids': {0: 'batch_size', 1: 'text_len'},
            'logits': {0: 'batch_size', 1: 'text_len'}
        }
    )

    print(f"✓ Model exported to ONNX")
    print(f"  Saved to: {output_path}")

    # Verify ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"✓ ONNX model verified")


def optimize_onnx_fp16(onnx_path, output_path):
    """
    Optimize ONNX model and convert to FP16

    Args:
        onnx_path: Path to ONNX model
        output_path: Path to save optimized FP16 ONNX model
    """
    print(f"\nOptimizing ONNX model to FP16...")

    # Load ONNX model
    model = onnx.load(onnx_path)

    # Convert to FP16
    model_fp16 = float16.convert_float_to_float16(model)

    # Save optimized model
    onnx.save(model_fp16, output_path)

    # Calculate sizes
    import os
    original_size = os.path.getsize(onnx_path) / (1024**2)
    optimized_size = os.path.getsize(output_path) / (1024**2)

    print(f"✓ ONNX model optimized to FP16")
    print(f"  Original size: {original_size:.1f} MB")
    print(f"  Optimized size: {optimized_size:.1f} MB")
    print(f"  Reduction: {(1 - optimized_size/original_size)*100:.1f}%")
    print(f"  Saved to: {output_path}")


def main():
    """Main conversion pipeline"""

    print("="*60)
    print("Model Conversion Pipeline")
    print("="*60)

    # Paths
    input_model = "alfatiha_model_variable.pt"  # or alfatiha_model_gpu.pt
    fp16_model = "alfatiha_model_fp16.pt"
    onnx_model = "alfatiha_model.onnx"
    onnx_fp16_model = "alfatiha_model_fp16.onnx"

    # Load vocabulary
    with open("vocabulary.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)

    print(f"\nInput model: {input_model}")
    print(f"Vocabulary size: {vocab_size}")

    # Step 1: Convert to FP16
    print(f"\n{'='*60}")
    print("Step 1: Converting PyTorch model to FP16")
    print(f"{'='*60}")
    model_fp16 = convert_to_fp16(input_model, fp16_model)

    # Step 2: Export to ONNX (FP32 first, for compatibility)
    print(f"\n{'='*60}")
    print("Step 2: Exporting to ONNX (FP32)")
    print(f"{'='*60}")

    # Load FP32 model for ONNX export
    model = ImprovedDecoderTransformer(
        vocab_size=vocab_size,
        d_model=800,
        n_layers=5,
        n_heads=10,
        d_ff=3200,
        dropout=0.1
    )
    model.load_state_dict(torch.load(input_model))

    export_to_onnx(model, onnx_model, vocab_size)

    # Step 3: Optimize ONNX to FP16
    print(f"\n{'='*60}")
    print("Step 3: Optimizing ONNX model to FP16")
    print(f"{'='*60}")

    try:
        optimize_onnx_fp16(onnx_model, onnx_fp16_model)
    except Exception as e:
        print(f"⚠️  FP16 optimization failed: {e}")
        print(f"   Using FP32 ONNX model instead")

    # Summary
    print(f"\n{'='*60}")
    print("Conversion Complete!")
    print(f"{'='*60}")
    print(f"\nGenerated files:")
    print(f"  1. {fp16_model} - PyTorch FP16 model (~120 MB)")
    print(f"  2. {onnx_model} - ONNX FP32 model (~240 MB)")
    print(f"  3. {onnx_fp16_model} - ONNX FP16 model (~120 MB)")
    print(f"\nFor Android/iOS inference, use: {onnx_fp16_model}")


if __name__ == "__main__":
    main()
