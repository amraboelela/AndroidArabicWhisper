#!/usr/bin/env python3
"""
Quantize Whisper ONNX models to INT8 or FP16 for better mobile performance.
"""

import argparse
import os
from pathlib import Path
import onnx
from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType
from onnxruntime.quantization.calibrate import CalibrationDataReader
import numpy as np


def quantize_to_int8_dynamic(model_path: str, output_path: str):
    """
    Quantize model to INT8 using dynamic quantization.
    This is the easiest and fastest method, good for decoder models.
    """
    print(f"📦 Quantizing {model_path} to INT8 (dynamic)...")

    quantize_dynamic(
        model_input=model_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=True,  # Better for mobile processors
    )

    # Check file sizes
    original_size = os.path.getsize(model_path) / (1024 * 1024)
    quantized_size = os.path.getsize(output_path) / (1024 * 1024)
    compression_ratio = original_size / quantized_size

    print(f"✅ INT8 quantization complete!")
    print(f"   Original: {original_size:.2f} MB")
    print(f"   Quantized: {quantized_size:.2f} MB")
    print(f"   Compression: {compression_ratio:.2f}x smaller")
    print()


def quantize_to_fp16(model_path: str, output_path: str):
    """
    Quantize model to FP16 (half precision).
    This provides a good balance between speed and accuracy.
    """
    print(f"📦 Converting {model_path} to FP16...")

    from onnxconverter_common import float16

    model = onnx.load(model_path)
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, output_path)

    # Check file sizes
    original_size = os.path.getsize(model_path) / (1024 * 1024)
    quantized_size = os.path.getsize(output_path) / (1024 * 1024)
    compression_ratio = original_size / quantized_size

    print(f"✅ FP16 conversion complete!")
    print(f"   Original: {original_size:.2f} MB")
    print(f"   Quantized: {quantized_size:.2f} MB")
    print(f"   Compression: {compression_ratio:.2f}x smaller")
    print()


def main():
    parser = argparse.ArgumentParser(description="Quantize Whisper ONNX models")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="app/src/main/assets/whisper_onnx",
        help="Directory containing encoder_model.onnx and decoder_model.onnx"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="app/src/main/assets/whisper_onnx",
        help="Directory to save quantized models"
    )
    parser.add_argument(
        "--quant-type",
        type=str,
        choices=["int8", "fp16", "both"],
        default="int8",
        help="Quantization type: int8, fp16, or both"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Backup original models before quantizing"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder_path = input_dir / "encoder_model.onnx"
    decoder_path = input_dir / "decoder_model.onnx"

    if not encoder_path.exists() or not decoder_path.exists():
        print(f"❌ Error: Models not found in {input_dir}")
        print(f"   Looking for: encoder_model.onnx and decoder_model.onnx")
        return 1

    print("=" * 60)
    print("🚀 Whisper ONNX Model Quantization")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Quantization type: {args.quant_type}")
    print()

    # Backup if requested
    if args.backup and output_dir == input_dir:
        print("📋 Creating backups...")
        backup_dir = output_dir / "backup_original"
        backup_dir.mkdir(exist_ok=True)

        import shutil
        shutil.copy2(encoder_path, backup_dir / "encoder_model.onnx")
        shutil.copy2(decoder_path, backup_dir / "decoder_model.onnx")
        print(f"✅ Backups saved to {backup_dir}\n")

    # Quantize encoder
    if args.quant_type in ["int8", "both"]:
        encoder_int8_path = output_dir / "encoder_model.onnx"
        quantize_to_int8_dynamic(str(encoder_path), str(encoder_int8_path))

        decoder_int8_path = output_dir / "decoder_model.onnx"
        quantize_to_int8_dynamic(str(decoder_path), str(decoder_int8_path))

    if args.quant_type in ["fp16", "both"]:
        encoder_fp16_path = output_dir / "encoder_model_fp16.onnx"
        quantize_to_fp16(str(encoder_path), str(encoder_fp16_path))

        decoder_fp16_path = output_dir / "decoder_model_fp16.onnx"
        quantize_to_fp16(str(decoder_path), str(decoder_fp16_path))

    print("=" * 60)
    print("✅ All models quantized successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Build and install the app: ./gradlew assembleDebug")
    print("2. Test the quantized models on your device")
    print("3. Compare performance and accuracy with original models")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
