#!/usr/bin/env python3
"""
Properly quantize ONNX Whisper models using ONNX Runtime quantization with preprocessing.
"""

import os
from pathlib import Path
import onnx
from onnx import numpy_helper
from onnxruntime.quantization import quantize, QuantizationMode, QuantType, QuantFormat
from onnxruntime.quantization.preprocess import quant_pre_process
import shutil


def preprocess_and_quantize_int8(model_path: str, output_path: str):
    """
    Preprocess and quantize model to INT8.
    """
    print(f"📦 Processing {model_path}...")

    # Step 1: Preprocess the model
    preprocessed_path = output_path.replace('.onnx', '_preprocessed.onnx')
    print(f"   Step 1/2: Preprocessing model...")
    try:
        quant_pre_process(
            input_model_path=model_path,
            output_model_path=preprocessed_path,
            skip_optimization=False,
            skip_onnx_shape=False,
            skip_symbolic_shape=False,
            auto_merge=True,
            int_max=2**31-1,
            guess_output_rank=False,
            verbose=0
        )
        print(f"   ✅ Preprocessing complete")
    except Exception as e:
        print(f"   ⚠️  Preprocessing had warnings, continuing...")
        print(f"       {str(e)[:100]}")
        # If preprocessing fails, use original
        shutil.copy(model_path, preprocessed_path)

    # Step 2: Quantize the preprocessed model
    print(f"   Step 2/2: Quantizing to INT8...")
    try:
        quantize(
            model_input=preprocessed_path,
            model_output=output_path,
            quant_format=QuantFormat.QDQ,  # Quantize/Dequantize format, better for NNAPI
            per_channel=True,
            reduce_range=True,  # Better compatibility with mobile hardware
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=['Conv', 'MatMul', 'Attention', 'Gemm'],  # Key operations in transformers
            optimize_model=True,
            use_external_data_format=False,
        )
        print(f"   ✅ INT8 quantization complete")
    except Exception as e:
        print(f"   ❌ Quantization failed: {e}")
        return False
    finally:
        # Clean up preprocessed file
        if os.path.exists(preprocessed_path):
            os.remove(preprocessed_path)

    return True


def convert_to_fp16(model_path: str, output_path: str):
    """
    Convert model to FP16.
    """
    print(f"📦 Converting {model_path} to FP16...")

    try:
        from onnxconverter_common import float16

        model = onnx.load(model_path)
        model_fp16 = float16.convert_float_to_float16(
            model,
            keep_io_types=True,
            disable_shape_infer=False
        )
        onnx.save(model_fp16, output_path)
        print(f"   ✅ FP16 conversion complete")
        return True
    except Exception as e:
        print(f"   ❌ FP16 conversion failed: {e}")
        return False


def main():
    print("=" * 70)
    print("🚀 ONNX Whisper Model Quantization")
    print("=" * 70)
    print()

    # Paths
    source_dir = Path("app/src/main/assets/whisper_onnx/backup_original")
    int8_output_dir = Path("app/src/main/assets/whisper_onnx_int8")
    fp16_output_dir = Path("app/src/main/assets/whisper_onnx_fp16")

    # Create output directories
    int8_output_dir.mkdir(parents=True, exist_ok=True)
    fp16_output_dir.mkdir(parents=True, exist_ok=True)

    encoder_path = source_dir / "encoder_model.onnx"
    decoder_path = source_dir / "decoder_model.onnx"

    if not encoder_path.exists() or not decoder_path.exists():
        print(f"❌ Error: Models not found in {source_dir}")
        return 1

    print(f"Source directory: {source_dir}")
    print(f"INT8 output: {int8_output_dir}")
    print(f"FP16 output: {fp16_output_dir}")
    print()

    # INT8 Quantization
    print("=" * 70)
    print("INT8 Quantization (QDQ format for NNAPI)")
    print("=" * 70)
    print()

    # Quantize encoder
    encoder_int8 = int8_output_dir / "encoder_model.onnx"
    success_enc = preprocess_and_quantize_int8(str(encoder_path), str(encoder_int8))

    if success_enc:
        original_size = os.path.getsize(encoder_path) / (1024 * 1024)
        quantized_size = os.path.getsize(encoder_int8) / (1024 * 1024)
        print(f"   📊 Encoder: {original_size:.2f} MB → {quantized_size:.2f} MB ({original_size/quantized_size:.2f}x)")
        print()

    # Quantize decoder
    decoder_int8 = int8_output_dir / "decoder_model.onnx"
    success_dec = preprocess_and_quantize_int8(str(decoder_path), str(decoder_int8))

    if success_dec:
        original_size = os.path.getsize(decoder_path) / (1024 * 1024)
        quantized_size = os.path.getsize(decoder_int8) / (1024 * 1024)
        print(f"   📊 Decoder: {original_size:.2f} MB → {quantized_size:.2f} MB ({original_size/quantized_size:.2f}x)")
        print()

    # FP16 Conversion
    print("=" * 70)
    print("FP16 Conversion")
    print("=" * 70)
    print()

    # Convert encoder
    encoder_fp16 = fp16_output_dir / "encoder_model.onnx"
    success_enc_fp16 = convert_to_fp16(str(encoder_path), str(encoder_fp16))

    if success_enc_fp16:
        original_size = os.path.getsize(encoder_path) / (1024 * 1024)
        fp16_size = os.path.getsize(encoder_fp16) / (1024 * 1024)
        print(f"   📊 Encoder: {original_size:.2f} MB → {fp16_size:.2f} MB ({original_size/fp16_size:.2f}x)")
        print()

    # Convert decoder
    decoder_fp16 = fp16_output_dir / "decoder_model.onnx"
    success_dec_fp16 = convert_to_fp16(str(decoder_path), str(decoder_fp16))

    if success_dec_fp16:
        original_size = os.path.getsize(decoder_path) / (1024 * 1024)
        fp16_size = os.path.getsize(decoder_fp16) / (1024 * 1024)
        print(f"   📊 Decoder: {original_size:.2f} MB → {fp16_size:.2f} MB ({original_size/fp16_size:.2f}x)")
        print()

    # Copy tokenizer files to output directories
    print("📋 Copying tokenizer files...")
    asset_dir = Path("app/src/main/assets/whisper_onnx")
    for file in ["vocab.json", "merges.txt", "added_tokens.json"]:
        src = asset_dir / file
        if src.exists():
            shutil.copy(src, int8_output_dir / file)
            shutil.copy(src, fp16_output_dir / file)
    print("   ✅ Tokenizer files copied")
    print()

    print("=" * 70)
    print("✅ Quantization Complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print(f"1. Test INT8 models from: {int8_output_dir}")
    print(f"2. Test FP16 models from: {fp16_output_dir}")
    print()
    print("To use INT8 models:")
    print(f"   cp {int8_output_dir}/*.onnx app/src/main/assets/whisper_onnx/")
    print()
    print("To use FP16 models:")
    print(f"   cp {fp16_output_dir}/*.onnx app/src/main/assets/whisper_onnx/")
    print()
    print("Then build and test: ./gradlew assembleDebug")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
