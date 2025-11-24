#!/usr/bin/env python3
"""
Convert Tarteel Whisper model from HuggingFace/Transformers format to CTranslate2 format
This is required for faster-whisper to use the model
"""
import os
import shutil
from ctranslate2.converters import TransformersConverter

def main():
    # Use local cached model path instead of downloading
    local_model_path = "/Users/amraboelela/.cache/huggingface/hub/models--tarteel-ai--whisper-base-ar-quran/snapshots/5c3c53fdf9272c4f6ee0bee09a1e5a4a615ee25c"

    # Output directory for converted model
    output_dir = os.path.join(os.path.dirname(__file__), "../models/tarteel_ct2")

    print(f"Converting local Tarteel model to CTranslate2 format...")
    print(f"Source: {local_model_path}")
    print(f"Output: {output_dir}")
    print()

    # Create converter using local path
    converter = TransformersConverter(local_model_path)

    # Convert with int8 quantization for smaller size and faster inference
    print("Converting... (this may take a few minutes)")
    converter.convert(output_dir, quantization="int8")

    print("✓ Model conversion complete!")
    print()

    # Copy tokenizer files
    print("Copying tokenizer files...")
    tokenizer_files = [
        "added_tokens.json",
        "merges.txt",
        "normalizer.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.json"
    ]

    for file in tokenizer_files:
        src = os.path.join(local_model_path, file)
        dst = os.path.join(output_dir, file)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  ✓ Copied {file}")
        else:
            print(f"  ⚠  Warning: {file} not found")

    print()
    print(f"✓ All files ready in: {output_dir}")
    print()
    print("The model is now ready to use in transcribe_segments.py")
    print()

if __name__ == "__main__":
    main()
