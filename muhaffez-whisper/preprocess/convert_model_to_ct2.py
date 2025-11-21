#!/usr/bin/env python3
"""
Convert Hugging Face whisper model to CTranslate2 format for faster-whisper
Usage: python3 convert_model_to_ct2.py
"""
import os
import sys

def main():
    # Input: Hugging Face model name
    hf_model = "tarteel-ai/whisper-base-ar-quran"

    # Output: Local CTranslate2 model directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "../models/tarteel-whisper-base-ar-quran-ct2")

    print(f"Converting {hf_model} to CTranslate2 format...")
    print(f"Output directory: {output_dir}")

    try:
        # Import the converter
        from ctranslate2.converters import TransformersConverter

        # Convert the model
        converter = TransformersConverter(hf_model)
        converter.convert(output_dir, quantization="int8")

        print(f"\n✓ Model successfully converted to: {output_dir}")
        print(f"You can now use this model with faster-whisper")

    except ImportError:
        print("❌ Error: ctranslate2 package not found")
        print("Please install it with: pip install ctranslate2")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
