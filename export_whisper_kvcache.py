#!/usr/bin/env python3
"""
Export Whisper model to ONNX with KV-cache support for faster decoding.
"""

import sys
from pathlib import Path

def export_whisper_with_kv_cache():
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

    model_path = "onnx/models/custom-whisper-ar-quran"
    output_path = "onnx/models/custom-whisper-ar-quran-onnx-kvcache"

    print("=" * 70)
    print("🚀 Exporting Whisper Model with KV-Cache")
    print("=" * 70)
    print(f"Source model: {model_path}")
    print(f"Output path: {output_path}")
    print()

    if not Path(model_path).exists():
        print(f"❌ Error: Model not found at {model_path}")
        return 1

    # Create output directory
    Path(output_path).mkdir(parents=True, exist_ok=True)

    print("📦 Loading and exporting model with KV-cache...")
    print("   This will take a few minutes...")
    print()

    try:
        # Load the processor first
        print("   Loading processor...")
        processor = WhisperProcessor.from_pretrained(model_path)

        # Export to ONNX with KV-cache
        print("   Exporting to ONNX with use_cache=True (KV-cache enabled)...")
        ort_model = ORTModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            export=True,
            use_cache=True,  # Enable KV-cache - this is the key!
        )

        print("   Saving exported model...")
        ort_model.save_pretrained(output_path)
        processor.save_pretrained(output_path)

        print()
        print("✅ Export complete!")
        print()

        # Check what was created
        print("📊 Exported files:")
        for file in sorted(Path(output_path).iterdir()):
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"   {file.name}: {size_mb:.2f} MB")

        print()
        print("=" * 70)
        print("KV-Cache Model Structure:")
        print("=" * 70)
        print("The decoder now has MULTIPLE inputs/outputs:")
        print("  Inputs:")
        print("    - input_ids: Current token")
        print("    - encoder_hidden_states: From encoder")
        print("    - past_key_values: Cached attention states (6 layers x 4 tensors)")
        print()
        print("  Outputs:")
        print("    - logits: Next token predictions")
        print("    - present_key_values: Updated cache for next iteration")
        print()
        print("=" * 70)
        print("Next steps:")
        print("=" * 70)
        print(f"1. Copy decoder models:")
        print(f"   cp {output_path}/decoder_*.onnx app/src/main/assets/whisper_onnx/")
        print()
        print("2. Copy encoder (if changed):")
        print(f"   cp {output_path}/encoder_model.onnx app/src/main/assets/whisper_onnx/")
        print()
        print("3. Update Kotlin code to handle KV-cache inputs/outputs")
        print("   (I'll help you with this)")
        print()

        return 0

    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(export_whisper_with_kv_cache())
