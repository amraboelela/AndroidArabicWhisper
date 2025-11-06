#!/usr/bin/env python3
"""
Convert custom vocabulary from list format to Whisper dictionary format.
"""

import json
import shutil
from pathlib import Path

def convert_vocabulary():
    print("=" * 70)
    print("🔧 Fixing Custom Vocabulary for ONNX Model")
    print("=" * 70)
    print()

    # Load custom vocabulary (list format)
    custom_vocab_path = Path("onnx/vocabulary.json")
    print(f"📖 Loading custom vocabulary from: {custom_vocab_path}")

    with open(custom_vocab_path, 'r', encoding='utf-8') as f:
        vocab_list = json.load(f)

    print(f"   ✅ Loaded {len(vocab_list)} tokens")
    print(f"   First 10 tokens: {vocab_list[:10]}")
    print()

    # Convert to dictionary format {token: id}
    print("🔄 Converting to Whisper dictionary format...")
    vocab_dict = {token: idx for idx, token in enumerate(vocab_list)}
    print(f"   ✅ Converted to dictionary format")
    print()

    # Output directory
    output_dir = Path("app/src/main/assets/whisper_onnx_custom")

    if not output_dir.exists():
        print(f"❌ Error: Directory not found: {output_dir}")
        print(f"   Please run export_custom_model.py first!")
        return 1

    # Save converted vocabulary
    output_vocab = output_dir / "vocab.json"
    print(f"💾 Saving converted vocabulary to: {output_vocab}")

    with open(output_vocab, 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=None)

    orig_size = custom_vocab_path.stat().st_size / 1024
    new_size = output_vocab.stat().st_size / 1024
    print(f"   ✅ Saved: {new_size:.1f} KB (original: {orig_size:.1f} KB)")
    print()

    # Check what other tokenizer files we need
    print("=" * 70)
    print("📋 Checking Other Tokenizer Files")
    print("=" * 70)
    print()

    # Copy merges.txt if it exists in onnx/
    onnx_merges = Path("onnx/merges.txt")
    if onnx_merges.exists():
        output_merges = output_dir / "merges.txt"
        shutil.copy(onnx_merges, output_merges)
        print(f"   ✅ Copied merges.txt")
    else:
        # Check if we already have merges.txt (from whisper_onnx)
        existing_merges = output_dir / "merges.txt"
        if existing_merges.exists():
            print(f"   ⚠️  Using existing merges.txt (from Whisper)")
            print(f"      If vocabulary was retrained, merges.txt should also be updated!")
        else:
            print(f"   ❌ merges.txt not found!")

    # Copy added_tokens.json if it exists
    onnx_added = Path("onnx/added_tokens.json")
    if onnx_added.exists():
        output_added = output_dir / "added_tokens.json"
        shutil.copy(onnx_added, output_added)
        print(f"   ✅ Copied added_tokens.json")
    else:
        existing_added = output_dir / "added_tokens.json"
        if existing_added.exists():
            print(f"   ℹ️  Using existing added_tokens.json")
        else:
            # Create a minimal one with special tokens
            print(f"   ⚠️  Creating minimal added_tokens.json with special tokens")
            added_tokens = {}
            output_added = output_dir / "added_tokens.json"
            with open(output_added, 'w', encoding='utf-8') as f:
                json.dump(added_tokens, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 70)
    print("✅ Custom Vocabulary Fixed!")
    print("=" * 70)
    print()
    print(f"📊 Summary:")
    print(f"   Vocabulary size: {len(vocab_list)} tokens")
    print(f"   Format: Dictionary (Whisper-compatible)")
    print(f"   Location: {output_dir}")
    print()
    print("📁 Files in custom model directory:")
    for file in sorted(output_dir.iterdir()):
        size = file.stat().st_size / (1024 * 1024) if file.stat().st_size > 1024*1024 else file.stat().st_size / 1024
        unit = "MB" if file.stat().st_size > 1024*1024 else "KB"
        print(f"   {file.name:30} {size:8.1f} {unit}")
    print()
    print("⚠️  Important Note:")
    print("   Your custom vocabulary has 14,754 tokens (not 50,000).")
    print("   Make sure your Kotlin code is configured to use this vocab size!")
    print()
    print("Next steps:")
    print("1. Update WhisperOnnxKotlinHelper.kt to use vocab_size = 14754")
    print("2. Test the custom model with Android app")
    print("3. Compare results with standard Whisper model")
    print()

    return 0

if __name__ == "__main__":
    import sys
    try:
        sys.exit(convert_vocabulary())
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
