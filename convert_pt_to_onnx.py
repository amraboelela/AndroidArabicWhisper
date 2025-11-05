#!/usr/bin/env python3
"""
Convert encoder_decoder_model.pt to ONNX format for Android.
"""

import sys
import torch
from pathlib import Path

def convert_pytorch_to_onnx():
    model_path = "onnx/models/encoder_decoder_model.pt"
    output_dir = Path("onnx/models/encoder_decoder_onnx")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🚀 Converting PyTorch Model to ONNX")
    print("=" * 70)
    print(f"Source: {model_path}")
    print(f"Output: {output_dir}")
    print()

    # Load the model
    print("📦 Loading PyTorch model...")
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"   ✅ Model loaded")
        print()

        # Inspect the checkpoint structure
        print("📊 Checkpoint structure:")
        if isinstance(checkpoint, dict):
            print("   Type: Dictionary")
            print("   Keys:", list(checkpoint.keys())[:10])

            # Check if it's a state dict or a full model
            if 'model' in checkpoint:
                print("   Contains 'model' key")
                model = checkpoint['model']
            elif 'state_dict' in checkpoint:
                print("   Contains 'state_dict' key")
                # Need to load into a model architecture
                print("   ⚠️  This is a state dict, need model architecture")
            else:
                print("   Direct state dict format")
                # This might be the model state dict itself

            # Print some keys to understand structure
            if isinstance(checkpoint, dict):
                print("\n   Sample keys:")
                for i, key in enumerate(list(checkpoint.keys())[:5]):
                    value = checkpoint[key]
                    if isinstance(value, torch.Tensor):
                        print(f"     {key}: Tensor {value.shape}")
                    else:
                        print(f"     {key}: {type(value)}")

        elif isinstance(checkpoint, torch.nn.Module):
            print("   Type: torch.nn.Module (full model)")
            model = checkpoint
        else:
            print(f"   Type: {type(checkpoint)}")

        print()

        # Try to understand what kind of model this is
        print("=" * 70)
        print("Analysis:")
        print("=" * 70)
        print()

        # Check if it's a Whisper model by looking at keys
        if isinstance(checkpoint, dict):
            keys = list(checkpoint.keys())
            has_encoder = any('encoder' in k for k in keys)
            has_decoder = any('decoder' in k for k in keys)

            print(f"   Has encoder keys: {has_encoder}")
            print(f"   Has decoder keys: {has_decoder}")
            print(f"   Total parameters: {len(keys)}")
            print()

            if has_encoder and has_decoder:
                print("   ✅ This looks like an encoder-decoder model")
                print()
                print("To export to ONNX, we need:")
                print("1. The model architecture (WhisperForConditionalGeneration)")
                print("2. Load this state dict into that architecture")
                print("3. Then export to ONNX")
                print()

                # Try loading into Whisper architecture
                try:
                    from transformers import WhisperForConditionalGeneration, WhisperConfig

                    print("   Attempting to load into Whisper architecture...")

                    # Try to infer config from the checkpoint
                    # First, let's check the shapes to understand model size
                    print("\n   Analyzing model dimensions...")

                    # Look for embedding dimensions
                    for key in keys:
                        if 'embed_tokens.weight' in key:
                            shape = checkpoint[key].shape
                            print(f"     {key}: {shape}")
                            vocab_size, d_model = shape
                            print(f"     → vocab_size: {vocab_size}, d_model: {d_model}")
                            break

                    # Count decoder layers
                    decoder_layers = max([
                        int(k.split('.')[2]) for k in keys
                        if 'model.decoder.layers.' in k and k.split('.')[2].isdigit()
                    ], default=0) + 1

                    # Count encoder layers
                    encoder_layers = max([
                        int(k.split('.')[2]) for k in keys
                        if 'model.encoder.layers.' in k and k.split('.')[2].isdigit()
                    ], default=0) + 1

                    print(f"     Encoder layers: {encoder_layers}")
                    print(f"     Decoder layers: {decoder_layers}")
                    print()

                    # This seems to be a smaller/custom model
                    print("=" * 70)
                    print("Next Steps:")
                    print("=" * 70)
                    print()
                    print("This appears to be a state dict. To convert to ONNX:")
                    print()
                    print("1. Identify the exact model architecture")
                    print("2. Create a model with that architecture")
                    print("3. Load this state dict: model.load_state_dict(checkpoint)")
                    print("4. Export to ONNX")
                    print()
                    print("Do you know what model architecture this is from?")
                    print("(e.g., whisper-tiny, whisper-base, custom model?)")

                except Exception as e:
                    print(f"   ❌ Error analyzing: {e}")

        return 0

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(convert_pytorch_to_onnx())
