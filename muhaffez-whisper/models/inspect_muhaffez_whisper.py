#!/usr/bin/env python3
"""
Inspect muhaffez_whisper checkpoint and save metadata to JSON file
Usage: python3 inspect_muhaffez_model.py [checkpoint_file] [output_json]
"""
import sys
import torch
import json
import os

def inspect_checkpoint(checkpoint_path="muhaffez_whisper.pt", json_output=None):
    """Inspect checkpoint and save non-tensor metadata to JSON"""

    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint file not found: {checkpoint_path}")
        return

    # Default JSON output name based on checkpoint name
    if json_output is None:
        base_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        json_output = f"{base_name}.json"

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

    # Build metadata dictionary (excluding large tensors)
    metadata = {}

    if isinstance(checkpoint, dict):
        for key, value in checkpoint.items():
            if key == 'model_state_dict':
                # Shared model weights
                if isinstance(value, dict):
                    tensor_count = len(value)
                    metadata[key] = f"<{tensor_count} tensors>"
                else:
                    metadata[key] = "<present>"
            elif isinstance(value, dict):
                # This is a training type key (full, augmented, curriculum)
                metadata[key] = {}
                for sub_key, sub_value in value.items():
                    if sub_key in ['model_state_dict', 'optimizer_state_dict']:
                        # Don't print tensor details, just indicate presence/absence
                        if sub_value is None:
                            metadata[key][sub_key] = None
                        elif isinstance(sub_value, dict):
                            # Count tensors
                            tensor_count = len(sub_value)
                            metadata[key][sub_key] = f"<{tensor_count} tensors>"
                        else:
                            metadata[key][sub_key] = "<present>"
                    else:
                        # Include scalar values (epoch, loss, lr)
                        metadata[key][sub_key] = sub_value
            else:
                # Top-level non-dict value
                metadata[key] = str(value) if not isinstance(value, (int, float, str, bool, type(None))) else value

    # Save to JSON file
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(metadata, indent=2, fp=f, default=str)

    print(f"\n✓ Checkpoint metadata saved to: {json_output}")
    print(f"\nMetadata:")
    print("=" * 60)
    print(json.dumps(metadata, indent=2, default=str))
    print("=" * 60)

    # Summary
    print("\nSummary:")

    # Check for shared model weights
    if 'model_state_dict' in checkpoint:
        print(f"\n✓ Shared model weights: {len(checkpoint['model_state_dict'])} tensors")

    if isinstance(checkpoint, dict):
        for key in ['full', 'augmented', 'curriculum']:
            if key in checkpoint:
                data = checkpoint[key]
                # Check if model_state_dict is in training type key (old format)
                has_model_inline = data.get('model_state_dict') is not None
                has_optimizer = data.get('optimizer_state_dict') is not None
                epoch = data.get('epoch', 'N/A')
                lr = data.get('lr', 'N/A')
                loss = data.get('loss', 'N/A')

                print(f"\n{key.upper()}:")
                print(f"  Epoch: {epoch}")
                print(f"  LR: {lr}")
                print(f"  Loss: {loss}")
                if has_model_inline:
                    print(f"  Model: ✓ (inline - old format)")
                print(f"  Optimizer: {'✓' if has_optimizer else '✗'}")

if __name__ == "__main__":
    checkpoint_file = sys.argv[1] if len(sys.argv) > 1 else "muhaffez_whisper.pt"
    json_file = sys.argv[2] if len(sys.argv) > 2 else None
    inspect_checkpoint(checkpoint_file, json_file)
