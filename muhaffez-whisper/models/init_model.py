#!/usr/bin/env python3
"""
Initialize a brand new model checkpoint with NEW format (random weights)
Usage: python3 init_model.py [output_path] [vocab_path]

Creates a fresh model with:
- Random PyTorch initialization
- NEW checkpoint format (full/augmented/curriculum keys)
- All training types starting at epoch 0, LR=1e-3
"""
import sys
import os
import subprocess
import torch
import json
from encoder_decoder_transformer import EncoderDecoderTransformer

def init_model(output_path="muhaffez_whisper.pt", vocab_path="vocabulary.json"):
    """Create a new model with random initialization in NEW checkpoint format"""

    # Load vocabulary to get vocab size
    if not os.path.exists(vocab_path):
        print(f"❌ Error: Vocabulary file not found: {vocab_path}")
        return

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    vocab_size = len(vocab)
    print(f"Creating new model with vocabulary size: {vocab_size}")

    # Create model with same architecture as training scripts
    model = EncoderDecoderTransformer(
        vocab_size=vocab_size,
        d_model=128,
        n_encoder_layers=4,
        n_decoder_layers=4,
        n_heads=4,
        d_ff=512,
        dropout=0.1
    )

    print(f"Model architecture:")
    print(f"  - d_model: 128")
    print(f"  - Encoder layers: 4")
    print(f"  - Decoder layers: 4")
    print(f"  - Attention heads: 4")
    print(f"  - Feed-forward dim: 512")
    print(f"  - Dropout: 0.1")

    # Get state dict (randomly initialized weights)
    state_dict = model.state_dict()

    # Create NEW format checkpoint: model weights shared, optimizer states separate
    checkpoint = {
        'model_state_dict': state_dict,  # Shared model weights
        'full': {
            'epoch': 0,
            'optimizer_state_dict': None,
            'loss': float('inf'),
            'lr': 1e-3,
        },
        'augmented': {
            'epoch': 0,
            'optimizer_state_dict': None,
            'loss': float('inf'),
            'lr': 1e-3,
        },
        'curriculum': {
            'epoch': 0,
            'optimizer_state_dict': None,
            'loss': float('inf'),
            'lr': 1e-3,
        }
    }

    # Save checkpoint
    torch.save(checkpoint, output_path)

    print(f"\n✓ Model initialized successfully!")
    print(f"  Saved to: {output_path}")
    print(f"  Format: NEW (with 'full', 'augmented', 'curriculum' keys)")
    print(f"  Initialization: Random (PyTorch default)")
    print(f"  Ready for training with any training type")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Automatically update JSON file with checkpoint metadata
    try:
        abs_model_path = os.path.abspath(output_path)
        model_dir = os.path.dirname(abs_model_path)
        inspect_script = os.path.join(model_dir, "inspect_muhaffez_whisper.py")

        if os.path.exists(inspect_script):
            subprocess.run(
                [sys.executable, inspect_script, abs_model_path],
                cwd=model_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10
            )
            print(f"  JSON metadata updated: {os.path.splitext(output_path)[0]}.json")
    except Exception:
        # Silently ignore inspection errors
        pass

if __name__ == "__main__":
    output_file = sys.argv[1] if len(sys.argv) > 1 else "muhaffez_whisper.pt"
    vocab_file = sys.argv[2] if len(sys.argv) > 2 else "vocabulary.json"
    init_model(output_file, vocab_file)
