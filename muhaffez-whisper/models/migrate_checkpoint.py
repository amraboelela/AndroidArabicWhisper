#!/usr/bin/env python3
"""
Migrate OLD checkpoint format (raw model weights) to NEW format (with training type keys)
Usage: python3 migrate_checkpoint.py [checkpoint_file]
"""
import sys
import torch
import os

def migrate_checkpoint(checkpoint_path="muhaffez_whisper.pt"):
    """Migrate checkpoint from old format to new format"""

    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: File not found: {checkpoint_path}")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

    # Check if already in new format
    if 'full' in checkpoint or 'augmented' in checkpoint or 'curriculum' in checkpoint:
        print("✓ Checkpoint is already in NEW format")
        return

    print("Converting from OLD format (raw model weights) to NEW format...")

    # Create new checkpoint: model weights shared, optimizer states separate
    new_checkpoint = {
        'model_state_dict': checkpoint,  # Shared model weights (the old checkpoint IS the state_dict)
        'full': {
            'epoch': 0,
            'optimizer_state_dict': None,  # No optimizer state in old format
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

    # Save migrated checkpoint
    torch.save(new_checkpoint, checkpoint_path)
    print(f"\n✓ Migration complete!")
    print(f"  Saved to: {checkpoint_path}")
    print(f"  Format: NEW (with 'full', 'augmented', 'curriculum' keys)")
    print(f"  Each training type starts fresh with:")
    print(f"    - Epoch: 0")
    print(f"    - LR: 1e-3")
    print(f"    - Optimizer state: None (will be created on first training)")

if __name__ == "__main__":
    checkpoint_file = sys.argv[1] if len(sys.argv) > 1 else "muhaffez_whisper.pt"
    migrate_checkpoint(checkpoint_file)
