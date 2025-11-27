#!/usr/bin/env python3
"""
Migrate old model format to new checkpoint format
Usage: python3 migrate_model.py
"""
import torch
import os
import shutil

OLD_MODEL = "../models/muhaffez_whisper.pt"
BACKUP_MODEL = "../models/muhaffez_whisper_old.pt"

def main():
    if not os.path.exists(OLD_MODEL):
        print(f"❌ Error: {OLD_MODEL} does not exist")
        return

    print("Migrating model to new checkpoint format...")
    print(f"Reading old model: {OLD_MODEL}")

    # Load old model (just state_dict)
    old_model_weights = torch.load(OLD_MODEL, map_location='cpu', weights_only=True)

    # Create backup
    shutil.copy(OLD_MODEL, BACKUP_MODEL)
    print(f"✓ Backup created: {BACKUP_MODEL}")

    # Create new checkpoint format with all three training types
    new_checkpoint = {
        'full': {
            'epoch': 0,
            'model_state_dict': old_model_weights,
            'optimizer_state_dict': None,  # Will be initialized on first training
            'loss': float('inf'),
            'lr': 1e-3,
        },
        'augmented': {
            'epoch': 0,
            'model_state_dict': old_model_weights,
            'optimizer_state_dict': None,
            'loss': float('inf'),
            'lr': 1e-3,
        },
        'curriculum': {
            'epoch': 0,
            'model_state_dict': old_model_weights,
            'optimizer_state_dict': None,
            'loss': float('inf'),
            'lr': 1e-3,
        }
    }

    # Save new format
    torch.save(new_checkpoint, OLD_MODEL)
    print(f"✓ Saved new checkpoint format: {OLD_MODEL}")

    print("\n✓ Migration complete!")
    print("  - All training types (full, augmented, curriculum) initialized")
    print("  - Model weights copied from old format")
    print("  - Optimizer states will be initialized on first training run")
    print(f"  - Old model backed up to: {BACKUP_MODEL}")

if __name__ == "__main__":
    main()
