#!/bin/bash

# Reset optimizer states by removing them from the checkpoint
# This will force all training types to start with fresh optimizer state (LR=1e-3, no momentum)
# Model weights are preserved!
# Usage: ./reset_optimizer.sh

cd "$(dirname "$0")"

python3 << 'EOF'
import torch
import os

MODEL_FILE = "../models/muhaffez_whisper.pt"

if not os.path.exists(MODEL_FILE):
    print(f"⚠️  {MODEL_FILE} does not exist")
    exit(0)

print("Resetting optimizer states (preserving model weights)...")

# Load checkpoint
checkpoint = torch.load(MODEL_FILE, map_location='cpu', weights_only=True)

# Reset optimizer states for each training type
for training_type in ['full', 'augmented', 'curriculum']:
    if training_type in checkpoint:
        checkpoint[training_type]['optimizer_state_dict'] = None
        checkpoint[training_type]['lr'] = 1e-3
        checkpoint[training_type]['epoch'] = 0
        checkpoint[training_type]['loss'] = float('inf')
        checkpoint[training_type]['accuracy'] = 0.0
        print(f"  ✓ Reset {training_type} optimizer state")

# Save updated checkpoint
torch.save(checkpoint, MODEL_FILE)

print("\n✓ Optimizer states reset!")
print("  - Model weights preserved")
print("  - Learning rates reset to 1.0e-03")
print("  - Momentum buffers cleared")
print("")
print("Next training session will start with fresh optimizer state")
EOF
