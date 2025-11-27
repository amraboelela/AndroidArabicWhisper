"""Checkpoint management utilities for training state persistence"""
import os
import sys
import subprocess
import torch

def save_checkpoint(model, optimizer, epoch, loss, model_path="../models/muhaffez_whisper.pt", training_type="full"):
    """
    Save complete training checkpoint including model, optimizer, and metadata
    Model weights are shared across all training types; only optimizer state differs

    Args:
        model: The neural network model
        optimizer: The optimizer with current state
        epoch: Current epoch number
        loss: Current loss value
        model_path: Path for model file
        training_type: Type of training (full, augmented, curriculum)
    """
    # Load existing checkpoint if it exists
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        except:
            checkpoint = {}
    else:
        checkpoint = {}

    # Save model weights ONCE at top level (shared across all training types)
    checkpoint['model_state_dict'] = model.state_dict()

    # Update the training-type-specific optimizer state
    checkpoint[training_type] = {
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'lr': optimizer.param_groups[0]['lr'],
    }

    torch.save(checkpoint, model_path)

    # Automatically update JSON file with checkpoint metadata
    try:
        # Determine the absolute path to the model file
        abs_model_path = os.path.abspath(model_path)
        model_dir = os.path.dirname(abs_model_path)

        # Path to inspection script (should be in same directory as model)
        inspect_script = os.path.join(model_dir, "inspect_muhaffez_whisper.py")

        if os.path.exists(inspect_script):
            # Run inspection script silently to update JSON
            subprocess.run(
                [sys.executable, inspect_script, abs_model_path],
                cwd=model_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10
            )
    except Exception:
        # Silently ignore inspection errors - don't interrupt training
        pass

def load_checkpoint(model, optimizer, model_path="../models/muhaffez_whisper.pt", training_type="full", device='cpu'):
    """
    Load training checkpoint if it exists
    Model weights are loaded from top-level, optimizer state from training-type key

    Args:
        model: The neural network model to load weights into
        optimizer: The optimizer to load state into
        model_path: Path for model file
        training_type: Type of training (full, augmented, curriculum)
        device: Device to load tensors to

    Returns:
        dict with keys: epoch, loss, lr, restored (True if checkpoint was loaded)
    """
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)

            # Check if this is NEW format (with shared model_state_dict at top level)
            if 'model_state_dict' in checkpoint and training_type in checkpoint:
                # Load shared model weights
                model.load_state_dict(checkpoint['model_state_dict'])

                training_data = checkpoint[training_type]

                # Load optimizer state if it exists
                if training_data.get('optimizer_state_dict') is not None:
                    optimizer.load_state_dict(training_data['optimizer_state_dict'])
                    restored = True
                else:
                    # Model loaded but no optimizer state yet (fresh migration)
                    restored = False

                return {
                    'epoch': training_data.get('epoch', 0),
                    'loss': training_data.get('loss', float('inf')),
                    'lr': training_data.get('lr', 1e-3),
                    'restored': restored
                }
            elif training_type in checkpoint and 'model_state_dict' in checkpoint[training_type]:
                # OLD NEW format (model_state_dict duplicated per training type)
                training_data = checkpoint[training_type]
                model.load_state_dict(training_data['model_state_dict'])

                # Load optimizer state if it exists
                if training_data.get('optimizer_state_dict') is not None:
                    optimizer.load_state_dict(training_data['optimizer_state_dict'])
                    restored = True
                else:
                    restored = False

                return {
                    'epoch': training_data.get('epoch', 0),
                    'loss': training_data.get('loss', float('inf')),
                    'lr': training_data.get('lr', 1e-3),
                    'restored': restored
                }
            else:
                # OLDEST format - just model weights (state_dict)
                model.load_state_dict(checkpoint)
                print(f"⚠️  Loaded old-format model (model weights only, no optimizer state)")
                return {
                    'epoch': 0,
                    'loss': float('inf'),
                    'lr': 1e-3,
                    'restored': False
                }
        except Exception as e:
            print(f"⚠️  Warning: Could not load checkpoint: {e}")

    # Nothing to load - start fresh
    return {
        'epoch': 0,
        'loss': float('inf'),
        'lr': 1e-3,
        'restored': False
    }

def get_saved_lr(model_path="../models/muhaffez_whisper.pt", training_type="full"):
    """
    Get saved learning rate from checkpoint without loading full state

    Args:
        model_path: Path for model file
        training_type: Type of training (full, augmented, curriculum)

    Returns:
        Learning rate if found, None otherwise
    """
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
            if training_type in checkpoint:
                return checkpoint[training_type].get('lr', None)
        except Exception:
            pass

    return None
