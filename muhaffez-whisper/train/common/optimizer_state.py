"""Checkpoint management utilities for training state persistence"""
import os
import torch

def save_checkpoint(model, optimizer, epoch, loss, model_path="../models/muhaffez_whisper.pt", training_type="full"):
    """
    Save complete training checkpoint including model, optimizer, and metadata
    Everything is saved in ONE file with separate keys for each training type (PyTorch best practice)

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

    # Update the training-type-specific key
    checkpoint[training_type] = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'lr': optimizer.param_groups[0]['lr'],
    }

    torch.save(checkpoint, model_path)

def load_checkpoint(model, optimizer, model_path="../models/muhaffez_whisper.pt", training_type="full", device='cpu'):
    """
    Load training checkpoint if it exists

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

            # Check if this is NEW format (with training type keys)
            if training_type in checkpoint:
                training_data = checkpoint[training_type]

                # Load model state
                model.load_state_dict(training_data['model_state_dict'])

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
            else:
                # OLD format - just model weights (state_dict)
                # Try to load it as model weights directly
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
