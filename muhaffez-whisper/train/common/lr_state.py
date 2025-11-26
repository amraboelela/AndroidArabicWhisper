"""Learning rate state persistence utilities"""
import json
import os

def save_lr_state(lr, model_path="../models/muhaffez_whisper.pt", training_type="full"):
    """Save current learning rate to file for specific training type"""
    lr_file = model_path.replace('.pt', '_lr.json')

    # Load existing data
    existing_data = {}
    if os.path.exists(lr_file):
        try:
            with open(lr_file, 'r') as f:
                existing_data = json.load(f)
        except (json.JSONDecodeError, KeyError):
            existing_data = {}

    # Update the specific training type
    existing_data[training_type] = lr

    # Save back
    with open(lr_file, 'w') as f:
        json.dump(existing_data, f, indent=2)

def load_lr_state(model_path="../models/muhaffez_whisper.pt", training_type="full", default_lr=1e-3):
    """Load learning rate from file for specific training type, or return default if not found"""
    lr_file = model_path.replace('.pt', '_lr.json')

    if os.path.exists(lr_file) and os.path.exists(model_path):
        try:
            with open(lr_file, 'r') as f:
                data = json.load(f)
                lr = data.get(training_type, default_lr)
                return lr
        except (json.JSONDecodeError, KeyError):
            return default_lr

    return default_lr
