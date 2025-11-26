"""Learning rate state persistence utilities"""
import json
import os

def save_lr_state(lr, model_path="../models/muhaffez_whisper.pt"):
    """Save current learning rate to file"""
    lr_file = model_path.replace('.pt', '_lr.json')
    with open(lr_file, 'w') as f:
        json.dump({'learning_rate': lr}, f)

def load_lr_state(model_path="../models/muhaffez_whisper.pt", default_lr=1e-3):
    """Load learning rate from file, or return default if not found"""
    lr_file = model_path.replace('.pt', '_lr.json')

    if os.path.exists(lr_file) and os.path.exists(model_path):
        try:
            with open(lr_file, 'r') as f:
                data = json.load(f)
                lr = data.get('learning_rate', default_lr)
                return lr
        except (json.JSONDecodeError, KeyError):
            return default_lr

    return default_lr
