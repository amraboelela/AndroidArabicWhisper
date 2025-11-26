"""Data loading and preprocessing utilities"""
import os
import torch


def load_mel_features(mel_path, target_seconds=None):
    """Load precomputed mel features from .pt file, optionally trimming to target_seconds"""
    if not os.path.exists(mel_path):
        raise FileNotFoundError(f"Precomputed mel features not found: {mel_path}\nPlease run precompute_mel_features.py first")

    mel_features = torch.load(mel_path, map_location='cpu', weights_only=True)

    # Trim to target seconds if specified
    # Mel features are at 100 fps (frames per second)
    if target_seconds is not None:
        target_frames = int(target_seconds * 100)
        if mel_features.shape[0] > target_frames:
            mel_features = mel_features[:target_frames, :]

    return mel_features


def tokenize_text(text, vocab):
    """Convert text to token IDs using vocabulary"""
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    words = text.split()
    return [word_to_idx.get(word, 0) for word in words]  # 0 = unknown
