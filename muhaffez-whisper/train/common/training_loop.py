"""Common training loop logic shared across all training scripts"""
import torch
import random
import time
from .data_utils import load_mel_features, tokenize_text


def run_training_epoch(model, training_tuples, vocab, criterion, optimizer, device):
    """
    Run a single training epoch

    Args:
        model: The model to train
        training_tuples: List of (file, text, target_seconds, target_words) tuples
        vocab: Vocabulary list
        criterion: Loss criterion
        optimizer: Optimizer
        device: Device to train on

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    total_iterations = 0

    random.shuffle(training_tuples)

    for seg_file, text, target_sec, target_wrd in training_tuples:
        mel_features = load_mel_features(seg_file, target_seconds=target_sec)
        audio_batch = mel_features.transpose(0, 1).unsqueeze(0).to(device)

        if target_wrd:
            words = text.split()
            if len(words) < target_wrd:
                continue
            target_text = " ".join(words[:target_wrd])
        else:
            target_text = text

        if not target_text:
            continue

        text_tokens = tokenize_text(target_text, vocab)
        full_sequence = [1] + text_tokens + [2]
        input_ids = torch.tensor([full_sequence[:-1]], dtype=torch.long, device=device)
        labels = torch.tensor([full_sequence[1:]], dtype=torch.long, device=device)

        logits = model(mel_features=audio_batch, text_ids=input_ids)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_iterations += 1

    if total_iterations == 0:
        return None

    return total_loss / total_iterations


def update_learning_rate(optimizer, avg_loss, prev_loss, min_lr=1e-7, decay_factor=0.5):
    """
    Update learning rate if loss increases

    Args:
        optimizer: The optimizer
        avg_loss: Current average loss
        prev_loss: Previous average loss
        min_lr: Minimum learning rate
        decay_factor: Factor to multiply LR by when loss increases

    Returns:
        Tuple of (new_lr, lr_changed) where lr_changed is True if LR was updated
    """
    if avg_loss > prev_loss:
        current_lr = optimizer.param_groups[0]['lr']
        new_lr = max(current_lr * decay_factor, min_lr)
        if new_lr != current_lr:
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            return new_lr, True
    return optimizer.param_groups[0]['lr'], False


def format_time(seconds):
    """Format elapsed time in a human-readable format"""
    if seconds >= 3600:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m" if minutes > 0 else f"{hours}h"
    elif seconds >= 60:
        return f"{int(round(seconds / 60))}m"
    else:
        return f"{int(round(seconds))}s"
