"""Training utilities - initial metrics, skip checks, etc."""
import torch
from .data_utils import load_mel_features, tokenize_text


def calculate_initial_metrics(model, segment_files, transcriptions, vocab, criterion, device, target_seconds=None, target_words=None):
    """
    Calculate initial loss on training data

    Args:
        model: The model to evaluate
        segment_files: List of segment file paths
        transcriptions: List of transcription texts
        vocab: Vocabulary list
        criterion: Loss criterion
        device: Device to run on
        target_seconds: Optional audio duration limit (for curriculum)
        target_words: Optional word count limit (for curriculum)

    Returns:
        Average loss across all segments
    """
    model.eval()
    total_loss = 0.0
    total_iterations = 0

    with torch.no_grad():
        for seg_file, text in zip(segment_files, transcriptions):
            mel_features = load_mel_features(seg_file, target_seconds=target_seconds)
            audio_batch = mel_features.transpose(0, 1).unsqueeze(0).to(device)

            # Extract target text (handle curriculum truncation)
            if target_words:
                words = text.split()
                if len(words) < target_words:
                    continue
                target_text = " ".join(words[:target_words])
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
            total_loss += loss.item()
            total_iterations += 1

    return total_loss / total_iterations if total_iterations > 0 else 0.0


def check_skip_training(accuracy, threshold=99.0):
    """
    Check if training should be skipped based on accuracy

    Args:
        accuracy: Current accuracy percentage
        threshold: Accuracy threshold to skip training (default 99%)

    Returns:
        True if training should be skipped, False otherwise
    """
    return accuracy > threshold
