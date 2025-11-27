"""Accuracy and metrics calculation"""
import torch
from .data_utils import load_mel_features


def calculate_accuracy(model, segment_files, transcriptions, vocab, device):
    """Calculate overall accuracy on regular (non-augmented) segments only"""
    model.eval()
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for seg_file, expected_text in zip(segment_files, transcriptions):
            # Load precomputed mel features
            mel_features = load_mel_features(seg_file)
            audio_batch = mel_features.transpose(0, 1).unsqueeze(0).to(device)

            # Calculate audio duration from mel spectrogram
            # For 8kHz audio with hop_length=80: duration = (time_frames * hop_length) / sample_rate
            time_frames = mel_features.shape[0]
            sample_rate = 8000
            hop_length = 80
            audio_duration = (time_frames * hop_length) / sample_rate

            # Generate
            generated = model.generate(
                audio_batch,
                max_new_tokens=50,
                temperature=1.0,
                min_tokens=1,
                use_sampling=False,
                audio_duration_seconds=audio_duration
            )
            tokens = generated[0].tolist()

            # Clean tokens
            if tokens and tokens[0] == 1:
                tokens = tokens[1:]
            if 2 in tokens:
                tokens = tokens[:tokens.index(2)]

            generated_words = [vocab[idx] for idx in tokens if idx < len(vocab)]
            generated_text = " ".join(generated_words)

            # Token-level accuracy (word-by-word comparison)
            expected_words = expected_text.split()
            min_len = min(len(expected_words), len(generated_words))
            total_correct += sum(1 for i in range(min_len) if generated_words[i] == expected_words[i])
            total_tokens += len(expected_words)

    accuracy = (total_correct / total_tokens * 100) if total_tokens > 0 else 0.0
    return accuracy


def calculate_comprehensive_accuracy(model, segment_files, transcriptions, vocab, target_seconds, target_words, device, debug=False):
    """Calculate accuracy across all segments"""
    model.eval()
    total_correct = 0
    total_expected = 0
    segment_accuracies = []

    with torch.no_grad():
        for idx, (seg_file, transcription) in enumerate(zip(segment_files, transcriptions)):
            # Load precomputed mel features
            audio_features = load_mel_features(seg_file, target_seconds=target_seconds)
            audio_batch = audio_features.transpose(0, 1).unsqueeze(0).to(device)

            # Get expected text
            expected_words = transcription.split()[:target_words] if target_words else transcription.split()
            expected_text = " ".join(expected_words)

            if not expected_text:
                continue

            # Generate with timeout protection
            # For large datasets, reduce max_tokens to avoid hanging
            if len(segment_files) > 20:
                max_tokens = min((target_words * 5) if target_words else 30, 50)
            else:
                max_tokens = min((target_words * 10) if target_words else 50, 100)

            # Calculate audio duration from mel features (100 fps)
            audio_duration = target_seconds if target_seconds else (audio_features.shape[0] / 100.0)

            try:
                generated = model.generate(audio_batch, max_new_tokens=max_tokens, audio_duration_seconds=audio_duration, use_sampling=False)
                generated_ids = generated[0].tolist()
            except Exception as e:
                print(f"    Warning: Generation failed for segment {idx}: {e}", flush=True)
                continue

            # Clean up generated IDs
            if generated_ids and generated_ids[0] == 1:
                generated_ids = generated_ids[1:]
            if 2 in generated_ids:
                generated_ids = generated_ids[:generated_ids.index(2)]

            generated_words = [vocab[idx] for idx in generated_ids if idx < len(vocab)]

            # Calculate confidence and filter low confidence words
            if len(generated_ids) > 0:
                encoder_output = model.encode(audio_batch)
                text_ids = torch.tensor([[1] + generated_ids[:len(generated_words)]], dtype=torch.long, device=device)
                logits, _ = model.decode(text_ids, encoder_output)
                probs = torch.softmax(logits, dim=-1)

                # Get confident words only (>= 20% threshold)
                confident_words = []
                for i, token_id in enumerate(generated_ids[:len(generated_words)]):
                    if i < logits.shape[1] - 1:
                        token_prob = probs[0, i, token_id].item()
                        if token_prob >= 0.2:  # 20% threshold
                            confident_words.append(generated_words[i])

                # Count correct confident words
                correct = sum(1 for i, word in enumerate(confident_words) if i < len(expected_words) and word == expected_words[i])
            else:
                correct = 0

            # Calculate segment accuracy
            segment_acc = (correct / len(expected_words) * 100) if expected_words else 0
            segment_accuracies.append(segment_acc)
            total_correct += correct
            total_expected += len(expected_words)

            if debug:
                print(f"  Seg {idx}: expected={len(expected_words)} words, correct={correct}, acc={segment_acc:.1f}%")

    # Calculate overall accuracy
    overall_accuracy = (total_correct / total_expected * 100) if total_expected > 0 else 0
    avg_segment_accuracy = sum(segment_accuracies) / len(segment_accuracies) if segment_accuracies else 0

    return overall_accuracy, avg_segment_accuracy, segment_accuracies


def calculate_curriculum_accuracy(model, all_curriculum_files, all_curriculum_transcriptions,
                                   all_curriculum_target_seconds, all_curriculum_target_words,
                                   vocab, device, sample_rate=8):
    """
    Calculate accuracy on curriculum samples (mixed stages of varying lengths).

    This tests the model on a representative sample of curriculum data,
    including short sequences (1-2 words) and longer sequences (up to full length).

    Args:
        model: The trained model
        all_curriculum_files: List of all curriculum mel file paths
        all_curriculum_transcriptions: List of all curriculum transcriptions
        all_curriculum_target_seconds: List of target audio durations for each sample
        all_curriculum_target_words: List of target word counts for each sample
        vocab: Vocabulary list
        device: torch device
        sample_rate: Sample every Nth curriculum sample (default: 8)

    Returns:
        float: Overall accuracy percentage
    """
    # Test on curriculum-appropriate samples (sample from all curriculum stages)
    # Take a representative sample of curriculum data (e.g., every 8th sample)
    sample_indices = list(range(0, len(all_curriculum_files), sample_rate))
    sample_files = [all_curriculum_files[i] for i in sample_indices]
    sample_texts = [all_curriculum_transcriptions[i] for i in sample_indices]
    sample_target_secs = [all_curriculum_target_seconds[i] for i in sample_indices]
    sample_target_wrds = [all_curriculum_target_words[i] for i in sample_indices]

    # Calculate accuracy on curriculum samples
    total_correct = 0
    total_expected = 0
    model.eval()

    with torch.no_grad():
        for seg_file, transcription, target_sec, target_wrd in zip(sample_files, sample_texts, sample_target_secs, sample_target_wrds):
            audio_features = load_mel_features(seg_file, target_seconds=target_sec)
            audio_batch = audio_features.transpose(0, 1).unsqueeze(0).to(device)

            expected_words = transcription.split()[:target_wrd] if target_wrd else transcription.split()
            if not expected_words:
                continue

            audio_duration = target_sec if target_sec else (audio_features.shape[0] / 100.0)
            max_tokens = min((target_wrd * 5) if target_wrd else 30, 50)

            try:
                generated = model.generate(audio_batch, max_new_tokens=max_tokens, audio_duration_seconds=audio_duration, use_sampling=False)
                generated_ids = generated[0].tolist()
            except:
                continue

            if generated_ids and generated_ids[0] == 1:
                generated_ids = generated_ids[1:]
            if 2 in generated_ids:
                generated_ids = generated_ids[:generated_ids.index(2)]

            generated_words = [vocab[idx] for idx in generated_ids if idx < len(vocab)]

            # Calculate with confidence filtering (20% threshold)
            if len(generated_ids) > 0:
                encoder_output = model.encode(audio_batch)
                text_ids = torch.tensor([[1] + generated_ids[:len(generated_words)]], dtype=torch.long, device=device)
                logits, _ = model.decode(text_ids, encoder_output)
                probs = torch.softmax(logits, dim=-1)

                confident_words = []
                for i, token_id in enumerate(generated_ids[:len(generated_words)]):
                    if i < logits.shape[1] - 1:
                        token_prob = probs[0, i, token_id].item()
                        if token_prob >= 0.2:  # 20% threshold
                            confident_words.append(generated_words[i])

                correct = sum(1 for i, word in enumerate(confident_words) if i < len(expected_words) and word == expected_words[i])
            else:
                correct = 0

            total_correct += correct
            total_expected += len(expected_words)

    overall_acc = (total_correct / total_expected * 100) if total_expected > 0 else 0
    return overall_acc
