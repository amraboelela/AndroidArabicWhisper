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
