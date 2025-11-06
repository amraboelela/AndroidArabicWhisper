#!/usr/bin/env python3
"""
Evaluate ONNX simplified model on 002-04 segments and calculate accuracy
"""

import os
import numpy as np
import onnxruntime as ort
import torchaudio
import torch
from transformers import WhisperProcessor
import re
import unicodedata

def normalize_arabic(text):
    """Normalize Arabic text for comparison"""
    # Remove diacritics
    text = ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Convert to lowercase (for Arabic this mainly affects Latin characters)
    text = text.lower()

    # Remove common punctuation
    text = re.sub(r'[.,!?;:\-_()"\']', '', text)

    return text.strip()

def calculate_wer(reference, hypothesis):
    """Calculate Word Error Rate"""
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    # Dynamic programming for edit distance
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))

    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    wer = d[len(ref_words)][len(hyp_words)] / len(ref_words) if len(ref_words) > 0 else 0
    return wer

def load_onnx_model(model_dir):
    """Load ONNX encoder and decoder"""
    print(f"Loading ONNX model from {model_dir}...")

    encoder_path = os.path.join(model_dir, "encoder_model.onnx")
    decoder_path = os.path.join(model_dir, "decoder_model.onnx")

    env = ort.SessionOptions()
    env.intra_op_num_threads = 4
    env.inter_op_num_threads = 4

    encoder_session = ort.InferenceSession(encoder_path, env)
    decoder_session = ort.InferenceSession(decoder_path, env)

    print(f"  ✓ Models loaded")
    return encoder_session, decoder_session

def transcribe_audio_onnx(encoder_session, decoder_session, processor, audio_path, max_length=200):
    """Transcribe audio using ONNX model"""
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    # Convert to numpy
    audio_array = waveform.squeeze().numpy()

    # Process audio with WhisperProcessor
    input_features = processor(
        audio_array,
        sampling_rate=sample_rate,
        return_tensors="pt"
    ).input_features

    # Convert to numpy for ONNX
    input_features_np = input_features.numpy()

    # Run encoder
    encoder_outputs = encoder_session.run(
        None,
        {"input_features": input_features_np}
    )
    encoder_hidden_states = encoder_outputs[0]

    # Initialize decoder with proper Whisper prefix tokens
    decoder_start_token_id = 50258  # <|startoftranscript|>
    lang_token_id = 50272  # <|ar|> (Arabic)
    task_token_id = 50359  # <|transcribe|>
    no_timestamps_token_id = 50363  # <|notimestamps|>
    eos_token_id = 50257  # <|endoftext|>

    # Start with proper prefix
    generated_tokens = [decoder_start_token_id, lang_token_id, task_token_id, no_timestamps_token_id]
    decoder_input_ids = np.array([generated_tokens], dtype=np.int64)

    for step in range(max_length):
        # Run decoder
        decoder_outputs = decoder_session.run(
            None,
            {
                "input_ids": decoder_input_ids,
                "encoder_hidden_states": encoder_hidden_states
            }
        )

        logits = decoder_outputs[0]  # Shape: [batch, seq_len, vocab_size]

        # Get next token (greedy)
        next_token = int(np.argmax(logits[0, -1, :]))

        # Stop if EOS token
        if next_token == eos_token_id:
            break

        generated_tokens.append(next_token)

        # Update decoder input with all tokens so far
        decoder_input_ids = np.array([generated_tokens], dtype=np.int64)

    # Decode tokens
    transcription = processor.decode(generated_tokens, skip_special_tokens=True)

    return transcription

def main():
    print("="*70)
    print("Evaluating ONNX Simplified Model on 002-04 Segments")
    print("="*70)

    model_dir = "models/custom-whisper-ar-quran-onnx-simplified"
    audio_dir = "datasets/base/audio"
    text_file = "datasets/base/text/002-04.txt"

    # Load reference text
    print(f"\nLoading reference text from {text_file}...")
    with open(text_file, 'r', encoding='utf-8') as f:
        reference_lines = [line.strip() for line in f.readlines() if line.strip()]
    print(f"  ✓ Loaded {len(reference_lines)} reference lines")

    # Load processor
    print("\nLoading processor...")
    processor = WhisperProcessor.from_pretrained(model_dir)
    print("  ✓ Processor loaded")

    # Load ONNX models
    print()
    encoder_session, decoder_session = load_onnx_model(model_dir)

    # Find all 002-04 audio files
    audio_files = sorted([f for f in os.listdir(audio_dir) if f.startswith("002-04-") and f.endswith(".wav")])

    print(f"\n{'='*70}")
    print(f"Testing {len(audio_files)} audio files")
    print(f"{'='*70}\n")

    results = []
    total_wer = 0.0
    perfect_matches = 0

    for i, audio_file in enumerate(audio_files):
        audio_path = os.path.join(audio_dir, audio_file)

        # Get corresponding reference text
        # 002-04-01.wav corresponds to line 0 (index-1)
        segment_num = int(audio_file.split('-')[2].split('.')[0])
        ref_index = segment_num - 1

        if ref_index >= len(reference_lines):
            print(f"⚠️  {audio_file}: No reference text (index {ref_index})")
            continue

        reference = reference_lines[ref_index]

        print(f"[{i+1}/{len(audio_files)}] {audio_file}")

        try:
            # Transcribe
            hypothesis = transcribe_audio_onnx(
                encoder_session,
                decoder_session,
                processor,
                audio_path,
                max_length=200
            )

            # Normalize both texts
            ref_normalized = normalize_arabic(reference)
            hyp_normalized = normalize_arabic(hypothesis)

            # Calculate WER
            wer = calculate_wer(ref_normalized, hyp_normalized)
            total_wer += wer

            # Check if perfect match
            if ref_normalized == hyp_normalized:
                perfect_matches += 1
                match_symbol = "✅"
            else:
                match_symbol = "❌"

            print(f"  Reference: {reference}")
            print(f"  Hypothesis: {hypothesis}")
            print(f"  WER: {wer*100:.2f}% {match_symbol}")

            results.append({
                'file': audio_file,
                'reference': reference,
                'hypothesis': hypothesis,
                'wer': wer,
                'perfect': ref_normalized == hyp_normalized
            })
            print()

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            print()

    # Print summary
    print("="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"\nTotal files tested: {len(results)}")
    print(f"Perfect matches: {perfect_matches}/{len(results)} ({perfect_matches/len(results)*100:.1f}%)")

    if results:
        avg_wer = total_wer / len(results)
        accuracy = (1 - avg_wer) * 100
        print(f"\nAverage WER: {avg_wer*100:.2f}%")
        print(f"Average Accuracy: {accuracy:.2f}%")

    # Show worst cases
    if results:
        print(f"\n{'='*70}")
        print("WORST 5 CASES (by WER)")
        print(f"{'='*70}")
        worst_results = sorted(results, key=lambda x: x['wer'], reverse=True)[:5]
        for r in worst_results:
            print(f"\n{r['file']} - WER: {r['wer']*100:.2f}%")
            print(f"  Ref: {r['reference']}")
            print(f"  Hyp: {r['hypothesis']}")

    print(f"\n{'='*70}")
    print("✓ Evaluation Complete!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
