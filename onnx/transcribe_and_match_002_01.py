#!/usr/bin/env python3
"""
Transcribe 002-01 segments using faster_whisper, normalize, and match against quran-simple-norm.txt
"""
import os
import re
from faster_whisper import WhisperModel
from difflib import SequenceMatcher

def remove_tashkeel(text):
    """Remove Arabic diacritics (tashkeel)"""
    tashkeel_pattern = r'[\u064B-\u065F\u0670]'
    return re.sub(tashkeel_pattern, '', text)

def remove_control_characters(text):
    """Remove control characters"""
    control_chars = r'[\u200B-\u200F\u202A-\u202E\u2060-\u2069\uFEFF]'
    return re.sub(control_chars, '', text)

def normalize_arabic(text):
    """Normalize Arabic text"""
    text = remove_tashkeel(text)
    text = remove_control_characters(text)

    # Normalize hamza variants
    hamza_map = {
        'إ': 'ا',
        'أ': 'ا',
        'آ': 'ا',
    }

    for old_char, new_char in hamza_map.items():
        text = text.replace(old_char, new_char)

    return text.strip()

def find_best_match(transcription, quran_lines):
    """Find the best matching line in the Quran text"""
    normalized_trans = normalize_arabic(transcription)

    best_match_idx = -1
    best_ratio = 0.0

    for idx, line in enumerate(quran_lines):
        normalized_line = normalize_arabic(line)

        # Skip empty lines and separators
        if not normalized_line or normalized_line in ['-', '*']:
            continue

        # Calculate similarity
        ratio = SequenceMatcher(None, normalized_trans, normalized_line).ratio()

        if ratio > best_ratio:
            best_ratio = ratio
            best_match_idx = idx

    return best_match_idx, best_ratio

def main():
    datasets_dir = "datasets/base"
    output_file = "002-01.txt"
    quran_file = "quran-simple-norm.txt"

    # Load Quran text
    with open(quran_file, "r", encoding="utf-8") as f:
        quran_lines = f.readlines()

    print(f"Loaded {len(quran_lines)} lines from {quran_file}")

    # Initialize faster_whisper model (using local CT2 model)
    print("\nLoading faster_whisper model...")
    model_path = "../app/src/main/assets/whisper_ct2/"
    model = WhisperModel(model_path, device="cpu")

    # Get all 002-01 segments
    segment_files = sorted([
        f for f in os.listdir(datasets_dir)
        if f.startswith("002-01-") and f.endswith(".wav")
    ])

    print(f"\nFound {len(segment_files)} segments to transcribe\n")

    results = []

    for i, segment_file in enumerate(segment_files, 1):
        segment_path = os.path.join(datasets_dir, segment_file)

        print(f"[{i}/{len(segment_files)}] Transcribing {segment_file}...", end=" ")

        # Transcribe with faster_whisper
        segments, info = model.transcribe(
            segment_path,
            language="ar",
            beam_size=5
        )

        # Combine all segment texts
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        transcription = ' '.join(text_parts)

        # Find best match in Quran
        match_idx, match_ratio = find_best_match(transcription, quran_lines)

        if match_idx >= 0:
            matched_text = quran_lines[match_idx].strip()
            print(f"Match: {match_ratio:.2%} (line {match_idx + 1})")
            results.append(matched_text)

            # Show comparison if not perfect match
            if match_ratio < 0.95:
                print(f"  Transcribed: {transcription}")
                print(f"  Matched:     {matched_text}")
        else:
            print("No match found!")
            results.append(transcription)

    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        for line in results:
            f.write(line + "\n")

    print(f"\n✓ Saved {len(results)} transcriptions to {output_file}")

if __name__ == "__main__":
    main()
