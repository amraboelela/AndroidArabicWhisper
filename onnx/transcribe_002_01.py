#!/usr/bin/env python3
"""
Transcribe 002-01 audio segments using faster_whisper
"""
from faster_whisper import WhisperModel
import glob
import os

def main():
    """Transcribe all 002-01 segments using faster_whisper"""

    segments_dir = "segments"
    output_path = "002-01.txt"

    # Load faster_whisper model (offline mode)
    print("Loading faster_whisper model...")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    print("Model loaded successfully!")

    # Get all 002-01 segment files
    segment_files = sorted(glob.glob(os.path.join(segments_dir, "002-01-*.wav")))
    print(f"\nFound {len(segment_files)} segments")

    # Transcribe each segment
    transcriptions = []

    for segment_file in segment_files:
        segment_name = os.path.basename(segment_file)

        # Transcribe with faster_whisper
        segments, info = model.transcribe(segment_file, language="ar", beam_size=5)

        # Collect all text from segments
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        transcription = ' '.join(text_parts)
        transcriptions.append(transcription)

        print(f"{segment_name}: {transcription}")

    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        for line in transcriptions:
            f.write(line + "\n")

    print(f"\n✓ Saved {len(transcriptions)} transcriptions to {output_path}")


if __name__ == "__main__":
    main()
