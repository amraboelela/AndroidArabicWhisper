#!/usr/bin/env python3
"""
Transcribe audio segments using Whisper
"""
import whisper
import glob
import os

def main():
    """Transcribe all segments in segments/ directory using Whisper"""

    datasets_dir = "datasets/base"
    output_path = "001.txt"

    # Load Whisper model
    print("Loading Whisper model...")
    model = whisper.load_model("base")
    print("Model loaded successfully!")

    # Get all segment files
    segment_files = sorted(glob.glob(os.path.join(datasets_dir, "001-*.wav")))
    print(f"\nFound {len(segment_files)} segments")

    # Transcribe each segment
    transcriptions = []

    for segment_file in segment_files:
        segment_name = os.path.basename(segment_file)

        # Transcribe with Whisper
        result = model.transcribe(segment_file, language="ar")
        transcription = result["text"].strip()
        transcriptions.append(transcription)

        print(f"{segment_name}: {transcription}")

    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        for line in transcriptions:
            f.write(line + "\n")

    print(f"\n✓ Saved transcriptions to {output_path}")
    print(f"\nFull transcription:")
    print(' '.join(transcriptions))


if __name__ == "__main__":
    main()
