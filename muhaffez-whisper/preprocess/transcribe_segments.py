#!/usr/bin/env python3
"""
Transcribe audio segments using faster-whisper with tarteel-ai/whisper-base-ar-quran model
Usage: python3 transcribe_segments.py <dataset_name> <segment_name>
       python3 transcribe_segments.py Quran-A 002-04
       python3 transcribe_segments.py Quran-A 001
"""
import glob
import os
import sys
from faster_whisper import WhisperModel

# Set offline mode to prevent HuggingFace downloads
os.environ['HF_HUB_OFFLINE'] = '1'

def main():
    # Get dataset name and segment prefix from command line
    if len(sys.argv) < 3:
        print("Usage: python3 transcribe_segments.py <dataset_name> <segment_prefix>")
        print("Examples:")
        print("  python3 transcribe_segments.py Quran-A 002-04")
        print("  python3 transcribe_segments.py Quran-A 001")
        sys.exit(1)

    dataset_name = sys.argv[1]
    segment_prefix = sys.argv[2]

    # Setup paths (script is in preprocess folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Extract audio subdirectory (e.g., "002-04" -> "002", "001" -> "001")
    audio_subdir = segment_prefix.split('-')[0]

    # Determine audio directory based on segment structure
    # If segment_prefix has parts (e.g., "002-04"), look in subdirectory
    if '-' in segment_prefix:
        audio_dir = os.path.join(script_dir, f"../datasets/{dataset_name}/audio/raw", audio_subdir, segment_prefix)
    else:
        # For single segments like "001", look directly in prefix folder
        audio_dir = os.path.join(script_dir, f"../datasets/{dataset_name}/audio/raw", audio_subdir)

    output_file = os.path.join(script_dir, f"../datasets/{dataset_name}/text", f"{segment_prefix}.txt")

    print(f"Script directory: {script_dir}")
    print(f"Audio directory: {audio_dir}")
    print(f"Output file: {output_file}")

    # Use converted Tarteel model in CTranslate2 format
    model_size = os.path.join(script_dir, "../models/tarteel_ct2")

    print(f"\nLoading model: {model_size}...")
    try:
        # Load model - faster-whisper will download and cache it automatically
        # Using CPU for compatibility, can change to "cuda" if GPU available
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    # Get all segments matching the prefix
    audio_files = sorted(glob.glob(os.path.join(audio_dir, f"{segment_prefix}-*.wav")))
    print(f"\nFound {len(audio_files)} audio segments for {segment_prefix}")

    if len(audio_files) == 0:
        print("❌ No audio files found!")
        return

    # Transcribe each segment
    transcriptions = []
    print("\nTranscribing segments...")
    print("=" * 60)

    for i, audio_file in enumerate(audio_files, 1):
        filename = os.path.basename(audio_file)
        print(f"[{i}/{len(audio_files)}] {filename}...", end=" ", flush=True)

        try:
            # Transcribe using faster-whisper
            segments, info = model.transcribe(
                audio_file,
                beam_size=5,
                language="ar",
                condition_on_previous_text=False
            )

            # Collect all segments into one transcription
            transcription = " ".join([segment.text for segment in segments]).strip()
            transcriptions.append(transcription)

            print(f"✓ {transcription}")

        except Exception as e:
            print(f"❌ Error: {e}")
            transcriptions.append("")  # Add empty line on error

    print("=" * 60)

    # Save to file
    print(f"\nSaving transcriptions to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for transcription in transcriptions:
            f.write(transcription + "\n")

    print(f"✓ Saved {len(transcriptions)} transcriptions to {output_file}")

    # Show statistics
    non_empty = sum(1 for t in transcriptions if t.strip())
    print(f"\nStatistics:")
    print(f"  Total segments: {len(transcriptions)}")
    print(f"  Transcribed: {non_empty}")
    print(f"  Failed: {len(transcriptions) - non_empty}")

if __name__ == "__main__":
    main()
