#!/usr/bin/env python3
"""
Transcribe audio segments using openai/whisper-base model
Usage: python3 transcribe_segments.py <dataset_name> <segment_name>
       python3 transcribe_segments.py Quran-A 002-04
       python3 transcribe_segments.py Quran-A 001
"""
import glob
import os
import sys
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

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

    # Setup paths (script is in segmentation folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Extract audio subdirectory (e.g., "002-04" -> "002", "001" -> "001")
    audio_subdir = segment_prefix.split('-')[0]

    # Determine audio directory based on segment structure
    # If segment_prefix has parts (e.g., "002-04"), look in subdirectory
    if '-' in segment_prefix:
        audio_dir = os.path.join(script_dir, f"../{dataset_name}/audio/raw", audio_subdir, segment_prefix)
    else:
        # For single segments like "001", look directly in prefix folder
        audio_dir = os.path.join(script_dir, f"../{dataset_name}/audio/raw", audio_subdir)

    output_file = os.path.join(script_dir, f"../{dataset_name}/text", f"{segment_prefix}.txt")

    print(f"Script directory: {script_dir}")
    print(f"Audio directory: {audio_dir}")
    print(f"Output file: {output_file}")

    # Use local tarteel model from models directory
    model_path = os.path.join(script_dir, "../../models/custom-whisper-ar-quran")

    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        sys.exit(1)

    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Using Metal GPU (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("⚠️  Using CPU")

    # Load model from local directory
    print(f"\nLoading model from: {model_path}...")
    try:
        processor = WhisperProcessor.from_pretrained(model_path, local_files_only=True)
        model = WhisperForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
        model = model.to(device)
        model.eval()
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
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_file)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample to 16kHz if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            # Prepare input
            audio_array = waveform.squeeze().numpy()
            inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(device)

            # Generate transcription
            with torch.no_grad():
                # Simple generation
                predicted_ids = model.generate(input_features, max_length=225)

            # Decode
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            transcription = transcription.strip()
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
