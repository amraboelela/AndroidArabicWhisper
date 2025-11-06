#!/usr/bin/env python3
"""
Transcribe all 002-04 audio segments using tarteel-ai/whisper-base-ar-quran model
and save the results to 002-04.txt
Uses transformers library with PyTorch model
"""
import glob
import os
import sys
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def main():
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(script_dir, "datasets/base/audio/002")
    output_file = os.path.join(script_dir, "datasets/base/text/002-04.txt")

    print(f"Script directory: {script_dir}")
    print(f"Audio directory: {audio_dir}")
    print(f"Output file: {output_file}")

    # Use HuggingFace cached model
    model_name = "tarteel-ai/whisper-base-ar-quran"

    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Using Metal GPU (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("⚠️  Using CPU")

    # Load model
    print(f"\nLoading model from HuggingFace cache: {model_name}...")
    try:
        processor = WhisperProcessor.from_pretrained(model_name, local_files_only=True)
        model = WhisperForConditionalGeneration.from_pretrained(model_name, local_files_only=True)
        model = model.to(device)
        model.eval()
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("\nThe model needs to be downloaded first.")
        sys.exit(1)

    # Get all 002-04 segments
    audio_files = sorted(glob.glob(os.path.join(audio_dir, "002-04-*.wav")))
    print(f"\nFound {len(audio_files)} audio segments for 002-04")

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
                # Simple generation - the model is already Arabic-only
                predicted_ids = model.generate(
                    input_features,
                    max_length=225
                )

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
