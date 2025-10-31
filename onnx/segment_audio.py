#!/usr/bin/env python3
"""
Segment audio file based on silence detection
Target segment length: ~4 seconds
Each segment should be surrounded by at least one silent frame
"""
import numpy as np
import torchaudio
import torch
import os

def detect_silence(audio, sample_rate, threshold_db=-20, min_silence_frames=2):
    """Detect silent regions in audio"""
    # Convert to mono if needed
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # Calculate energy per frame (using hop_length for frame size)
    hop_length = sample_rate // 20  # 20 fps
    frame_energy = []

    for i in range(0, audio.shape[1], hop_length):
        frame = audio[:, i:i+hop_length]
        energy = (frame ** 2).mean().item()
        energy_db = 10 * np.log10(energy + 1e-10)
        frame_energy.append(energy_db)

    # Find silent frames
    silent_frames = [i for i, e in enumerate(frame_energy) if e < threshold_db]

    # Group consecutive silent frames
    silent_regions = []
    if silent_frames:
        start = silent_frames[0]
        prev = silent_frames[0]

        for frame in silent_frames[1:]:
            if frame != prev + 1:
                # End of current silent region
                if prev - start + 1 >= min_silence_frames:
                    silent_regions.append((start, prev))
                start = frame
            prev = frame

        # Add last region
        if prev - start + 1 >= min_silence_frames:
            silent_regions.append((start, prev))

    return silent_regions, hop_length

def segment_audio(audio_path, output_dir, target_duration=4.0, min_duration=2.0, max_duration=6.0):
    """Segment audio based on silence with target duration"""
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)

    print(f"Loaded {audio_path}")
    print(f"Sample rate: {sample_rate}, Duration: {waveform.shape[1] / sample_rate:.2f}s")

    # Detect silence
    silent_regions, hop_length = detect_silence(waveform, sample_rate)

    print(f"\nFound {len(silent_regions)} silent regions")

    # Convert to sample indices
    silent_samples = [(start * hop_length, end * hop_length) for start, end in silent_regions]

    # Create segments based on target duration
    segments = []
    current_start = 0
    target_samples = int(target_duration * sample_rate)
    min_samples = int(min_duration * sample_rate)
    max_samples = int(max_duration * sample_rate)

    for silence_start, silence_end in silent_samples:
        segment_length = silence_start - current_start

        # Check if this creates a good segment
        if segment_length >= min_samples:
            # If close to target, create segment
            if segment_length >= target_samples * 0.7:  # At least 70% of target
                # End segment at silence start
                segments.append((current_start, silence_start))
                current_start = silence_end
            # If too long, split it
            elif segment_length > max_samples:
                # Find best split point
                segments.append((current_start, silence_start))
                current_start = silence_end

    # Add final segment
    if current_start < waveform.shape[1]:
        segments.append((current_start, waveform.shape[1]))

    print(f"\nCreated {len(segments)} segments:")

    # Save segments
    base_name = os.path.splitext(os.path.basename(audio_path))[0]

    for i, (start, end) in enumerate(segments, 1):
        segment_audio = waveform[:, start:end]
        duration = segment_audio.shape[1] / sample_rate

        output_path = os.path.join(output_dir, f"{base_name}-{i:02d}.wav")
        torchaudio.save(output_path, segment_audio, sample_rate)

        print(f"  Segment {i:2d}: {duration:5.2f}s -> {output_path}")

    print(f"\n✓ Saved {len(segments)} segments to {output_dir}/")
    return len(segments)

def main():
    audio_path = "/Users/amraboelela/develop/android/AndroidArabicWhisper/app/src/main/assets/001.wav"
    output_dir = "segments"

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Segment the audio
    num_segments = segment_audio(
        audio_path,
        output_dir,
        target_duration=4.0,
        min_duration=2.0,
        max_duration=6.0
    )

if __name__ == "__main__":
    main()
