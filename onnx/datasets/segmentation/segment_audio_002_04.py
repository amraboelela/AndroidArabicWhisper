#!/usr/bin/env python3
"""
Split 002-04.mp3 audio file based on silence detection (-30dB threshold)
Output segments named like 002-04-01.wav, 002-04-02.wav, ...
"""

import numpy as np
import torchaudio
import torch
import os


def detect_silence(audio, sample_rate, threshold_db=-30, min_silence_frames=10):
    """Detect silent regions in audio"""
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)  # convert to mono

    hop_length = sample_rate // 20  # 20 frames per second
    frame_energy = []

    for i in range(0, audio.shape[1], hop_length):
        frame = audio[:, i:i + hop_length]
        energy = (frame ** 2).mean().item()
        energy_db = 10 * np.log10(energy + 1e-10)
        frame_energy.append(energy_db)

    silent_frames = [i for i, e in enumerate(frame_energy) if e < threshold_db]

    silent_regions = []
    if silent_frames:
        start = silent_frames[0]
        prev = silent_frames[0]
        for frame in silent_frames[1:]:
            if frame != prev + 1:
                if prev - start + 1 >= min_silence_frames:
                    silent_regions.append((start, prev))
                start = frame
            prev = frame
        if prev - start + 1 >= min_silence_frames:
            silent_regions.append((start, prev))

    return silent_regions, hop_length


def segment_audio_simple(audio_path, output_dir):
    """Segment audio purely based on silence - no duration constraints"""
    waveform, sample_rate = torchaudio.load(audio_path)

    print(f"Loaded {audio_path}")
    print(f"Sample rate: {sample_rate}, Duration: {waveform.shape[1] / sample_rate:.2f}s")

    silent_regions, hop_length = detect_silence(waveform, sample_rate)
    silent_samples = [(start * hop_length, end * hop_length) for start, end in silent_regions]

    segments = []
    current_start = 0
    for silence_start, silence_end in silent_samples:
        if silence_start > current_start:
            segments.append((current_start, silence_start))
        current_start = silence_end
    if current_start < waveform.shape[1]:
        segments.append((current_start, waveform.shape[1]))

    print(f"\nCreated {len(segments)} segments")

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    saved_count = 0

    os.makedirs(output_dir, exist_ok=True)

    for i, (start, end) in enumerate(segments, 1):
        segment_audio = waveform[:, start:end]
        duration = segment_audio.shape[1] / sample_rate

        if duration >= 0.5:  # skip very short noise
            saved_count += 1
            output_path = os.path.join(
                output_dir,
                f"{base_name}-{saved_count:02d}.wav"  # 2-digit format
            )
            torchaudio.save(output_path, segment_audio, sample_rate)
            print(f"  Segment {saved_count:3d}: {duration:5.2f}s -> {output_path}")

    print(f"\n✓ Saved {saved_count} segments to {output_dir}/")
    return saved_count


def main():
    audio_path = os.path.expanduser("~/audio/Quran-A/002-04.mp3")
    output_dir = "../base/audio"

    os.makedirs(output_dir, exist_ok=True)
    segment_audio_simple(audio_path, output_dir)


if __name__ == "__main__":
    main()
