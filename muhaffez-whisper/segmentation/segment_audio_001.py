#!/usr/bin/env python3
"""
Split 001.wav specifically - find parameters that give us 8 segments
"""

import numpy as np
import torchaudio
import torch
import os


def detect_silence(audio, sample_rate, threshold_db=-30, min_silence_frames=1):
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


def segment_audio(audio_path, threshold_db=-30, min_silence_frames=1):
    """Segment audio purely based on silence"""
    waveform, sample_rate = torchaudio.load(audio_path)

    print(f"Sample rate: {sample_rate}, Duration: {waveform.shape[1] / sample_rate:.2f}s")
    print(f"Parameters: threshold_db={threshold_db}, min_silence_frames={min_silence_frames}")

    silent_regions, hop_length = detect_silence(waveform, sample_rate, threshold_db, min_silence_frames)
    silent_samples = [(start * hop_length, end * hop_length) for start, end in silent_regions]

    segments = []
    current_start = 0
    for silence_start, silence_end in silent_samples:
        if silence_start > current_start:
            segments.append((current_start, silence_start))
        current_start = silence_end
    if current_start < waveform.shape[1]:
        segments.append((current_start, waveform.shape[1]))

    valid_segments = [s for s in segments if (s[1] - s[0]) / sample_rate >= 0.5]

    print(f"Created {len(valid_segments)} valid segments (>= 0.5s)\n")
    for i, (start, end) in enumerate(valid_segments, 1):
        duration = (end - start) / sample_rate
        print(f"  Segment {i}: {duration:5.2f}s")

    return len(valid_segments)


# Test different parameter combinations
audio_path = os.path.expanduser("~/audio/Quran-A/001.mp3")

print("="*60)
print("TESTING DIFFERENT PARAMETERS TO GET 8 SEGMENTS")
print("="*60)
print()

# Try different combinations
params = [
    (-30, 1),
    (-30, 2),
    (-30, 3),
    (-28, 1),
    (-28, 2),
    (-28, 3),
    (-25, 1),
    (-25, 2),
    (-32, 1),
    (-32, 2),
]

best_params = None
best_diff = float('inf')

for threshold_db, min_silence_frames in params:
    print(f"\n{'='*60}")
    count = segment_audio(audio_path, threshold_db, min_silence_frames)
    diff = abs(count - 8)

    if diff < best_diff:
        best_diff = diff
        best_params = (threshold_db, min_silence_frames, count)

    if count == 8:
        print(f"\n🎯 PERFECT! Found parameters that give exactly 8 segments!")
        print(f"   threshold_db={threshold_db}, min_silence_frames={min_silence_frames}")
        break

if best_params:
    print(f"\n{'='*60}")
    print(f"BEST RESULT:")
    print(f"  threshold_db={best_params[0]}, min_silence_frames={best_params[1]}")
    print(f"  Segments: {best_params[2]} (difference from target: {best_diff})")
    print(f"{'='*60}")
