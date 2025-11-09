#\!/usr/bin/env python3
import numpy as np
import torchaudio
import sys

def detect_silence(audio, sample_rate, threshold_db=-30, min_silence_frames=2):
    """Detect silent regions in audio"""
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    hop_length = sample_rate // 20
    frame_energy = []

    for i in range(0, audio.shape[1], hop_length):
        frame = audio[:, i:i + hop_length]
        energy = (frame ** 2).mean().item()
        energy_db = 10 * np.log10(energy + 1e-10)
        frame_energy.append(energy_db)

    print(f"Total frames: {len(frame_energy)}")
    
    silent_frames = [i for i, e in enumerate(frame_energy) if e < threshold_db]
    print(f"Silent frames: {len(silent_frames)}")

    silent_regions = []
    if silent_frames:
        start = silent_frames[0]
        prev = silent_frames[0]
        for frame in silent_frames[1:]:
            if frame \!= prev + 1:
                if prev - start + 1 >= min_silence_frames:
                    silent_regions.append((start, prev))
                    print(f"  Silent region: frames {start}-{prev} ({prev - start + 1} frames)")
                start = frame
            prev = frame
        if prev - start + 1 >= min_silence_frames:
            silent_regions.append((start, prev))
            print(f"  Silent region: frames {start}-{prev} ({prev - start + 1} frames)")

    print(f"Silent regions: {len(silent_regions)}")
    return silent_regions, hop_length

audio_path = "/Users/amraboelela/develop/android/AndroidArabicWhisper/app/src/main/assets/001.wav"
waveform, sample_rate = torchaudio.load(audio_path)
print(f"Loaded {waveform.shape[1]} samples at {sample_rate} Hz")

silent_regions, hop_length = detect_silence(waveform, sample_rate)
