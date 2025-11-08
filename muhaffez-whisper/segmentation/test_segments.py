#!/usr/bin/env python3
import torchaudio
import numpy as np

def detect_silence(audio, sample_rate, threshold_db=-25, min_silence_frames=3):
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    hop_length = sample_rate // 20
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

audio_path = '/Users/amraboelela/audio/Quran-A/001.mp3'
waveform, sample_rate = torchaudio.load(audio_path)
print(f'Sample rate: {sample_rate}')
print(f'Duration: {waveform.shape[1] / sample_rate:.2f}s')

silent_regions, hop_length = detect_silence(waveform, sample_rate, threshold_db=-25, min_silence_frames=3)
silent_samples = [(start * hop_length, end * hop_length) for start, end in silent_regions]

segments = []
current_start = 0
for silence_start, silence_end in silent_samples:
    if silence_start > current_start:
        segments.append((current_start, silence_start))
    current_start = silence_end
if current_start < waveform.shape[1]:
    segments.append((current_start, waveform.shape[1]))

print(f'Created {len(segments)} segments with threshold_db=-25, min_silence_frames=3')
for i, (start, end) in enumerate(segments, 1):
    duration = (end - start) / sample_rate
    if duration >= 0.5:
        print(f'  Segment {i}: {duration:.2f}s')
