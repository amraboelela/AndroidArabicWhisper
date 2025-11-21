#!/usr/bin/env python3
"""Test transcription of a single audio file"""
from transformers import pipeline

audio_file = "/Users/amraboelela/develop/android/AndroidArabicWhisper/muhaffez-whisper/datasets/Quran-A/audio/raw/001/001-03.wav"

print(f"Loading whisper-large-v3...")
pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", device=-1)

print(f"\nTranscribing {audio_file}...")
result = pipe(audio_file, generate_kwargs={"language": "arabic", "task": "transcribe"})

print(f"\nResult: {result['text']}")
