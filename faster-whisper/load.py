import sys
import os
from faster_whisper import WhisperModel

# Set offline mode
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Use existing model from assets
model_dir = "../app/src/main/assets/whisper_ct2/"

# Load the converted model
model = WhisperModel(model_dir, device="cpu")

# Get audio file from command-line parameter or use default
if len(sys.argv) < 2:
    audio_file = "../app/src/main/assets/002-01.wav"
else:
    audio_file = sys.argv[1]

# Transcribe the audio file
segments, info = model.transcribe(audio_file)

print(f"Detected language: {info.language}")
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

