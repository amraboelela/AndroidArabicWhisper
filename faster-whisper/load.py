import sys
import os
from faster_whisper import WhisperModel
from ctranslate2.converters import TransformersConverter

output_path = "../app/src/main/assets/"
# Paths
output_dir = output_path + "whisper_ct2/"

# Remove existing model folder if it exists
if not os.path.exists(output_dir):
#    shutil.rmtree(output_dir)

# Convert Hugging Face model to CTranslate2 format
    converter = TransformersConverter("tarteel-ai/whisper-base-ar-quran")
    #converter.convert(output_dir, quantization="int8")  # or "float16" for GPU
    converter.convert(output_dir, quantization="float16")  # or "float16" for GPU

# Load the converted model
model = WhisperModel(output_dir, device="cpu")  # or device="cuda"

# Get audio file from command-line parameter
if len(sys.argv) < 2:
    audio_file = output_path + "001.wav" #"tests/data/001.wav"
else:
    audio_file = sys.argv[1]

# Transcribe the audio file
segments, info = model.transcribe(audio_file)

print(f"Detected language: {info.language}")
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

