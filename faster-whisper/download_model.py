#!/usr/bin/env python3
"""
Download and convert Whisper model to CTranslate2 format
"""
import os
import shutil
from ctranslate2.converters import TransformersConverter

# Paths
output_dir = "../app/src/main/assets/whisper_ct2/"

# Remove existing model folder if it exists (keeping tokenizer.json and vocabulary.json)
if os.path.exists(output_dir):
    # Backup tokenizer and vocabulary files
    backup_files = {}
    for filename in ["tokenizer.json", "vocabulary.json", "config.json"]:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                backup_files[filename] = f.read()

    # Remove the directory
    shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Restore backup files
    for filename, content in backup_files.items():
        with open(os.path.join(output_dir, filename), 'wb') as f:
            f.write(content)
else:
    os.makedirs(output_dir)

print("Downloading and converting Whisper model to CTranslate2 format...")
print(f"Output directory: {output_dir}")

# Convert Hugging Face model to CTranslate2 format
# Using base model for Arabic
converter = TransformersConverter("openai/whisper-base")
converter.convert(output_dir, quantization="int8", force=True)

print(f"\nModel successfully converted and saved to: {output_dir}")
print("Files created:")
for filename in os.listdir(output_dir):
    filepath = os.path.join(output_dir, filename)
    size = os.path.getsize(filepath) / (1024 * 1024)  # Convert to MB
    print(f"  - {filename} ({size:.2f} MB)")
