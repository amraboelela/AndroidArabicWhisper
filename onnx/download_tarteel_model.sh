#!/bin/bash
# Download tarteel-ai/whisper-base-ar-quran model using git

set -e

MODEL_DIR="models/whisper-base-ar-quran-pt"

echo "Installing git-lfs if not already installed..."
git lfs install

echo ""
echo "Downloading tarteel-ai/whisper-base-ar-quran model..."
echo "This may take a while depending on your connection..."
echo ""

# Clone the model repository
git clone https://huggingface.co/tarteel-ai/whisper-base-ar-quran "$MODEL_DIR"

echo ""
echo "✓ Model downloaded successfully to $MODEL_DIR"
echo ""
echo "Files:"
ls -lh "$MODEL_DIR"
