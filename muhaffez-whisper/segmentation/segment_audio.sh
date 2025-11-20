#!/bin/bash
#
# Segment, transcribe, and normalize audio
# Usage: ./segment_audio.sh <dataset_name> <segment_name>
#        ./segment_audio.sh Quran-A 002-04
#        ./segment_audio.sh Quran-A 001
#
# This script runs five operations in sequence:
# 1. Segment audio file based on silence detection (16kHz raw)
# 2. Transcribe all segments using Whisper
# 3. Normalize the transcribed text
# 4. Convert current part to mobile mic quality (8kHz)
# 5. Precompute mel features for current part only
#

set -e  # Exit on any error

# Check if dataset name and segment name parameters are provided
if [ $# -lt 2 ]; then
    echo "Usage: ./segment_audio.sh <dataset_name> <segment_name>"
    echo "Examples:"
    echo "  ./segment_audio.sh Quran-A 002-04"
    echo "  ./segment_audio.sh Quran-A 001"
    exit 1
fi

DATASET_NAME=$1
SEGMENT_NAME=$2

echo "============================================================"
echo "PROCESSING DATASET: $DATASET_NAME, SEGMENT: $SEGMENT_NAME"
echo "Started: $(date)"
echo "============================================================"
echo ""

# Step 1: Segment audio
echo "============================================================"
echo "[1/5] Segmenting audio file..."
echo "============================================================"
python3 segment_audio.py "$DATASET_NAME" "$SEGMENT_NAME"
if [ $? -ne 0 ]; then
    echo "❌ Audio segmentation failed"
    exit 1
fi
echo ""

# Step 2: Transcribe segments
echo "============================================================"
echo "[2/5] Transcribing segments..."
echo "============================================================"
python3 transcribe_segments.py "$DATASET_NAME" "$SEGMENT_NAME"
if [ $? -ne 0 ]; then
    echo "❌ Transcription failed"
    exit 1
fi
echo ""

# Step 3: Normalize text
echo "============================================================"
echo "[3/5] Normalizing transcribed text..."
echo "============================================================"
python3 normalize_text.py "$DATASET_NAME" "$SEGMENT_NAME"
if [ $? -ne 0 ]; then
    echo "❌ Text normalization failed"
    exit 1
fi
echo ""

# Step 4: Convert raw audio to mic quality
echo "============================================================"
echo "[4/5] Converting to mobile mic quality (8kHz)..."
echo "============================================================"
python3 convert_to_mic_quality.py "$DATASET_NAME" "$SEGMENT_NAME"
if [ $? -ne 0 ]; then
    echo "❌ Conversion to mic quality failed"
    exit 1
fi
echo ""

# Step 5: Precompute mel features from mic quality audio
echo "============================================================"
echo "[5/5] Precomputing mel spectrogram features from mic audio..."
echo "============================================================"
python3 generate_mels.py "$DATASET_NAME" "$SEGMENT_NAME"
if [ $? -ne 0 ]; then
    echo "❌ Mel feature precomputation failed"
    exit 1
fi
echo ""

echo "============================================================"
echo "✓ ALL STEPS COMPLETED SUCCESSFULLY"
echo "Ended: $(date)"
echo "============================================================"
echo ""
echo "Output files:"
# Check if segment name has parts (e.g., "002-04")
if [[ "$SEGMENT_NAME" == *-* ]]; then
    SEGMENT_PREFIX=$(echo $SEGMENT_NAME | cut -d'-' -f1)
    echo "  Audio segments (raw):  ../$DATASET_NAME/audio/raw/$SEGMENT_PREFIX/$SEGMENT_NAME/$SEGMENT_NAME-*.wav"
    echo "  Audio segments (mic):  ../$DATASET_NAME/audio/mic/$SEGMENT_PREFIX/$SEGMENT_NAME/$SEGMENT_NAME-*.wav"
else
    echo "  Audio segments (raw):  ../$DATASET_NAME/audio/raw/$SEGMENT_NAME/$SEGMENT_NAME-*.wav"
    echo "  Audio segments (mic):  ../$DATASET_NAME/audio/mic/$SEGMENT_NAME/$SEGMENT_NAME-*.wav"
fi
echo "  Transcription:         ../$DATASET_NAME/text/$SEGMENT_NAME.txt (normalized)"
echo "  Mel features:          ../$DATASET_NAME/mels/ (precomputed)"
