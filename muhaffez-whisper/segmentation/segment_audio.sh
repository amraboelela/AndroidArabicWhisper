#!/bin/bash
#
# Segment, transcribe, and normalize audio
# Usage: ./segment_audio.sh <dataset_name> <segment_name>
#        ./segment_audio.sh Quran-A 002-04
#        ./segment_audio.sh Quran-A 001
#
# This script runs three operations in sequence:
# 1. Segment audio file based on silence detection
# 2. Transcribe all segments using Whisper
# 3. Normalize the transcribed text
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
echo "[1/3] Segmenting audio file..."
echo "============================================================"
python3 segment_audio.py "$DATASET_NAME" "$SEGMENT_NAME"
if [ $? -ne 0 ]; then
    echo "❌ Audio segmentation failed"
    exit 1
fi
echo ""

# Step 2: Transcribe segments
echo "============================================================"
echo "[2/3] Transcribing segments..."
echo "============================================================"
python3 transcribe_segments.py "$DATASET_NAME" "$SEGMENT_NAME"
if [ $? -ne 0 ]; then
    echo "❌ Transcription failed"
    exit 1
fi
echo ""

# Step 3: Normalize text
echo "============================================================"
echo "[3/3] Normalizing transcribed text..."
echo "============================================================"
python3 normalize_text.py "$DATASET_NAME" "$SEGMENT_NAME"
if [ $? -ne 0 ]; then
    echo "❌ Text normalization failed"
    exit 1
fi
echo ""

echo "============================================================"
echo "✓ ALL STEPS COMPLETED SUCCESSFULLY"
echo "Ended: $(date)"
echo "============================================================"
echo ""
echo "Output files:"
echo "  Audio segments: ../$DATASET_NAME/audio/$(echo $SEGMENT_NAME | cut -d'-' -f1)/$SEGMENT_NAME-*.wav"
echo "  Transcription:  ../$DATASET_NAME/text/$SEGMENT_NAME.txt (normalized)"
