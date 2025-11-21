#!/bin/bash
#
# Preprocess audio: segment, transcribe, normalize, convert to mic quality, and generate mels
# Usage: ./preprocess.sh <dataset_name> <segment_name>
#        ./preprocess.sh Quran-A 002-04
#        ./preprocess.sh Quran-A 001
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
    echo "Usage: ./preprocess.sh <dataset_name> <segment_name>"
    echo "Examples:"
    echo "  ./preprocess.sh Quran-A 002-04"
    echo "  ./preprocess.sh Quran-A 001"
    exit 1
fi

DATASET_NAME=$1
SEGMENT_NAME=$2

# Setup logging
LOG_FILE="log.txt"
BACKUP_LOG="log_backup.txt"
TEMP_LOG=$(mktemp)

# Backup existing log if it exists
if [ -f "$LOG_FILE" ]; then
    cp "$LOG_FILE" "$BACKUP_LOG"
    echo "Backed up previous log to $BACKUP_LOG"
fi

# Clear the log file to start fresh
> "$LOG_FILE"

echo "============================================================"
echo "PROCESSING DATASET: $DATASET_NAME, Surah (part): $SEGMENT_NAME"
echo "Started: $(date)"
echo "============================================================" | tee -a "$LOG_FILE"
echo ""

# Step 1: Segment audio
echo "[1/6] Segmenting audio file..." | tee -a "$LOG_FILE"
python3 segment_audio.py "$DATASET_NAME" "$SEGMENT_NAME" > "$TEMP_LOG" 2>&1
if [ $? -ne 0 ]; then
    cat "$TEMP_LOG" | tee -a "$LOG_FILE"
    echo "❌ Audio segmentation failed" | tee -a "$LOG_FILE"
    rm "$TEMP_LOG"
    exit 1
fi
# Extract and show summary
tail -10 "$TEMP_LOG" | grep -E "Created|Segment" | tee -a "$LOG_FILE"
cat "$TEMP_LOG" >> "$LOG_FILE"
echo ""

# Step 2: Transcribe segments
echo "[2/6] Transcribing segments..." | tee -a "$LOG_FILE"
python3 transcribe_segments.py "$DATASET_NAME" "$SEGMENT_NAME" > "$TEMP_LOG" 2>&1
if [ $? -ne 0 ]; then
    cat "$TEMP_LOG" | tee -a "$LOG_FILE"
    echo "❌ Transcription failed" | tee -a "$LOG_FILE"
    rm "$TEMP_LOG"
    exit 1
fi
# Extract and show summary
tail -10 "$TEMP_LOG" | grep -E "✓ Saved|Statistics|Transcribed|Total" | tee -a "$LOG_FILE"
cat "$TEMP_LOG" >> "$LOG_FILE"
echo ""

# Step 3: Fix with Quran database and normalize text
echo "[3/6] Fixing and normalizing text..." | tee -a "$LOG_FILE"
python3 fix_text.py "$DATASET_NAME" "$SEGMENT_NAME" > "$TEMP_LOG" 2>&1
if [ $? -ne 0 ]; then
    cat "$TEMP_LOG" | tee -a "$LOG_FILE"
    echo "❌ Text fix/normalization failed" | tee -a "$LOG_FILE"
    rm "$TEMP_LOG"
    exit 1
fi
# Extract and show summary
tail -5 "$TEMP_LOG" | grep "✓" | tee -a "$LOG_FILE"
cat "$TEMP_LOG" >> "$LOG_FILE"
echo ""

# Step 4: Convert raw audio to mic quality
echo "[4/6] Converting to mobile mic quality (8kHz)..." | tee -a "$LOG_FILE"
python3 convert_to_mic_quality.py "$DATASET_NAME" "$SEGMENT_NAME" > "$TEMP_LOG" 2>&1
if [ $? -ne 0 ]; then
    cat "$TEMP_LOG" | tee -a "$LOG_FILE"
    echo "❌ Conversion to mic quality failed" | tee -a "$LOG_FILE"
    rm "$TEMP_LOG"
    exit 1
fi
# Extract and show summary
tail -10 "$TEMP_LOG" | grep -E "Converted:|Skipped:|Errors:" | tee -a "$LOG_FILE"
cat "$TEMP_LOG" >> "$LOG_FILE"
echo ""

# Step 5: Generate augmented audio variations
echo "[5/6] Generating augmented audio (pitch/speed variations)..." | tee -a "$LOG_FILE"
python3 generate_augmented.py "$DATASET_NAME" "$SEGMENT_NAME" > "$TEMP_LOG" 2>&1
if [ $? -ne 0 ]; then
    cat "$TEMP_LOG" | tee -a "$LOG_FILE"
    echo "❌ Audio augmentation failed" | tee -a "$LOG_FILE"
    rm "$TEMP_LOG"
    exit 1
fi
# Extract and show summary
tail -10 "$TEMP_LOG" | grep -E "Generated:|Skipped:|Errors:" | tee -a "$LOG_FILE"
cat "$TEMP_LOG" >> "$LOG_FILE"
echo ""

# Step 6: Precompute mel features from mic quality audio (including augmented)
echo "[6/6] Precomputing mel spectrogram features from mic audio..." | tee -a "$LOG_FILE"
python3 generate_mels.py "$DATASET_NAME" "$SEGMENT_NAME" > "$TEMP_LOG" 2>&1
if [ $? -ne 0 ]; then
    cat "$TEMP_LOG" | tee -a "$LOG_FILE"
    echo "❌ Mel feature precomputation failed" | tee -a "$LOG_FILE"
    rm "$TEMP_LOG"
    exit 1
fi
# Extract and show summary
tail -10 "$TEMP_LOG" | grep -E "Generated:|Skipped:|Total:" | tee -a "$LOG_FILE"
cat "$TEMP_LOG" >> "$LOG_FILE"
rm "$TEMP_LOG"
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
