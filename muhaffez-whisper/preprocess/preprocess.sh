#!/bin/bash
#
# Preprocess audio: segment, transcribe, normalize, convert to mic quality, and generate mels
# Usage: ./preprocess.sh <dataset_name> <segment_name>
#        ./preprocess.sh Quran-A 002-04
#        ./preprocess.sh Quran-A 001
#
# This script runs six operations in sequence:
# 1. Segment audio file based on silence detection (16kHz raw)
# 2. Transcribe all segments using Whisper
# 3. Normalize the transcribed text
# 4. Fix vocabulary mismatches with closest matches
# 5. Convert current part to mobile mic quality (8kHz)
# 6. Generate augmented audio variations
# 7. Precompute mel features for current part only
#

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
echo "" | tee -a "$LOG_FILE"
echo "[1/7] Segmenting audio file..." | tee -a "$LOG_FILE"
python3 segment_audio.py "$DATASET_NAME" "$SEGMENT_NAME" > "$TEMP_LOG" 2>&1
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    cat "$TEMP_LOG" | tee -a "$LOG_FILE"
    echo "❌ Audio segmentation failed" | tee -a "$LOG_FILE"
    rm "$TEMP_LOG"
    exit 1
fi
# Extract and show summary
tail -10 "$TEMP_LOG" | grep -E "Created|Segment" | tee -a "$LOG_FILE"
cat "$TEMP_LOG" >> "$LOG_FILE"

# Step 2: Transcribe segments
echo "" | tee -a "$LOG_FILE"
echo "[2/7] Transcribing segments..." | tee -a "$LOG_FILE"
python3 transcribe_segments.py "$DATASET_NAME" "$SEGMENT_NAME" > "$TEMP_LOG" 2>&1
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    cat "$TEMP_LOG" | tee -a "$LOG_FILE"
    echo "❌ Transcription failed" | tee -a "$LOG_FILE"
    rm "$TEMP_LOG"
    exit 1
fi
# Extract and show summary
tail -10 "$TEMP_LOG" | grep -E "✓ Saved|Statistics|Transcribed|Total" | tee -a "$LOG_FILE"
cat "$TEMP_LOG" >> "$LOG_FILE"

# Step 3: Normalize text
echo "" | tee -a "$LOG_FILE"
echo "[3/7] Normalizing text..." | tee -a "$LOG_FILE"
python3 normalize_text.py "$DATASET_NAME" "$SEGMENT_NAME" > "$TEMP_LOG" 2>&1
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    cat "$TEMP_LOG" | tee -a "$LOG_FILE"
    echo "❌ Text normalization failed" | tee -a "$LOG_FILE"
    rm "$TEMP_LOG"
    exit 1
fi
# Extract and show summary
tail -5 "$TEMP_LOG" | grep "✓" | tee -a "$LOG_FILE"
cat "$TEMP_LOG" >> "$LOG_FILE"

# Step 4: Fix vocabulary mismatches
echo "" | tee -a "$LOG_FILE"
echo "[4/7] Fixing vocabulary mismatches..." | tee -a "$LOG_FILE"
python3 fix_text.py "$DATASET_NAME" "$SEGMENT_NAME" > "$TEMP_LOG" 2>&1
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    cat "$TEMP_LOG" | tee -a "$LOG_FILE"
    echo "❌ Vocabulary fix failed" | tee -a "$LOG_FILE"
    rm "$TEMP_LOG"
    exit 1
fi
# Extract and show summary
tail -10 "$TEMP_LOG" | grep -E "✓|⚠" | tee -a "$LOG_FILE"
cat "$TEMP_LOG" >> "$LOG_FILE"

# Step 5: Convert raw audio to mic quality
echo "" | tee -a "$LOG_FILE"
echo "[5/7] Converting to mobile mic quality (8kHz)..." | tee -a "$LOG_FILE"
python3 convert_to_mic_quality.py "$DATASET_NAME" "$SEGMENT_NAME" > "$TEMP_LOG" 2>&1
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    cat "$TEMP_LOG" | tee -a "$LOG_FILE"
    echo "❌ Conversion to mic quality failed" | tee -a "$LOG_FILE"
    rm "$TEMP_LOG"
    exit 1
fi
# Extract and show summary
tail -10 "$TEMP_LOG" | grep -E "Converted:|Skipped:|Errors:" | tee -a "$LOG_FILE"
cat "$TEMP_LOG" >> "$LOG_FILE"

# Step 6: Generate augmented audio variations
echo "" | tee -a "$LOG_FILE"
echo "[6/7] Generating augmented audio (pitch/speed variations)..." | tee -a "$LOG_FILE"
python3 generate_augmented.py "$DATASET_NAME" "$SEGMENT_NAME" > "$TEMP_LOG" 2>&1
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    cat "$TEMP_LOG" | tee -a "$LOG_FILE"
    echo "❌ Audio augmentation failed" | tee -a "$LOG_FILE"
    rm "$TEMP_LOG"
    exit 1
fi
# Extract and show summary
tail -10 "$TEMP_LOG" | grep -E "Generated:|Skipped:|Errors:" | tee -a "$LOG_FILE"
cat "$TEMP_LOG" >> "$LOG_FILE"

# Step 7: Precompute mel features from mic quality audio (including augmented)
echo "" | tee -a "$LOG_FILE"
echo "[7/7] Precomputing mel spectrogram features from mic audio..." | tee -a "$LOG_FILE"
python3 generate_mels.py "$DATASET_NAME" "$SEGMENT_NAME" > "$TEMP_LOG" 2>&1
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
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
