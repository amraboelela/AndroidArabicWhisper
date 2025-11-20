#!/bin/bash
#
# Train on ALL segments in a dataset using full, curriculum, and augmented methods
# Usage:
#   ./train_all.sh <dataset_name>
#
# Example:
#   ./train_all.sh Quran-A
#

# Main training log
TRAIN_LOG="log_train_all.txt"
TRAIN_LOG_BACKUP="log_train_all_backup.txt"

# If this is the initial call (not recursive), set up logging
if [ -z "$TRAIN_LOGGING_ACTIVE" ]; then
    export TRAIN_LOGGING_ACTIVE=1

    # Backup existing log if it exists
    if [ -f "$TRAIN_LOG" ]; then
        cp "$TRAIN_LOG" "$TRAIN_LOG_BACKUP"
        echo "✓ Training log backup created: $TRAIN_LOG_BACKUP"
    fi

    # Clear the log file and write device info
    > "$TRAIN_LOG"
    {
        echo "============================================================"
        if command -v python3 &> /dev/null; then
            python3 -c "import torch; print('🚀 Using Metal GPU (Apple Silicon)' if torch.backends.mps.is_available() else ('🚀 Using CUDA GPU' if torch.cuda.is_available() else '⚠️  Using CPU (slower)')); print(f'Device: {\"mps\" if torch.backends.mps.is_available() else (\"cuda\" if torch.cuda.is_available() else \"cpu\")}')" 2>/dev/null || echo "Device: unknown"
        else
            echo "Device: unknown"
        fi
        echo "============================================================"
        echo ""
    } >> "$TRAIN_LOG"

    # Re-run this script with output redirected to log and console
    "$0" "$@" 2>&1 | tee -a "$TRAIN_LOG"
    exit $?
fi

# Get parameters
DATASET=${1}

# If no dataset parameter, find all datasets
if [ -z "$DATASET" ]; then
    DATASETS=($(ls -d ../datasets/*/ 2>/dev/null | xargs -n 1 basename))
    if [ ${#DATASETS[@]} -eq 0 ]; then
        echo "❌ No datasets found in ../datasets/"
        exit 1
    fi

    echo "Found ${#DATASETS[@]} dataset(s): ${DATASETS[@]}"
    echo ""

    # Recursively call this script for each dataset
    for ds in "${DATASETS[@]}"; do
        echo "════════════════════════════════════════════════════════════"
        echo "Processing dataset: $ds"
        echo "════════════════════════════════════════════════════════════"
        "$0" "$ds" || exit 1
        echo ""
    done

    echo "✓ All datasets processed successfully!"
    exit 0
fi

# Check if dataset directory exists
if [ ! -d "../datasets/${DATASET}" ]; then
    echo "❌ Dataset directory not found: ../datasets/${DATASET}"
    exit 1
fi

# Create model backup at the very beginning
MODEL_PATH="../models/muhaffez_whisper.pt"
if [ -f "$MODEL_PATH" ]; then
    BACKUP_PATH="../models/muhaffez_whisper_backup.pt"
    cp "$MODEL_PATH" "$BACKUP_PATH"
    echo "✓ Model backup created: $BACKUP_PATH"
    echo ""
fi

# Track overall start time
OVERALL_START_TIME=$(date +%s)

# Track results
PASSED=0
FAILED=0

echo "════════════════════════════════════════════════════════════"
echo "TRAINING ALL SEGMENTS - DATASET: $DATASET"
echo "════════════════════════════════════════════════════════════"
echo ""

# Run train_all_curriculum.py first
SUITE_START=$(date +%s)
if python3 -u train_all_curriculum.py "$DATASET"; then
    SUITE_END=$(date +%s)
    ELAPSED=$((SUITE_END - SUITE_START))

    if [ $ELAPSED -lt 60 ]; then
        TIME_STR="${ELAPSED}s"
    else
        MINUTES=$(echo "scale=0; m = ($ELAPSED + 30) / 60; if (m < 1) 1 else m" | bc)
        TIME_STR="${MINUTES}m"
    fi

    echo "✓ Curriculum ($TIME_STR)"
    PASSED=$((PASSED + 1))
else
    SUITE_END=$(date +%s)
    ELAPSED=$((SUITE_END - SUITE_START))

    if [ $ELAPSED -lt 60 ]; then
        TIME_STR="${ELAPSED}s"
    else
        MINUTES=$(echo "scale=0; m = ($ELAPSED + 30) / 60; if (m < 1) 1 else m" | bc)
        TIME_STR="${MINUTES}m"
    fi

    echo "✗ Curriculum ($TIME_STR) FAILED"
    FAILED=$((FAILED + 1))
fi

echo ""

# Run train_all_full.py
SUITE_START=$(date +%s)
if python3 -u train_all_full.py "$DATASET"; then
    SUITE_END=$(date +%s)
    ELAPSED=$((SUITE_END - SUITE_START))

    if [ $ELAPSED -lt 60 ]; then
        TIME_STR="${ELAPSED}s"
    else
        MINUTES=$(echo "scale=0; m = ($ELAPSED + 30) / 60; if (m < 1) 1 else m" | bc)
        TIME_STR="${MINUTES}m"
    fi

    echo "✓ Full ($TIME_STR)"
    PASSED=$((PASSED + 1))
else
    SUITE_END=$(date +%s)
    ELAPSED=$((SUITE_END - SUITE_START))

    if [ $ELAPSED -lt 60 ]; then
        TIME_STR="${ELAPSED}s"
    else
        MINUTES=$(echo "scale=0; m = ($ELAPSED + 30) / 60; if (m < 1) 1 else m" | bc)
        TIME_STR="${MINUTES}m"
    fi

    echo "✗ Full ($TIME_STR) FAILED"
    FAILED=$((FAILED + 1))
fi

echo ""

# Run train_all_full_augmented.py
SUITE_START=$(date +%s)
if python3 -u train_all_full_augmented.py; then
    SUITE_END=$(date +%s)
    ELAPSED=$((SUITE_END - SUITE_START))

    if [ $ELAPSED -lt 60 ]; then
        TIME_STR="${ELAPSED}s"
    else
        MINUTES=$(echo "scale=0; m = ($ELAPSED + 30) / 60; if (m < 1) 1 else m" | bc)
        TIME_STR="${MINUTES}m"
    fi

    echo "✓ Augmented ($TIME_STR)"
    PASSED=$((PASSED + 1))
else
    SUITE_END=$(date +%s)
    ELAPSED=$((SUITE_END - SUITE_START))

    if [ $ELAPSED -lt 60 ]; then
        TIME_STR="${ELAPSED}s"
    else
        MINUTES=$(echo "scale=0; m = ($ELAPSED + 30) / 60; if (m < 1) 1 else m" | bc)
        TIME_STR="${MINUTES}m"
    fi

    echo "✗ Augmented ($TIME_STR) FAILED"
    FAILED=$((FAILED + 1))
fi

echo ""

# Calculate total time
OVERALL_END_TIME=$(date +%s)
TOTAL_ELAPSED=$((OVERALL_END_TIME - OVERALL_START_TIME))

if [ $TOTAL_ELAPSED -lt 60 ]; then
    TOTAL_TIME_STR="${TOTAL_ELAPSED}s"
elif [ $TOTAL_ELAPSED -ge 3600 ]; then
    HOURS=$((TOTAL_ELAPSED / 3600))
    REMAINING_MINUTES=$(((TOTAL_ELAPSED % 3600) / 60))
    TOTAL_TIME_STR="${HOURS}h ${REMAINING_MINUTES}m"
else
    TOTAL_MINUTES=$(echo "scale=0; m = ($TOTAL_ELAPSED + 30) / 60; if (m < 1) 1 else m" | bc)
    TOTAL_TIME_STR="${TOTAL_MINUTES}m"
fi

# Summary
echo ""
echo "Training Summary:"
echo "  Dataset: $DATASET"
echo "  Completed suites: $PASSED"
if [ $FAILED -gt 0 ]; then
    echo "  Failed: $FAILED suites"
fi
echo "  Total time: ${TOTAL_TIME_STR}"

# Exit with error if any training failed
if [ $FAILED -gt 0 ]; then
    echo ""
    echo "⚠️  Some training suites failed."
    exit 1
else
    echo ""
    echo "✓ All training suites completed successfully!"
    exit 0
fi
