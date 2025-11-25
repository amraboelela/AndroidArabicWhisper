#!/bin/bash
#
# Unified training script - handles both per-part and whole-dataset training
# Usage:
#   ./train.sh all                               # Train ALL datasets (full suite with augmentation)
#   ./train.sh <dataset_name> all                # Train entire dataset (full suite with augmentation)
#   ./train.sh <dataset_name> <surah>            # Train all parts of specific surah (per-part)
#   ./train.sh <dataset_name> <surah_part>       # Train specific surah part (per-part)
#
# Examples:
#   ./train.sh all                               # Train everything with augmentation
#   ./train.sh Quran-A all                       # Train entire Quran-A with augmentation
#   ./train.sh Quran-A 002                       # Train all parts of surah 002 (per-part)
#   ./train.sh Quran-A 002-04                    # Train only part 002-04 (per-part)
#

# Main training log
TRAIN_LOG="log.txt"
TRAIN_LOG_BACKUP="log_backup.txt"

# If this is the initial call (not recursive), set up logging
if [ -z "$TRAIN_LOGGING_ACTIVE" ]; then
    export TRAIN_LOGGING_ACTIVE=1

    # Backup existing log if it exists
    if [ -f "$TRAIN_LOG" ]; then
        cp "$TRAIN_LOG" "$TRAIN_LOG_BACKUP"
        echo "✓ Training log backup created: $TRAIN_LOG_BACKUP"
    fi

    # Clear the log file
    > "$TRAIN_LOG"

    # Re-run this script with output redirected to log and console
    "$0" "$@" 2>&1 | tee "$TRAIN_LOG"
    exit $?
fi

# Get parameters
DATASET=${1}
SURAH_OR_PART=${2}

# Check if required parameter is provided
if [ -z "$DATASET" ]; then
    # List available datasets (must have audio directory)
    DATASETS=()
    for dir in ../datasets/*/; do
        dirname=$(basename "$dir")
        if [ -d "$dir/audio" ] || [ -d "$dir/audio/mic" ]; then
            DATASETS+=("$dirname")
        fi
    done

    echo "Usage: ./train.sh <dataset_name|all> [surah_part|all]"
    echo ""
    echo "Examples:"
    echo "  ./train.sh all                    # Train all datasets"
    echo "  ./train.sh Quran-A all            # Train entire Quran-A dataset"
    echo "  ./train.sh Quran-A 002            # Train all parts of surah 002"
    echo "  ./train.sh Quran-A 002-04         # Train specific part 002-04"

    if [ ${#DATASETS[@]} -gt 0 ]; then
        echo ""
        echo "Available datasets:"
        for DS in "${DATASETS[@]}"; do
            echo "  - $DS"
        done
    fi

    exit 1
fi

# If dataset parameter is "all", find all datasets and run whole-dataset training
if [ "$DATASET" = "all" ]; then
    DATASETS=()
    for dir in ../datasets/*/; do
        dirname=$(basename "$dir")
        if [ -d "$dir/audio" ] || [ -d "$dir/audio/mic" ]; then
            DATASETS+=("$dirname")
        fi
    done

    if [ ${#DATASETS[@]} -eq 0 ]; then
        echo "❌ Error: No datasets found in ../datasets/"
        exit 1
    fi
    echo "Found ${#DATASETS[@]} dataset(s): ${DATASETS[@]}"
    echo ""

    # Train each dataset with full suite
    for DS in "${DATASETS[@]}"; do
        echo "════════════════════════════════════════════════════════════"
        echo "Processing dataset: $DS"
        echo "════════════════════════════════════════════════════════════"
        # Recursively call this script with the dataset name and "all"
        "$0" "$DS" "all"

        if [ $? -ne 0 ]; then
            echo "❌ Error: Training failed for dataset $DS"
            exit 1
        fi
        echo ""
    done

    echo ""
    echo "✓ All datasets trained successfully!"
    exit 0
fi

# If second parameter is not provided or is "all", run whole-dataset training with augmentation
if [ -z "$SURAH_OR_PART" ] || [ "$SURAH_OR_PART" = "all" ]; then
    # Check if dataset directory exists
    if [ ! -d "../datasets/${DATASET}" ]; then
        echo "❌ Dataset directory not found: ../datasets/${DATASET}"
        exit 1
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

    # Run train_curriculum.py (whole-dataset mode)
    SUITE_START=$(date +%s)
    if python3 -u train_curriculum.py "$DATASET" "all"; then
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

    # Run train_full.py (whole-dataset mode)
    SUITE_START=$(date +%s)
    if python3 -u train_full.py "$DATASET" "all"; then
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

    # Run train_augmented.py (whole-dataset mode with pitch and speed augmentation)
    SUITE_START=$(date +%s)
    if python3 -u train_augmented.py "$DATASET" "all"; then
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
fi

# If second parameter exists, run per-part training
# Find parts to train based on second parameter
if [[ "$SURAH_OR_PART" =~ ^[0-9]{3}$ ]]; then
    # Parameter looks like a surah number (e.g., "002"), find all parts for that surah
    SURAH_PARTS=($(ls ../datasets/${DATASET}/text/${SURAH_OR_PART}*.txt 2>/dev/null | xargs -n 1 basename | sed 's/.txt//'))
    if [ ${#SURAH_PARTS[@]} -eq 0 ]; then
        echo "❌ Error: No text files found for surah ${SURAH_OR_PART} in ../datasets/${DATASET}/text/"
        exit 1
    fi
    echo "Found ${#SURAH_PARTS[@]} parts for surah ${SURAH_OR_PART}"
# Otherwise, treat as specific surah part
else
    SURAH_PARTS=("$SURAH_OR_PART")
fi

# Track results
TOTAL=0
PASSED=0
FAILED=0
START_TIME=$(date +%s)
declare -a SUITE_ACCURACIES  # Array to store accuracies for each suite

# Function to run a training suite for a specific surah part
run_training_suite() {
    local script=$1
    local suite_name=$2
    local surah_part=$3
    local log_file=$4
    TOTAL=$((TOTAL + 1))

    local suite_start=$(date +%s)

    if python3 -u "$script" "$DATASET" "$surah_part" >> "$log_file" 2>&1; then
        local suite_end=$(date +%s)
        local elapsed=$((suite_end - suite_start))

        # Format time: show seconds if < 60s, otherwise minutes
        if [ $elapsed -lt 60 ]; then
            local time_str="${elapsed}s"
        else
            # Round to nearest minute with minimum of 1
            local minutes=$(echo "scale=0; m = ($elapsed + 30) / 60; if (m < 1) 1 else m" | bc)
            local time_str="${minutes}m"
        fi

        # Extract accuracy from log file
        local accuracy=$(grep "FINAL_ACCURACY:" "$log_file" | tail -1 | sed 's/.*FINAL_ACCURACY: //')

        if [ -n "$accuracy" ]; then
            echo "✓ $suite_name ($time_str) - Accuracy: $accuracy"
            SUITE_ACCURACIES+=("$DATASET|$surah_part|$suite_name|$accuracy")
        else
            echo "✓ $suite_name ($time_str)"
        fi
        PASSED=$((PASSED + 1))
    else
        local suite_end=$(date +%s)
        local elapsed=$((suite_end - suite_start))

        # Format time: show seconds if < 60s, otherwise minutes
        if [ $elapsed -lt 60 ]; then
            local time_str="${elapsed}s"
        else
            # Round to nearest minute with minimum of 1
            local minutes=$(echo "scale=0; m = ($elapsed + 30) / 60; if (m < 1) 1 else m" | bc)
            local time_str="${minutes}m"
        fi

        echo "✗ $suite_name ($time_str) FAILED"
        echo "   Check $log_file for details. Last 30 lines:"
        tail -30 "$log_file"
        FAILED=$((FAILED + 1))
        return 1
    fi

    return 0
}

# Create model backup at the very beginning (once per training run)
MODEL_PATH="../models/muhaffez_whisper.pt"
BACKUP_PATH="../models/muhaffez_whisper_backup.pt"
if [ -f "$MODEL_PATH" ] && [ ! -f "$BACKUP_PATH" ]; then
    cp "$MODEL_PATH" "$BACKUP_PATH"
    echo "✓ Model backup created: $BACKUP_PATH"
    echo ""
fi

# Run both training suites for each surah part
for SURAH_PART in "${SURAH_PARTS[@]}"; do
    # Extract surah number (e.g., "002" from "002-04")
    SURAH_NUM=$(echo "$SURAH_PART" | cut -d'-' -f1)

    # Always use log.txt for all training output
    LOG_FILE="$TRAIN_LOG"

    echo ""

    # Detect device info (only print once per surah) - write to log file
    {
        echo "============================================================"
        if command -v python3 &> /dev/null; then
            python3 -c "import torch; print('🚀 Using Metal GPU (Apple Silicon)' if torch.backends.mps.is_available() else ('🚀 Using CUDA GPU' if torch.cuda.is_available() else '⚠️  Using CPU (slower)')); print(f'Device: {\"mps\" if torch.backends.mps.is_available() else (\"cuda\" if torch.cuda.is_available() else \"cpu\")}')" 2>/dev/null || echo "Device: unknown"
        else
            echo "Device: unknown"
        fi
        echo "============================================================"
        echo ""
    } >> "$LOG_FILE"

    echo "════════════════════════════════════════════════════════════"
    echo "TRAINING SURAH PART: $SURAH_PART"
    echo "════════════════════════════════════════════════════════════"

    # Track start time for this surah part
    SURAH_START_TIME=$(date +%s)

    echo "Training $DATASET $SURAH_PART..."
    run_training_suite "train_curriculum.py" "Curriculum" "$SURAH_PART" "$LOG_FILE" || exit 1
    run_training_suite "train_full.py" "Full" "$SURAH_PART" "$LOG_FILE" || exit 1

    # Calculate total time for this surah part
    SURAH_END_TIME=$(date +%s)
    SURAH_ELAPSED=$((SURAH_END_TIME - SURAH_START_TIME))

    # Format time: show seconds if < 60s, otherwise minutes
    if [ $SURAH_ELAPSED -lt 60 ]; then
        SURAH_TIME_STR="${SURAH_ELAPSED}s"
    else
        # Round to nearest minute with minimum of 1
        SURAH_MINUTES=$(echo "scale=0; m = ($SURAH_ELAPSED + 30) / 60; if (m < 1) 1 else m" | bc)
        SURAH_TIME_STR="${SURAH_MINUTES}m"
    fi

    # Append total training time to log file
    echo "   Total training time: ${SURAH_TIME_STR}" >> "$LOG_FILE"
done

# Summary
END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))

# Format time: show seconds if < 60s, hours+minutes if >= 60m, otherwise minutes
if [ $TOTAL_ELAPSED -lt 60 ]; then
    TOTAL_TIME_STR="${TOTAL_ELAPSED}s"
elif [ $TOTAL_ELAPSED -ge 3600 ]; then
    # Show hours and minutes
    HOURS=$((TOTAL_ELAPSED / 3600))
    REMAINING_MINUTES=$(((TOTAL_ELAPSED % 3600) / 60))
    TOTAL_TIME_STR="${HOURS}h ${REMAINING_MINUTES}m"
else
    # Round to nearest minute with minimum of 1
    TOTAL_MINUTES=$(echo "scale=0; m = ($TOTAL_ELAPSED + 30) / 60; if (m < 1) 1 else m" | bc)
    TOTAL_TIME_STR="${TOTAL_MINUTES}m"
fi

echo ""
echo "Training Summary:"
echo "  Dataset: $DATASET"
echo "  Surah parts: ${SURAH_PARTS[@]}"
echo "  Completed suites: $PASSED"
if [ $FAILED -gt 0 ]; then
    echo "  Failed: $FAILED suites"
fi
echo "  Total time: ${TOTAL_TIME_STR}"

# Display accuracies grouped by surah part
if [ ${#SUITE_ACCURACIES[@]} -gt 0 ]; then
    echo ""
    echo "Accuracies:"

    # Track current surah part to group output
    local current_surah=""
    for acc in "${SUITE_ACCURACIES[@]}"; do
        # Split by | delimiter: dataset|surah_part|suite_name|accuracy
        IFS='|' read -r dataset surah_part suite_name accuracy <<< "$acc"

        # Print surah part header if changed
        if [ "$surah_part" != "$current_surah" ]; then
            echo "$dataset $surah_part"
            current_surah="$surah_part"
        fi

        # Print suite accuracy indented
        echo "  $suite_name: $accuracy"
    done
fi

# Exit with error if any training failed
if [ $FAILED -gt 0 ]; then
    echo ""
    echo "⚠️  Some training suites failed. Check individual log files for details."
    exit 1
else
    echo ""
    echo "✓ All training suites completed successfully!"
    exit 0
fi
