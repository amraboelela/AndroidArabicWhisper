#!/bin/bash
#
# Run both training suites: curriculum learning and full dataset training
# Usage:
#   ./train.sh                                   # Train all datasets
#   ./train.sh <dataset_name>                    # Train all surah parts in dataset
#   ./train.sh <dataset_name> <surah>            # Train all parts of specific surah (e.g., 002)
#   ./train.sh <dataset_name> <surah_part>       # Train specific surah part (e.g., 002-04)
#

# Main training log
TRAIN_LOG="log_train.txt"
TRAIN_LOG_BACKUP="log_train_backup.txt"

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

# If no dataset parameter, find all datasets
if [ -z "$DATASET" ]; then
    DATASETS=($(ls -d ../datasets/*/ 2>/dev/null | xargs -n 1 basename))
    if [ ${#DATASETS[@]} -eq 0 ]; then
        echo "❌ Error: No datasets found in ../datasets/"
        exit 1
    fi
    echo "Found ${#DATASETS[@]} dataset(s): ${DATASETS[@]}"

    # Train each dataset
    for DS in "${DATASETS[@]}"; do
        # Recursively call this script with the dataset name
        "$0" "$DS"

        if [ $? -ne 0 ]; then
            echo "❌ Error: Training failed for dataset $DS"
            exit 1
        fi
    done

    echo ""
    echo "✓ All datasets trained successfully!"
    exit 0
fi

# If no second parameter, find all text files in dataset
if [ -z "$SURAH_OR_PART" ]; then
    SURAH_PARTS=($(ls ../datasets/${DATASET}/text/*.txt 2>/dev/null | xargs -n 1 basename | sed 's/.txt//'))
    if [ ${#SURAH_PARTS[@]} -eq 0 ]; then
        echo "❌ Error: No text files found in ../datasets/${DATASET}/text/"
        exit 1
    fi
    echo "Found ${#SURAH_PARTS[@]} surah parts in ${DATASET}"
# If parameter looks like a surah number (e.g., "002"), find all parts for that surah
elif [[ "$SURAH_OR_PART" =~ ^[0-9]{3}$ ]]; then
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
            SUITE_ACCURACIES+=("$suite_name - $DATASET $surah_part: $accuracy")
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

# Track which surahs we've cleared logs for (using simple variable)
CLEARED_LOGS=""

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

    # Set up log file for this dataset and surah
    LOG_FILE="log_${DATASET}_${SURAH_NUM}.txt"

    # Backup and clear the log file to start fresh (only once per surah)
    if [[ ! "$CLEARED_LOGS" =~ $SURAH_NUM ]]; then
        # Create backup if log file exists
        if [ -f "$LOG_FILE" ]; then
            BACKUP_LOG="log_${DATASET}_${SURAH_NUM}_backup.txt"
            cp "$LOG_FILE" "$BACKUP_LOG"
            echo ""
            echo "✓ Log backup created: $BACKUP_LOG"
        fi
        > "$LOG_FILE"
        CLEARED_LOGS="$CLEARED_LOGS $SURAH_NUM"
    fi

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "TRAINING SURAH PART: $SURAH_PART"
    echo "════════════════════════════════════════════════════════════"
    echo ""

    # Track start time for this surah part
    SURAH_START_TIME=$(date +%s)

    echo "Training $DATASET $SURAH_PART..."
    run_training_suite "train_full.py" "Full" "$SURAH_PART" "$LOG_FILE" || exit 1
    run_training_suite "train_curriculum.py" "Curriculum" "$SURAH_PART" "$LOG_FILE" || exit 1

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

# Format time: show seconds if < 60s, otherwise minutes
if [ $TOTAL_ELAPSED -lt 60 ]; then
    TOTAL_TIME_STR="${TOTAL_ELAPSED}s"
else
    # Round to nearest minute with minimum of 1
    TOTAL_MINUTES=$(echo "scale=0; m = ($TOTAL_ELAPSED + 30) / 60; if (m < 1) 1 else m" | bc)
    TOTAL_TIME_STR="${TOTAL_MINUTES}m"
fi

echo ""
echo "Training Summary:"
echo "  Dataset: $DATASET"
echo "  Surah parts: ${SURAH_PARTS[@]}"
echo "  Total runs: $TOTAL training suites"
echo "  Completed: $PASSED suites"
echo "  Failed: $FAILED suites"
echo "  Time: ${TOTAL_TIME_STR}"

# Display accuracies for each suite
if [ ${#SUITE_ACCURACIES[@]} -gt 0 ]; then
    echo ""
    echo "Accuracies:"
    for acc in "${SUITE_ACCURACIES[@]}"; do
        echo "  $acc"
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
