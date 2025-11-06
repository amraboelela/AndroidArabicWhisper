#!/bin/bash
#
# Run both training suites: curriculum learning and full dataset training
# Usage:
#   ./train.sh <dataset_name>                    # Train all surah parts in dataset
#   ./train.sh <dataset_name> <surah>            # Train all parts of specific surah (e.g., 002)
#   ./train.sh <dataset_name> <surah_part>       # Train specific surah part (e.g., 002-04)
#

# Get parameters
DATASET=${1:-Quran-A}
SURAH_OR_PART=${2}

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

LOG_FILE="log_train.txt"

echo "============================================================" | tee "$LOG_FILE"
echo "RUNNING BOTH TRAINING SUITES" | tee -a "$LOG_FILE"
echo "Dataset: $DATASET" | tee -a "$LOG_FILE"
echo "Surah parts to train: ${SURAH_PARTS[@]}" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Track results
TOTAL=0
PASSED=0
FAILED=0
START_TIME=$(date +%s)

# Function to run a training suite for a specific surah part
run_training_suite() {
    local script=$1
    local suite_name=$2
    local surah_part=$3
    TOTAL=$((TOTAL + 1))

    echo "============================================================" | tee -a "$LOG_FILE"
    echo "[$TOTAL] Starting: $suite_name - $surah_part" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"

    local suite_start=$(date +%s)

    if python3 "$script" "$DATASET" "$surah_part" 2>&1 | tee -a "$LOG_FILE"; then
        local suite_end=$(date +%s)
        local elapsed=$((suite_end - suite_start))
        local minutes=$((elapsed / 60))
        local seconds=$((elapsed % 60))

        echo "" | tee -a "$LOG_FILE"
        echo "✓ COMPLETED: $suite_name - $surah_part (${minutes}m ${seconds}s)" | tee -a "$LOG_FILE"
        PASSED=$((PASSED + 1))
    else
        local suite_end=$(date +%s)
        local elapsed=$((suite_end - suite_start))
        local minutes=$((elapsed / 60))
        local seconds=$((elapsed % 60))

        echo "" | tee -a "$LOG_FILE"
        echo "✗ FAILED: $suite_name - $surah_part (${minutes}m ${seconds}s)" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
        echo "ERROR: Training suite failed. Check $LOG_FILE for details." | tee -a "$LOG_FILE"
        echo "Last 30 lines of log:" | tee -a "$LOG_FILE"
        tail -30 "$LOG_FILE" | tee -a "$LOG_FILE"
        FAILED=$((FAILED + 1))
        return 1
    fi

    echo "" | tee -a "$LOG_FILE"
    return 0
}

# Run both training suites for each surah part
for SURAH_PART in "${SURAH_PARTS[@]}"; do
    echo "" | tee -a "$LOG_FILE"
    echo "════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
    echo "TRAINING SURAH PART: $SURAH_PART" | tee -a "$LOG_FILE"
    echo "════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    run_training_suite "train_curriculum.py" "Curriculum Learning" "$SURAH_PART" || exit 1
    run_training_suite "train_full.py" "Full Segments" "$SURAH_PART" || exit 1
done

# Summary
END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((TOTAL_ELAPSED / 60))
SECONDS=$((TOTAL_ELAPSED % 60))

echo "============================================================" | tee -a "$LOG_FILE"
echo "TRAINING PIPELINE SUMMARY" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "Dataset:       $DATASET" | tee -a "$LOG_FILE"
echo "Surah parts:   ${SURAH_PARTS[@]}" | tee -a "$LOG_FILE"
echo "Total runs:    $TOTAL training suites" | tee -a "$LOG_FILE"
echo "Completed:     $PASSED suites" | tee -a "$LOG_FILE"
echo "Failed:        $FAILED suites" | tee -a "$LOG_FILE"
echo "Time:          ${MINUTES}m ${SECONDS}s" | tee -a "$LOG_FILE"
echo "Ended:         $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# Exit with error if any training failed
if [ $FAILED -gt 0 ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "⚠️  Some training suites failed. Check $LOG_FILE for details." | tee -a "$LOG_FILE"
    exit 1
else
    echo "" | tee -a "$LOG_FILE"
    echo "✓ All training suites completed successfully!" | tee -a "$LOG_FILE"
    echo "Model saved to: ../models/encoder_decoder_model.pt" | tee -a "$LOG_FILE"
    exit 0
fi
