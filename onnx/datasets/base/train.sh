#!/bin/bash
#
# Run all training scripts by calling individual train_*.sh scripts (curriculum learning)
#

LOG_FILE="../../log_train.txt"

echo "============================================================" | tee "$LOG_FILE"
echo "RUNNING ALL TRAINING SCRIPTS (CURRICULUM LEARNING)" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Track results
TOTAL=0
PASSED=0
FAILED=0
START_TIME=$(date +%s)

# Function to run a training suite
run_training_suite() {
    local script=$1
    local suite_name=$2
    TOTAL=$((TOTAL + 1))

    echo "============================================================" | tee -a "$LOG_FILE"
    echo "[$TOTAL/2] Starting: $suite_name" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"

    local suite_start=$(date +%s)

    if ./"$script" >> "$LOG_FILE" 2>&1; then
        local suite_end=$(date +%s)
        local elapsed=$((suite_end - suite_start))
        local minutes=$((elapsed / 60))
        local seconds=$((elapsed % 60))

        echo "" | tee -a "$LOG_FILE"
        echo "✓ COMPLETED: $suite_name (${minutes}m ${seconds}s)" | tee -a "$LOG_FILE"
        PASSED=$((PASSED + 1))
    else
        local suite_end=$(date +%s)
        local elapsed=$((suite_end - suite_start))
        local minutes=$((elapsed / 60))
        local seconds=$((elapsed % 60))

        echo "" | tee -a "$LOG_FILE"
        echo "✗ FAILED: $suite_name (${minutes}m ${seconds}s)" | tee -a "$LOG_FILE"
        FAILED=$((FAILED + 1))

        # Ask if user wants to continue
        echo "" | tee -a "$LOG_FILE"
        echo "⚠️  Training failed at $suite_name" | tee -a "$LOG_FILE"
        echo "Do you want to continue with remaining suites? (y/n)" | tee -a "$LOG_FILE"
        read -r response
        if [ "$response" != "y" ] && [ "$response" != "Y" ]; then
            echo "Aborting training pipeline." | tee -a "$LOG_FILE"
            return 1
        fi
    fi

    echo "" | tee -a "$LOG_FILE"
    return 0
}

# Run all training suites in curriculum learning order (001 then 002)
run_training_suite "train_001.sh" "Al-Fatiha (001) Training" || exit 1
run_training_suite "train_002.sh" "Al-Baqara (002) Training" || exit 1

# Summary
END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((TOTAL_ELAPSED / 60))
SECONDS=$((TOTAL_ELAPSED % 60))

echo "============================================================" | tee -a "$LOG_FILE"
echo "TRAINING PIPELINE SUMMARY" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "Total:     $TOTAL training suites" | tee -a "$LOG_FILE"
echo "Completed: $PASSED suites" | tee -a "$LOG_FILE"
echo "Failed:    $FAILED suites" | tee -a "$LOG_FILE"
echo "Time:      ${MINUTES}m ${SECONDS}s" | tee -a "$LOG_FILE"
echo "Ended:     $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# Exit with error if any training failed
if [ $FAILED -gt 0 ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "⚠️  Some training suites failed. Check $LOG_FILE for details." | tee -a "$LOG_FILE"
    exit 1
else
    echo "" | tee -a "$LOG_FILE"
    echo "✓ All training suites completed successfully!" | tee -a "$LOG_FILE"
    echo "Model saved to: encoder_decoder_model.pt" | tee -a "$LOG_FILE"
    exit 0
fi
