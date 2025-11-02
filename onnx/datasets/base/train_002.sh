#!/bin/bash
#
# Run training scripts for Al-Baqara (002) and log results to log_train_002.txt
#
# Training scripts (curriculum learning order):
# 1. train_002_1.py - Al-Baqara (002) first 1 second → first word
# 2. train_002_3.py - Al-Baqara (002) first 3 seconds → first 2 words
# 3. train_002.py - Al-Baqara (002) full segments → full transcriptions
#

LOG_FILE="../../log_train_002.txt"

echo "============================================================" | tee "$LOG_FILE"
echo "RUNNING AL-BAQARA (002) TRAINING SCRIPTS (CURRICULUM LEARNING)" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Track results
TOTAL=0
PASSED=0
FAILED=0
START_TIME=$(date +%s)

# Function to run a training script
run_training() {
    local script=$1
    TOTAL=$((TOTAL + 1))

    echo "============================================================" | tee -a "$LOG_FILE"
    echo "[$TOTAL/3] Starting: $script" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"

    local script_start=$(date +%s)

    if python3 "$script" >> "$LOG_FILE" 2>&1; then
        local script_end=$(date +%s)
        local elapsed=$((script_end - script_start))

        # Extract accuracy percentage from log file if available
        local accuracy=$(grep -o "Accuracy: [0-9]*/[0-9]* ([0-9.]*%)" "$LOG_FILE" | tail -1 | grep -o "([0-9.]*%)" | tr -d "()")

        echo "" | tee -a "$LOG_FILE"
        if [ -n "$accuracy" ]; then
            echo "✓ COMPLETED: $script - $accuracy (${elapsed}s)" | tee -a "$LOG_FILE"
        else
            echo "✓ COMPLETED: $script (${elapsed}s)" | tee -a "$LOG_FILE"
        fi
        PASSED=$((PASSED + 1))
    else
        local script_end=$(date +%s)
        local elapsed=$((script_end - script_start))
        echo "" | tee -a "$LOG_FILE"
        echo "✗ FAILED: $script (${elapsed}s)" | tee -a "$LOG_FILE"
        FAILED=$((FAILED + 1))

        # Ask if user wants to continue
        echo "" | tee -a "$LOG_FILE"
        echo "⚠️  Training failed at $script" | tee -a "$LOG_FILE"
        echo "Do you want to continue with remaining scripts? (y/n)" | tee -a "$LOG_FILE"
        read -r response
        if [ "$response" != "y" ] && [ "$response" != "Y" ]; then
            echo "Aborting training pipeline." | tee -a "$LOG_FILE"
            return 1
        fi
    fi

    echo "" | tee -a "$LOG_FILE"
    return 0
}

# Run all training scripts for 002 in curriculum learning order
run_training "train_002_1.py" || exit 1
run_training "train_002_3.py" || exit 1
run_training "train_002.py" || exit 1

# Summary
END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((TOTAL_ELAPSED / 60))
SECONDS=$((TOTAL_ELAPSED % 60))

echo "============================================================" | tee -a "$LOG_FILE"
echo "AL-BAQARA (002) TRAINING SUMMARY" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "Total:     $TOTAL training scripts" | tee -a "$LOG_FILE"
echo "Completed: $PASSED scripts" | tee -a "$LOG_FILE"
echo "Failed:    $FAILED scripts" | tee -a "$LOG_FILE"
echo "Time:      ${MINUTES}m ${SECONDS}s" | tee -a "$LOG_FILE"
echo "Ended:     $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# Exit with error if any training failed
if [ $FAILED -gt 0 ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "⚠️  Some training scripts failed. Check $LOG_FILE for details." | tee -a "$LOG_FILE"
    exit 1
else
    echo "" | tee -a "$LOG_FILE"
    echo "✓ All training scripts completed successfully!" | tee -a "$LOG_FILE"
    echo "Model saved to: encoder_decoder_model.pt" | tee -a "$LOG_FILE"
    exit 0
fi
