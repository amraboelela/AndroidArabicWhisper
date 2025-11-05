#!/bin/bash
#
# Run test scripts for Al-Fatiha (001) and log results to log_test_001.txt
#
# Test scripts:
# 1. test_001.py - Al-Fatiha (001) full segments
# 2. test_001_01.py - Al-Fatiha (001) first 1 second
# 3. test_001_03.py - Al-Fatiha (001) first 3 seconds
#

# Get dataset name parameter (defaults to base)
DATASET=${1:-base}

LOG_FILE="log_test_001.txt"

echo "============================================================" | tee "$LOG_FILE"
echo "RUNNING AL-FATIHA (001) TEST SCRIPTS" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Track results
TOTAL=0
PASSED=0
FAILED=0
TOTAL_CORRECT=0
TOTAL_SAMPLES=0

# Function to run a test script
run_test() {
    local script=$1
    TOTAL=$((TOTAL + 1))

    echo "------------------------------------------------------------" | tee -a "$LOG_FILE"
    echo "[$TOTAL/3] Running: $script" | tee -a "$LOG_FILE"
    echo "------------------------------------------------------------" | tee -a "$LOG_FILE"

    if python3 "$script" >> "$LOG_FILE" 2>&1; then
        # Extract accuracy percentage from log file (handles both "Accuracy:" and "Token accuracy:")
        local accuracy=$(grep -E "(Token accuracy|Accuracy): [0-9]*/[0-9]* \([0-9.]*%\)" "$LOG_FILE" | tail -1 | grep -o "([0-9.]*%)" | tr -d "()")

        # Extract correct/total counts for overall accuracy
        local counts=$(grep -E "(Token accuracy|Accuracy): [0-9]*/[0-9]* \([0-9.]*%\)" "$LOG_FILE" | tail -1 | grep -o "[0-9]*/[0-9]*")
        if [ -n "$counts" ]; then
            local correct=$(echo "$counts" | cut -d'/' -f1)
            local total=$(echo "$counts" | cut -d'/' -f2)
            TOTAL_CORRECT=$((TOTAL_CORRECT + correct))
            TOTAL_SAMPLES=$((TOTAL_SAMPLES + total))
        fi

        if [ -n "$accuracy" ]; then
            echo "✓ PASSED: $script - $accuracy" | tee -a "$LOG_FILE"
        else
            echo "✓ PASSED: $script" | tee -a "$LOG_FILE"
        fi
        PASSED=$((PASSED + 1))
    else
        echo "✗ FAILED: $script" | tee -a "$LOG_FILE"
        FAILED=$((FAILED + 1))
    fi

    echo "" | tee -a "$LOG_FILE"
}

# Run all test scripts for 001
run_test "test_001.py"
run_test "test_001_01.py"
run_test "test_001_03.py"

# Summary
echo "============================================================" | tee -a "$LOG_FILE"
echo "AL-FATIHA (001) TEST SUMMARY" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "Total:  $TOTAL tests" | tee -a "$LOG_FILE"
echo "Passed: $PASSED tests" | tee -a "$LOG_FILE"
echo "Failed: $FAILED tests" | tee -a "$LOG_FILE"

# Calculate and display overall accuracy
if [ $TOTAL_SAMPLES -gt 0 ]; then
    OVERALL_ACCURACY=$(echo "scale=1; $TOTAL_CORRECT * 100 / $TOTAL_SAMPLES" | bc)
    echo "Overall Accuracy: $TOTAL_CORRECT/$TOTAL_SAMPLES ($OVERALL_ACCURACY%)" | tee -a "$LOG_FILE"
fi

echo "Ended:  $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# Exit with error if any tests failed
if [ $FAILED -gt 0 ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "⚠️  Some tests failed. Check $LOG_FILE for details." | tee -a "$LOG_FILE"
    exit 1
else
    echo "" | tee -a "$LOG_FILE"
    if [ $TOTAL_SAMPLES -gt 0 ]; then
        OVERALL_ACCURACY=$(echo "scale=1; $TOTAL_CORRECT * 100 / $TOTAL_SAMPLES" | bc)
        echo "✓ All tests passed! Overall accuracy: $OVERALL_ACCURACY%" | tee -a "$LOG_FILE"
    else
        echo "✓ All tests passed!" | tee -a "$LOG_FILE"
    fi
    exit 0
fi
