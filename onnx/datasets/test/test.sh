#!/bin/bash
#
# Run all test scripts by calling individual test_*.sh scripts
#

# Get dataset name parameter (defaults to base)
DATASET=${1:-base}

LOG_FILE="log_test.txt"

echo "============================================================" | tee "$LOG_FILE"
echo "RUNNING ALL TEST SCRIPTS" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Track results
TOTAL=0
PASSED=0
FAILED=0
TOTAL_CORRECT=0
TOTAL_SAMPLES=0

# Function to run a test suite
run_test_suite() {
    local script=$1
    local suite_name=$2
    TOTAL=$((TOTAL + 1))

    echo "============================================================" | tee -a "$LOG_FILE"
    echo "[$TOTAL/2] Running: $suite_name" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"

    if ./"$script" >> "$LOG_FILE" 2>&1; then
        # Extract overall accuracy from the suite's log
        local suite_log="${script/test_/log_test_}"
        suite_log="${suite_log/.sh/.txt}"
        local accuracy=$(grep "Overall Accuracy:" "$suite_log" | tail -1 | grep -o "([0-9.]*%)" | tr -d "()")

        # Extract correct/total counts
        local counts=$(grep "Overall Accuracy:" "$suite_log" | tail -1 | grep -o "[0-9]*/[0-9]*")
        if [ -n "$counts" ]; then
            local correct=$(echo "$counts" | cut -d'/' -f1)
            local total=$(echo "$counts" | cut -d'/' -f2)
            TOTAL_CORRECT=$((TOTAL_CORRECT + correct))
            TOTAL_SAMPLES=$((TOTAL_SAMPLES + total))
        fi

        if [ -n "$accuracy" ]; then
            echo "✓ PASSED: $suite_name - $accuracy" | tee -a "$LOG_FILE"
        else
            echo "✓ PASSED: $suite_name" | tee -a "$LOG_FILE"
        fi
        PASSED=$((PASSED + 1))
    else
        echo "✗ FAILED: $suite_name" | tee -a "$LOG_FILE"
        FAILED=$((FAILED + 1))
    fi

    echo "" | tee -a "$LOG_FILE"
}

# Run all test suites
run_test_suite "test_001.sh" "Al-Fatiha (001) Tests"
run_test_suite "test_002.sh" "Al-Baqara (002) Tests"

# Summary
echo "============================================================" | tee -a "$LOG_FILE"
echo "TEST SUMMARY" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "Total:  $TOTAL test suites" | tee -a "$LOG_FILE"
echo "Passed: $PASSED test suites" | tee -a "$LOG_FILE"
echo "Failed: $FAILED test suites" | tee -a "$LOG_FILE"

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
    echo "⚠️  Some test suites failed. Check $LOG_FILE for details." | tee -a "$LOG_FILE"
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
