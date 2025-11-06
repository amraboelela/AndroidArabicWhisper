#!/bin/bash
#
# Run both test suites: curriculum testing and full segments testing
# Usage:
#   ./test.sh                                   # Test all datasets
#   ./test.sh <dataset_name>                    # Test all surah parts in dataset
#   ./test.sh <dataset_name> <surah>            # Test all parts of specific surah (e.g., 002)
#   ./test.sh <dataset_name> <surah_part>       # Test specific surah part (e.g., 002-04)
#

# Summary log file setup (do this first, before any recursive calls)
SUMMARY_LOG="log_test.txt"
DAY_NUM=$(date +%u)  # Day of week (1=Monday, 7=Sunday)
if [ -f "$SUMMARY_LOG" ]; then
    mv "$SUMMARY_LOG" "log_test.txt.${DAY_NUM}"
fi

# Get parameters
DATASET=${1}
SURAH_OR_PART=${2}

# If no dataset parameter, find all datasets
if [ -z "$DATASET" ]; then
    DATASETS=($(ls -d ../datasets/*/ 2>/dev/null | xargs -n 1 basename))
    if [ ${#DATASETS[@]} -eq 0 ]; then
        MSG="❌ Error: No datasets found in ../datasets/"
        echo "$MSG"
        echo "$MSG" >> "$SUMMARY_LOG"
        exit 1
    fi
    MSG="Found ${#DATASETS[@]} dataset(s): ${DATASETS[@]}"
    echo "$MSG"
    echo "$MSG" >> "$SUMMARY_LOG"

    # Test each dataset
    for DS in "${DATASETS[@]}"; do
        # Recursively call this script with the dataset name
        "$0" "$DS"

        if [ $? -ne 0 ]; then
            MSG="❌ Error: Testing failed for dataset $DS"
            echo "$MSG"
            echo "$MSG" >> "$SUMMARY_LOG"
            exit 1
        fi
    done

    echo ""
    MSG="✓ All datasets tested successfully!"
    echo "$MSG"
    echo "$MSG" >> "$SUMMARY_LOG"
    exit 0
fi

# If no second parameter, find all text files in dataset
if [ -z "$SURAH_OR_PART" ]; then
    SURAH_PARTS=($(ls ../datasets/${DATASET}/text/*.txt 2>/dev/null | xargs -n 1 basename | sed 's/.txt//'))
    if [ ${#SURAH_PARTS[@]} -eq 0 ]; then
        MSG="❌ Error: No text files found in ../datasets/${DATASET}/text/"
        echo "$MSG"
        echo "$MSG" >> "$SUMMARY_LOG"
        exit 1
    fi
    MSG="Found ${#SURAH_PARTS[@]} surah parts in ${DATASET}"
    echo "$MSG"
    echo "$MSG" >> "$SUMMARY_LOG"
# If parameter looks like a surah number (e.g., "002"), find all parts for that surah
elif [[ "$SURAH_OR_PART" =~ ^[0-9]{3}$ ]]; then
    SURAH_PARTS=($(ls ../datasets/${DATASET}/text/${SURAH_OR_PART}*.txt 2>/dev/null | xargs -n 1 basename | sed 's/.txt//'))
    if [ ${#SURAH_PARTS[@]} -eq 0 ]; then
        MSG="❌ Error: No text files found for surah ${SURAH_OR_PART} in ../datasets/${DATASET}/text/"
        echo "$MSG"
        echo "$MSG" >> "$SUMMARY_LOG"
        exit 1
    fi
    MSG="Found ${#SURAH_PARTS[@]} parts for surah ${SURAH_OR_PART}"
    echo "$MSG"
    echo "$MSG" >> "$SUMMARY_LOG"
# Otherwise, treat as specific surah part
else
    SURAH_PARTS=("$SURAH_OR_PART")
fi

# Track results
TOTAL=0
PASSED=0
FAILED=0
START_TIME=$(date +%s)

# Function to run a test suite for a specific surah part
run_test_suite() {
    local script=$1
    local suite_name=$2
    local surah_part=$3
    local log_file=$4
    TOTAL=$((TOTAL + 1))

    local suite_start=$(date +%s)

    if python3 "$script" "$DATASET" "$surah_part" >> "$log_file" 2>&1; then
        local suite_end=$(date +%s)
        local elapsed=$((suite_end - suite_start))
        local minutes=$((elapsed / 60))
        local seconds=$((elapsed % 60))

        local msg="✓ $suite_name - $DATASET $surah_part (${minutes}m ${seconds}s)"
        echo "$msg"
        echo "$msg" >> "$SUMMARY_LOG"
        PASSED=$((PASSED + 1))
    else
        local suite_end=$(date +%s)
        local elapsed=$((suite_end - suite_start))
        local minutes=$((elapsed / 60))
        local seconds=$((elapsed % 60))

        local msg="✗ $suite_name - $DATASET $surah_part FAILED (${minutes}m ${seconds}s)"
        echo "$msg"
        echo "$msg" >> "$SUMMARY_LOG"
        echo "   Check $log_file for details. Last 30 lines:"
        echo "   Check $log_file for details. Last 30 lines:" >> "$SUMMARY_LOG"
        tail -30 "$log_file"
        tail -30 "$log_file" >> "$SUMMARY_LOG"
        FAILED=$((FAILED + 1))
        return 1
    fi

    return 0
}

# Run both test suites for each surah part
for SURAH_PART in "${SURAH_PARTS[@]}"; do
    # Extract surah number (e.g., "002" from "002-04")
    SURAH_NUM=$(echo "$SURAH_PART" | cut -d'-' -f1)

    # Set up log file for this dataset and surah with day rotation
    DAY_NUM=$(date +%u)  # Day of week (1=Monday, 7=Sunday)
    LOG_FILE="log_${DATASET}_${SURAH_NUM}.txt"

    # If log file exists, move it to day-specific backup
    if [ -f "$LOG_FILE" ]; then
        mv "$LOG_FILE" "log_${DATASET}_${SURAH_NUM}.txt.${DAY_NUM}"
    fi

    # Write header to log file
    echo "" >> "$LOG_FILE"
    echo "════════════════════════════════════════════════════════════" >> "$LOG_FILE"
    echo "TESTING SURAH PART: $SURAH_PART" >> "$LOG_FILE"
    echo "════════════════════════════════════════════════════════════" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"

    MSG="Testing $DATASET $SURAH_PART..."
    echo "$MSG"
    echo "$MSG" >> "$SUMMARY_LOG"
    run_test_suite "test_curriculum.py" "Curriculum" "$SURAH_PART" "$LOG_FILE" || exit 1
    run_test_suite "test_full.py" "Full" "$SURAH_PART" "$LOG_FILE" || exit 1
done

# Summary
END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((TOTAL_ELAPSED / 60))
SECONDS=$((TOTAL_ELAPSED % 60))

echo ""
echo "Testing Summary:"
echo "  Dataset: $DATASET"
echo "  Surah parts: ${SURAH_PARTS[@]}"
echo "  Total runs: $TOTAL test suites"
echo "  Completed: $PASSED suites"
echo "  Failed: $FAILED suites"
echo "  Time: ${MINUTES}m ${SECONDS}s"

# Write summary to log file
echo "" >> "$SUMMARY_LOG"
echo "Testing Summary:" >> "$SUMMARY_LOG"
echo "  Dataset: $DATASET" >> "$SUMMARY_LOG"
echo "  Surah parts: ${SURAH_PARTS[@]}" >> "$SUMMARY_LOG"
echo "  Total runs: $TOTAL test suites" >> "$SUMMARY_LOG"
echo "  Completed: $PASSED suites" >> "$SUMMARY_LOG"
echo "  Failed: $FAILED suites" >> "$SUMMARY_LOG"
echo "  Time: ${MINUTES}m ${SECONDS}s" >> "$SUMMARY_LOG"

# Exit with error if any test failed
if [ $FAILED -gt 0 ]; then
    echo ""
    echo "⚠️  Some test suites failed. Check individual log files for details."
    echo "" >> "$SUMMARY_LOG"
    echo "⚠️  Some test suites failed. Check individual log files for details." >> "$SUMMARY_LOG"
    exit 1
else
    echo ""
    echo "✓ All test suites completed successfully!"
    echo "" >> "$SUMMARY_LOG"
    echo "✓ All test suites completed successfully!" >> "$SUMMARY_LOG"
    exit 0
fi
