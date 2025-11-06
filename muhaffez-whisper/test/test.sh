#!/bin/bash
#
# Run both test suites: curriculum testing and full segments testing
# Usage:
#   ./test.sh                                   # Test all datasets
#   ./test.sh <dataset_name>                    # Test all surah parts in dataset
#   ./test.sh <dataset_name> <surah>            # Test all parts of specific surah (e.g., 002)
#   ./test.sh <dataset_name> <surah_part>       # Test specific surah part (e.g., 002-04)
#

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

    # Test each dataset
    for DS in "${DATASETS[@]}"; do
        # Recursively call this script with the dataset name
        "$0" "$DS"

        if [ $? -ne 0 ]; then
            echo "❌ Error: Testing failed for dataset $DS"
            exit 1
        fi
    done

    echo ""
    echo "✓ All datasets tested successfully!"
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

# Function to run a test suite for a specific surah part
run_test_suite() {
    local script=$1
    local suite_name=$2
    local surah_part=$3
    local log_file=$4
    TOTAL=$((TOTAL + 1))

    local suite_start=$(date +%s)

    if python3 -u "$script" "$DATASET" "$surah_part" >> "$log_file" 2>&1; then
        local suite_end=$(date +%s)
        local elapsed=$((suite_end - suite_start))
        local minutes=$((elapsed / 60))
        local seconds=$((elapsed % 60))

        echo "✓ $suite_name - $DATASET $surah_part (${minutes}m ${seconds}s)"
        PASSED=$((PASSED + 1))
    else
        local suite_end=$(date +%s)
        local elapsed=$((suite_end - suite_start))
        local minutes=$((elapsed / 60))
        local seconds=$((elapsed % 60))

        echo "✗ $suite_name - $DATASET $surah_part FAILED (${minutes}m ${seconds}s)"
        echo "   Check $log_file for details. Last 30 lines:"
        tail -30 "$log_file"
        FAILED=$((FAILED + 1))
        return 1
    fi

    return 0
}

# Run both test suites for each surah part
for SURAH_PART in "${SURAH_PARTS[@]}"; do
    # Extract surah number (e.g., "002" from "002-04")
    SURAH_NUM=$(echo "$SURAH_PART" | cut -d'-' -f1)

    # Set up log file for this dataset and surah with single backup
    LOG_FILE="log_${DATASET}_${SURAH_NUM}.txt"

    # If log file exists, move it to backup
    if [ -f "$LOG_FILE" ]; then
        mv "$LOG_FILE" "log_${DATASET}_${SURAH_NUM}_backup.txt"
    fi

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "TESTING SURAH PART: $SURAH_PART"
    echo "════════════════════════════════════════════════════════════"
    echo ""

    echo "Testing $DATASET $SURAH_PART..."
    run_test_suite "test_full.py" "Full" "$SURAH_PART" "$LOG_FILE" || exit 1
    run_test_suite "test_curriculum.py" "Curriculum" "$SURAH_PART" "$LOG_FILE" || exit 1
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

# Exit with error if any test failed
if [ $FAILED -gt 0 ]; then
    echo ""
    echo "⚠️  Some test suites failed. Check individual log files for details."
    exit 1
else
    echo ""
    echo "✓ All test suites completed successfully!"
    exit 0
fi
