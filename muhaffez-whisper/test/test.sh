#!/bin/bash
#
# Run both test suites: curriculum testing and full segments testing
# Usage:
#   ./test.sh                                   # Test all datasets
#   ./test.sh <dataset_name>                    # Test all surah parts in dataset
#   ./test.sh <dataset_name> <surah>            # Test all parts of specific surah (e.g., 002)
#   ./test.sh <dataset_name> <surah_part>       # Test specific surah part (e.g., 002-04)
#

# Main test log
TEST_LOG="log_test.txt"
TEST_LOG_BACKUP="log_test_backup.txt"

# If this is the initial call (not recursive), set up logging
if [ -z "$TEST_LOGGING_ACTIVE" ]; then
    export TEST_LOGGING_ACTIVE=1

    # Backup existing log if it exists
    if [ -f "$TEST_LOG" ]; then
        cp "$TEST_LOG" "$TEST_LOG_BACKUP"
        echo "✓ Test log backup created: $TEST_LOG_BACKUP"
    fi

    # Clear the log file
    > "$TEST_LOG"

    # Print device info once at the beginning of log_test.txt
    {
        echo "============================================================"
        if command -v python3 &> /dev/null; then
            python3 -c "import torch; print('🚀 Using Metal GPU (Apple Silicon)' if torch.backends.mps.is_available() else ('🚀 Using CUDA GPU' if torch.cuda.is_available() else '⚠️  Using CPU (slower)')); print(f'Device: {\"mps\" if torch.backends.mps.is_available() else (\"cuda\" if torch.cuda.is_available() else \"cpu\")}')" 2>/dev/null || echo "Device: unknown"
        else
            echo "Device: unknown"
        fi
        echo "============================================================"
        echo ""
    } > "$TEST_LOG"

    # Re-run this script with output redirected to log and console (append mode)
    "$0" "$@" 2>&1 | tee -a "$TEST_LOG"
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
TOTAL_CORRECT=0
TOTAL_TOKENS=0
# Array to store accuracy results for final summary
declare -a ACCURACY_RESULTS

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

        # Extract accuracy and token counts from log file (last occurrence of "Token accuracy")
        local accuracy=$(grep "Token accuracy:" "$log_file" | tail -1 | sed 's/.*(\([0-9.]*\)%).*/\1/')
        local correct=$(grep "Token accuracy:" "$log_file" | tail -1 | sed 's/Token accuracy: \([0-9]*\).*/\1/')
        local total=$(grep "Token accuracy:" "$log_file" | tail -1 | sed 's/Token accuracy: [0-9]*\/\([0-9]*\).*/\1/')

        # Accumulate totals
        TOTAL_CORRECT=$((TOTAL_CORRECT + correct))
        TOTAL_TOKENS=$((TOTAL_TOKENS + total))

        # Store result for final summary
        ACCURACY_RESULTS+=("$suite_name - $DATASET $surah_part: ${accuracy}%")

        echo "✓ $suite_name (${minutes}m ${seconds}s) - Accuracy: ${accuracy}%"
        PASSED=$((PASSED + 1))
    else
        local suite_end=$(date +%s)
        local elapsed=$((suite_end - suite_start))
        local minutes=$((elapsed / 60))
        local seconds=$((elapsed % 60))

        echo "✗ $suite_name (${minutes}m ${seconds}s) - FAILED"
        echo "   Check $log_file for details. Last 30 lines:"
        tail -30 "$log_file"
        FAILED=$((FAILED + 1))
        return 1
    fi

    return 0
}

# Track which surahs we've cleared logs for (using simple variable)
CLEARED_LOGS=""

# Run both test suites for each surah part
for SURAH_PART in "${SURAH_PARTS[@]}"; do
    # Extract surah number (e.g., "002" from "002-04")
    SURAH_NUM=$(echo "$SURAH_PART" | cut -d'-' -f1)

    # Set up log file for this dataset and surah (append mode)
    LOG_FILE="log_${DATASET}_${SURAH_NUM}.txt"

    # Create backup and clear log file only once per surah (on first part)
    if [[ ! "$CLEARED_LOGS" =~ $SURAH_NUM ]]; then
        if [ -f "$LOG_FILE" ]; then
            cp "$LOG_FILE" "log_${DATASET}_${SURAH_NUM}_backup.txt"
            echo ""
            echo "✓ Backup created: log_${DATASET}_${SURAH_NUM}_backup.txt"
        fi
        > "$LOG_FILE"
        CLEARED_LOGS="$CLEARED_LOGS $SURAH_NUM"
    fi

    # Print vocabulary size once per surah
    if [[ ! "$CLEARED_LOGS" =~ "DEVICE_$SURAH_NUM" ]]; then
        {
            # Print vocabulary size once
            if [ -f "../models/vocabulary.json" ]; then
                VOCAB_SIZE=$(python3 -c "import json; vocab = json.load(open('../models/vocabulary.json')); print(len(vocab))" 2>/dev/null)
                if [ -n "$VOCAB_SIZE" ]; then
                    echo "Vocabulary size: $VOCAB_SIZE"
                    echo ""
                fi
            fi
        } >> "$LOG_FILE"
        CLEARED_LOGS="$CLEARED_LOGS DEVICE_$SURAH_NUM"
    fi

    # Write surah part header to log file
    {
        echo ""
        echo "============================================================"
        echo "TESTING SURAH PART: $SURAH_PART"
        echo "============================================================"
        echo ""
    } >> "$LOG_FILE"

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "TESTING SURAH PART: $SURAH_PART"
    echo "════════════════════════════════════════════════════════════"
    echo "Testing $DATASET $SURAH_PART..."
    run_test_suite "test_full.py" "Full" "$SURAH_PART" "$LOG_FILE" || exit 1
    run_test_suite "test_curriculum.py" "Curriculum" "$SURAH_PART" "$LOG_FILE" || exit 1
    echo ""
done

# Summary
END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((TOTAL_ELAPSED / 60))
SECONDS=$((TOTAL_ELAPSED % 60))

# Calculate overall accuracy
OVERALL_ACCURACY=0
if [ $TOTAL_TOKENS -gt 0 ]; then
    OVERALL_ACCURACY=$(awk "BEGIN {printf \"%.1f\", ($TOTAL_CORRECT / $TOTAL_TOKENS) * 100}")
fi

echo ""
echo "Testing Summary:"
echo "  Dataset: $DATASET"
echo "  Surah parts: ${SURAH_PARTS[@]}"
echo "  Total runs: $TOTAL test suites"
echo "  Completed: $PASSED suites"
echo "  Failed: $FAILED suites"
echo "  Time: ${MINUTES}m ${SECONDS}s"
echo ""
echo "Accuracies:"

# Group accuracies by surah part
current_surah=""
for result in "${ACCURACY_RESULTS[@]}"; do
    # Split by | delimiter: suite_name - dataset surah_part: accuracy
    # Format: "Full - Quran-A 001: 100.0%"
    suite_name=$(echo "$result" | sed 's/ - .*//')
    rest=$(echo "$result" | sed 's/[^-]* - //')
    dataset_surah=$(echo "$rest" | sed 's/:.*//')
    accuracy=$(echo "$rest" | sed 's/.*: //')

    # Check if we're on a new surah part
    if [ "$dataset_surah" != "$current_surah" ]; then
        echo "$dataset_surah"
        current_surah="$dataset_surah"
    fi

    # Print suite accuracy indented
    echo "  $suite_name: $accuracy"
done

echo ""
echo "Overall accuracy: $TOTAL_CORRECT/$TOTAL_TOKENS ($OVERALL_ACCURACY%)"

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
