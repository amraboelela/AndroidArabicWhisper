#!/bin/bash

# run_all_tests.sh - Run all unit tests and stop on first failure
# Created by Amr Aboelela

set -e  # Exit immediately if a command exits with a non-zero status

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to run a test and handle results
run_test() {
    local test_script="$1"
    local test_name=$(basename "$test_script" .sh)

    print_status "Running $test_name..."
    echo "=================================================="

    if [ -f "$test_script" ] && [ -x "$test_script" ]; then
        if ./"$test_script"; then
            print_success "$test_name completed successfully"
            echo ""
            return 0
        else
            print_error "$test_name FAILED!"
            return 1
        fi
    else
        print_warning "$test_script not found or not executable, skipping..."
        return 0
    fi
}

# Cleanup function
cleanup() {
    print_status "Cleaning up any remaining test artifacts..."
    if [ -d "test_build" ]; then
        rm -rf test_build
        print_status "Removed test_build directory"
    fi
}

# Set trap for cleanup on exit
trap cleanup EXIT

# Main execution
echo "=================================================="
echo "🧪 Running All Unit Tests for AndroidArabicWhisper"
echo "=================================================="
echo ""

# Array of test scripts in logical execution order
test_scripts=(
    "utils_tests.sh"
    "audio_decoder_tests.sh"
    "feature_extractor_tests.sh"
    "whisper_audio_tests.sh"
    "whisper_tokenizer_tests.sh"
    "whisper_model_tests.sh"
)

# Track test results
total_tests=0
passed_tests=0
failed_tests=0
skipped_tests=0

print_status "Found ${#test_scripts[@]} test suites to run"
echo ""

# Run each test
for test_script in "${test_scripts[@]}"; do
    ((total_tests++))

    if run_test "$test_script"; then
        ((passed_tests++))
    elif [ $? -eq 1 ]; then
        ((failed_tests++))
        print_error "Test suite $test_script failed! Stopping execution."
        echo ""
        echo "=================================================="
        echo "❌ TEST RUN TERMINATED DUE TO FAILURE"
        echo "=================================================="
        echo "📊 Results so far:"
        echo "   Total tests run: $total_tests"
        echo "   Passed: $passed_tests"
        echo "   Failed: $failed_tests"
        echo "   Skipped: $skipped_tests"
        echo ""
        print_error "Fix the failing test before running the full suite again."
        exit 1
    else
        ((skipped_tests++))
    fi
done

# Final results
echo "=================================================="
echo "✅ ALL TESTS COMPLETED SUCCESSFULLY!"
echo "=================================================="
echo "📊 Final Results:"
echo "   Total test suites: $total_tests"
echo "   Passed: $passed_tests"
echo "   Failed: $failed_tests"
echo "   Skipped: $skipped_tests"
echo ""

if [ $failed_tests -eq 0 ]; then
    print_success "🎉 All $passed_tests test suites passed!"
    echo ""
    print_status "Your AndroidArabicWhisper native code is working correctly!"
else
    print_error "Some tests failed. Please review the output above."
    exit 1
fi