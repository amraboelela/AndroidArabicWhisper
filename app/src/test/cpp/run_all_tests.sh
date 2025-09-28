#!/bin/bash

# Master test runner for all Android Arabic Whisper unit tests
# Usage: ./run_all_tests.sh

set -e  # Exit on any error

echo "========================================================"
echo "=== Android Arabic Whisper - Complete Test Suite ==="
echo "========================================================"

# Track test results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to run a test suite
run_test_suite() {
    local test_name="$1"
    local test_script="$2"

    echo ""
    echo "▶️  Running $test_name..."
    echo "=========================================="

    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    if ./"$test_script"; then
        echo "✅ $test_name: PASSED"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo "❌ $test_name: FAILED"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi

    echo ""
}

# Navigate to test directory
cd "$(dirname "$0")"

# Run all test suites
echo "Starting complete test suite execution..."
echo ""

run_test_suite "Whisper Tokenizer Tests" "whisper_tokenizer_tests.sh"
run_test_suite "Whisper Audio Tests" "whisper_audio_tests.sh"
run_test_suite "WhisperModel Core Tests" "whisper_model_core_tests.sh"
run_test_suite "WhisperModel Segments Tests" "whisper_model_segments_tests.sh"
run_test_suite "WhisperModel Utils Tests" "whisper_model_utils_tests.sh"
run_test_suite "Utils Tests" "utils_tests.sh"
run_test_suite "Feature Extractor Tests" "feature_extractor_tests.sh"
run_test_suite "Audio Processing Tests" "audio_decoder_tests.sh"

# Final summary
echo "========================================================"
echo "=== COMPLETE TEST SUITE SUMMARY ==="
echo "========================================================"
echo "Total Test Suites: $TOTAL_TESTS"
echo "Passed: $PASSED_TESTS"
echo "Failed: $FAILED_TESTS"
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    echo "🎉 ALL TEST SUITES PASSED!"
    echo ""
    echo "Your Android Arabic Whisper project has:"
    echo "✅ Working tokenizer with Arabic support"
    echo "✅ Functional audio processing pipeline"
    echo "✅ Proper WhisperModel core functionality"
    echo "✅ Robust segment processing and timestamps"
    echo "✅ Comprehensive utility functions"
    echo "✅ Reliable helper utilities"
    echo "✅ Robust feature extraction"
    echo "✅ Comprehensive audio handling"
    echo ""
    echo "The codebase is ready for production use!"
else
    echo "⚠️  $FAILED_TESTS TEST SUITE(S) FAILED"
    echo "Please review the failed tests and fix any issues."
    exit 1
fi

echo "========================================================"
echo "Test execution completed at $(date)"
echo "========================================================"