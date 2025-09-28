#!/bin/bash

# WhisperModel Utils Unit Tests
# Tests helper functions for feature processing, compression, and timestamps
# Created by Amr Aboelela

set -e  # Exit on any error

echo "========================================"
echo "WHISPERMODEL UTILS UNIT TESTS"
echo "Testing utility functions in whisper_model_utils.cpp"
echo "========================================"

# Navigate to cpp directory
cd "$(dirname "$0")"

# Create build directory
if [ ! -d "utils_test_build" ]; then
    mkdir utils_test_build
fi

cd utils_test_build

echo "📋 Utils Test Coverage Overview:"
echo "   • slice_features() - feature slicing helper"
echo "   • pad_or_trim() - feature padding/trimming"
echo "   • get_ctranslate2_storage() - CTranslate2 conversion"
echo "   • get_compression_ratio() - text compression analysis"
echo "   • merge_punctuations() - punctuation handling"
echo "   • restore_speech_timestamps() - timestamp restoration"
echo "   • normalize_features() - feature normalization"
echo "   • apply_log_mel_spectrogram() - log mel processing"
echo "   • calculate_signal_to_noise_ratio() - audio analysis"
echo "   • is_silent_segment() - silence detection"
echo "   • detect_speech_activity() - speech activity detection"
echo "   • Integration and pipeline testing"
echo ""

echo "Configuring build with CMake..."
# Copy the utils test CMakeLists to be the main one for this build
cp ../cmak_lists/whisper_model_utils_tests.cmak ./CMakeLists.txt
cmake -DCMAKE_BUILD_TYPE=Release .

echo ""
echo "Building utils test executable..."
make

echo ""
echo "🚀 Running WhisperModel utils tests..."
echo "================================"
./test_whisper_model_utils

echo ""
echo "================================"
echo "📊 Utils test execution completed!"

# Optional: Run with CTest for more detailed output
echo ""
echo "Running with CTest for detailed results..."
make test

echo ""
echo "========================================"
echo "✅ UTILS TESTING COMPLETE"
echo "🎯 Utility Functions Tested: 11"
echo "🔧 Feature Processing: Validated"
echo "📊 Audio Analysis: Tested"
echo "⚙️  Helper Functions: Verified"
echo "🔄 Integration Pipeline: Tested"
echo ""
echo "🏆 Utility functions in whisper_model_utils.cpp are tested!"
echo "========================================"

cd ..
rm -rf utils_test_build

echo "Utils tests done!"