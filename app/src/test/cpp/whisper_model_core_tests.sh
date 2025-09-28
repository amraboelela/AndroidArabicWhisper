#!/bin/bash

# WhisperModel Core Unit Tests
# Tests constructor, basic functionality, and main transcribe entry point
# Created by Amr Aboelela

set -e  # Exit on any error

echo "========================================"
echo "WHISPERMODEL CORE UNIT TESTS"
echo "Testing core functionality in whisper_model_core.cpp"
echo "========================================"

# Navigate to cpp directory
cd "$(dirname "$0")"

# Create build directory
if [ ! -d "core_test_build" ]; then
    mkdir core_test_build
fi

cd core_test_build

echo "📋 Core Test Coverage Overview:"
echo "   • WhisperModel Constructor and initialization"
echo "   • supported_languages() - language support validation"
echo "   • get_feature_kwargs() - configuration parsing"
echo "   • transcribe() - complete workflow testing"
echo "   • encode() - feature encoding for CTranslate2"
echo "   • detect_language() - language detection"
echo "   • Error handling and parameter validation"
echo "   • Arabic language support validation"
echo "   • Duration calculation and feature extraction integration"
echo ""

echo "Configuring build with CMake..."
# Copy the core test CMakeLists to be the main one for this build
cp ../cmak_lists/whisper_model_core_tests.cmak ./CMakeLists.txt
cmake -DCMAKE_BUILD_TYPE=Release .

echo ""
echo "Building core test executable..."
make

echo ""
echo "🚀 Running WhisperModel core tests..."
echo "================================"
./test_whisper_model_core

echo ""
echo "================================"
echo "📊 Core test execution completed!"

# Optional: Run with CTest for more detailed output
echo ""
echo "Running with CTest for detailed results..."
make test

echo ""
echo "========================================"
echo "✅ CORE TESTING COMPLETE"
echo "🎯 Core Functions Tested: 6"
echo "🔧 Constructor & Basic Methods: Validated"
echo "🌐 Arabic Language Support: Tested"
echo "🎵 Audio Processing Integration: Verified"
echo ""
echo "🏆 Core functionality in whisper_model_core.cpp is tested!"
echo "========================================"

cd ..
rm -rf core_test_build

echo "Core tests done!"