#!/bin/bash

# Flexible test runner that can test different audio files
# Usage: ./run_audio_test.sh [filename]
# Example: ./run_audio_test.sh 002-01.wav
#          ./run_audio_test.sh 001.wav
#          ./run_audio_test.sh test.wav

set -e

AUDIO_FILE="${1:-002-01.wav}"  # Default to 002-01.wav if no argument provided

echo "=== Whisper Audio Integration Test ==="
echo "Testing with audio file: $AUDIO_FILE"
echo "======================================="

cd "$(dirname "$0")"

# Check if the specified audio file exists
if [ ! -f "../assets/$AUDIO_FILE" ]; then
    echo "⚠ Warning: Audio file ../assets/$AUDIO_FILE not found"
    echo "Available files in assets:"
    ls -la ../assets/*.wav ../assets/*.m4a 2>/dev/null || echo "No audio files found"
    echo ""
    echo "Proceeding with test (will fall back to synthetic audio)..."
fi

# Try to compile the test with minimal dependencies
echo "Compiling integration test..."

# Check if we have required headers
if [ ! -f "whisper/whisper_audio.h" ]; then
    echo "Error: Missing whisper_audio.h - please ensure whisper directory exists"
    exit 1
fi

if [ ! -f "include/feature_extractor.h" ]; then
    echo "Error: Missing feature_extractor.h - please ensure include directory exists"
    exit 1
fi

# Create a temporary main file that tests the specific audio file
cat > temp_test_main.cpp << EOF
#include "whisper_audio.h"
#include "feature_extractor.h"
#include "audio.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>  // For M_PI
#include <iomanip>  // For std::setprecision

// Include just the function declarations from test_integration.cpp
void test_whisper_audio_integration(const std::string& audio_filename = "002-01.wav");

int main() {
    // Test with the specified audio file
    test_whisper_audio_integration("$AUDIO_FILE");

    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "Single file test completed!" << std::endl;
    std::cout << "File tested: $AUDIO_FILE" << std::endl;
    std::cout << std::string(50, '=') << std::endl;

    return 0;
}
EOF

# Compile WITHOUT TESTING_MODE flag and include the test implementation
g++ -std=c++17 \
    -I./include \
    -I./whisper \
    -DTESTING_MODE \
    temp_test_main.cpp \
    whisper/test_integration.cpp \
    whisper/whisper_audio.cpp \
    feature_extractor.cpp \
    audio.cpp \
    utils.cpp \
    -lz -lm \
    -o test_audio_file

echo "Running integration test with $AUDIO_FILE..."
echo "================================"
./test_audio_file
echo "================================"
echo "Test completed!"

# Clean up
rm -f test_audio_file temp_test_main.cpp

echo ""
echo "Usage examples:"
echo "  ./run_audio_test.sh 002-01.wav  # Test large file"
echo "  ./run_audio_test.sh 001.wav     # Test medium file"
echo "  ./run_audio_test.sh test.wav    # Test small file"
echo "  ./run_audio_test.sh             # Default (002-01.wav)"

echo "Done!"