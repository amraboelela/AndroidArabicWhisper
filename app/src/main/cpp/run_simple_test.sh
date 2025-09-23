#!/bin/bash

# Simple test runner that compiles and runs the integration test
# This version tries to work with minimal dependencies

set -e

echo "=== Simple Whisper Audio Integration Test ==="

cd "$(dirname "$0")"

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

# Compile WITHOUT TESTING_MODE flag so main function is included
g++ -std=c++17 \
    -I./include \
    -I./whisper \
    whisper/test_integration.cpp \
    whisper/whisper_audio.cpp \
    feature_extractor.cpp \
    audio.cpp \
    utils.cpp \
    -lz -lm \
    -o test_whisper_audio_integration_simple

echo "Running integration test..."
echo "================================"
./test_whisper_audio_integration_simple
echo "================================"
echo "Test completed!"

# Clean up
rm -f test_whisper_audio_integration_simple

echo "Done!"