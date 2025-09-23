#!/bin/bash

# Script to build and run whisper audio integration test
# Usage: ./run_test.sh

set -e  # Exit on any error

echo "=== Building and Running Whisper Audio Integration Test ==="

# Navigate to cpp directory
cd "$(dirname "$0")"

# Create build directory
if [ ! -d "test_build" ]; then
    mkdir test_build
fi

cd test_build

echo "Configuring build with CMake..."
cmake -f ../CMakeLists_test.txt ..

echo "Building test executable..."
make

echo "Running integration test..."
echo "================================"
./test_whisper_audio_integration

echo "================================"
echo "Test execution completed!"

# Optional: Run with CTest for more detailed output
echo ""
echo "Running with CTest for detailed results..."
make test

echo "Done!"