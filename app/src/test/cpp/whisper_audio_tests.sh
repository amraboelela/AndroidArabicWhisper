#!/bin/bash

# Script to build and run whisper audio integration test

set -e  # Exit on any error

# Cleanup function that runs on exit (success or failure)
cleanup() {
    echo "Cleaning up test_build directory..."
    if [ -d "test_build" ]; then
        rm -rf test_build
        echo "✅ test_build directory removed"
    fi
}

# Set trap to run cleanup on script exit
trap cleanup EXIT
echo "=== Building and Running Whisper Audio Integration Test ==="

# Navigate to cpp directory
cd "$(dirname "$0")"

# Create build directory
if [ ! -d "test_build" ]; then
    mkdir test_build
fi

cd test_build

echo "Configuring build with CMake..."
# Copy the test CMakeLists to be the main one for this build
cp ../cmak_lists/whisper_audio_tests.cmak ./CMakeLists.txt
cmake -DCMAKE_BUILD_TYPE=Release .

echo "Building test executable..."
make

echo "Running integration test..."
echo "================================"
./test_whisper_audio

echo "================================"
echo "Test execution completed!"

# Optional: Run with CTest for more detailed output
echo ""
echo "Running with CTest for detailed results..."
make test

cd ..

echo "Done!"