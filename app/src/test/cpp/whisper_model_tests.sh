#!/bin/bash

# Script to build and run WhisperModel unit tests
# Usage: ./whisper_model_tests.sh

set -e  # Exit on any error

echo "=== Building and Running WhisperModel Unit Tests ==="

# Navigate to cpp directory
cd "$(dirname "$0")"

# Create build directory
if [ ! -d "model_test_build" ]; then
    mkdir model_test_build
fi

cd model_test_build

echo "Configuring build with CMake..."
# Copy the test CMakeLists to be the main one for this build
cp ../cmak_lists/whisper_model_tests.cmak ./CMakeLists.txt
cmake -DCMAKE_BUILD_TYPE=Release .

echo "Building test executable..."
make

echo "Running WhisperModel tests..."
echo "================================"
./test_whisper_model

echo "================================"
echo "Test execution completed!"

# Optional: Run with CTest for more detailed output
echo ""
echo "Running with CTest for detailed results..."
make test

cd ..
rm -rf model_test_build

echo "Done!"