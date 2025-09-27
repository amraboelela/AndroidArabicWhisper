#!/bin/bash

# Script to build and run audio processing unit tests
# Usage: ./audio_decoder_tests.sh

set -e  # Exit on any error

echo "=== Building and Running Audio Processing Unit Tests ==="

# Navigate to cpp directory
cd "$(dirname "$0")"

# Create build directory
if [ ! -d "audio_test_build" ]; then
    mkdir audio_test_build
fi

cd audio_test_build

echo "Configuring build with CMake..."
# Copy the test CMakeLists to be the main one for this build
cp ../cmak_lists/audio_decoder_tests.cmak ./CMakeLists.txt
cmake -DCMAKE_BUILD_TYPE=Release .

echo "Building test executable..."
make

echo "Running audio processing tests..."
echo "================================"
./test_audio_decoder

echo "================================"
echo "Test execution completed!"

# Optional: Run with CTest for more detailed output
echo ""
echo "Running with CTest for detailed results..."
make test

cd ..
rm -rf audio_test_build

echo "Done!"