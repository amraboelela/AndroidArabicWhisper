#!/bin/bash

# Script to build and run tokenizer unit tests
# Usage: ./whisper_tokenizer_tests.sh

set -e  # Exit on any error

echo "=== Building and Running Tokenizer Unit Tests ==="

# Navigate to cpp directory
cd "$(dirname "$0")"

# Create build directory
if [ ! -d "test_build" ]; then
    mkdir test_build
fi

cd test_build

echo "Configuring build with CMake..."
# Copy the test CMakeLists to be the main one for this build
cp ../whisper_tokenizer_tests.cmak ./CMakeLists.txt
cmake -DCMAKE_BUILD_TYPE=Release .

echo "Building test executable..."
make

echo "Running tokenizer tests..."
echo "================================"
./test_whisper_tokenizer

echo "================================"
echo "Test execution completed!"

# Optional: Run with CTest for more detailed output
echo ""
echo "Running with CTest for detailed results..."
make test

cd ..
rm -rf test_build

echo "Done!"