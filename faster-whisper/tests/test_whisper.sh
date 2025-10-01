#!/bin/bash

# Script to build and run whisper unit tests
# Usage: ./test_whisper.sh

set -e  # Exit on any error

echo "=== Building and Running Whisper Unit Tests ==="

# Navigate to current directory
cd "$(dirname "$0")"

rm -rf test_build

# Create build directory
if [ ! -d "test_build" ]; then
    mkdir test_build
fi

cd test_build

echo "Configuring build with CMake..."
# Copy the test CMakeLists to be the main one for this build
cp ../cmake_lists/test_whisper.cmake ./CMakeLists.txt
cmake -DCMAKE_BUILD_TYPE=Release .

echo "Building test executable..."
make

echo "Running whisper tests..."
echo "================================"
./bin/test_whisper

echo "================================"
echo "Test execution completed!"

cd ..

echo "Done!"