#!/bin/bash

# Script to build and run utils unit tests
# Usage: ./utils_tests.sh

set -e  # Exit on any error

echo "=== Building and Running Utils Unit Tests ==="

# Navigate to cpp directory
cd "$(dirname "$0")"

# Create build directory
if [ ! -d "utils_test_build" ]; then
    mkdir utils_test_build
fi

cd utils_test_build

echo "Configuring build with CMake..."
# Copy the test CMakeLists to be the main one for this build
cp ../utils_tests.cmak ./CMakeLists.txt
cmake -DCMAKE_BUILD_TYPE=Release .

echo "Building test executable..."
make

echo "Running utils tests..."
echo "================================"
./test_utils

echo "================================"
echo "Test execution completed!"

# Optional: Run with CTest for more detailed output
echo ""
echo "Running with CTest for detailed results..."
make test

cd ..
rm -rf utils_test_build

echo "Done!"