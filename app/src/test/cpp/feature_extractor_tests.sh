#!/bin/bash

# Script to build and run feature extractor unit tests
# Usage: ./feature_extractor_tests.sh

set -e  # Exit on any error

echo "=== Building and Running Feature Extractor Unit Tests ==="

# Navigate to cpp directory
cd "$(dirname "$0")"

# Create build directory
if [ ! -d "feature_test_build" ]; then
    mkdir feature_test_build
fi

cd feature_test_build

echo "Configuring build with CMake..."
# Copy the test CMakeLists to be the main one for this build
cp ../cmak_lists/feature_extractor_tests.cmak ./CMakeLists.txt
cmake -DCMAKE_BUILD_TYPE=Release .

echo "Building test executable..."
make

echo "Running feature extractor tests..."
echo "================================"
./test_feature_extractor

echo "================================"
echo "Test execution completed!"

# Optional: Run with CTest for more detailed output
echo ""
echo "Running with CTest for detailed results..."
make test

cd ..
rm -rf feature_test_build

echo "Done!"