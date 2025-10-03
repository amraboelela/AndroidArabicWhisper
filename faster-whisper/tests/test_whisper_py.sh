#!/bin/bash

# Script to run Python whisper unit tests
# Usage: ./test_whisper_py.sh

set -e  # Exit on any error

echo "=== Running Python Whisper Unit Tests ==="

# Navigate to current directory
cd "$(dirname "$0")"

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH"
    exit 1
fi

echo "Python version:"
python3 --version

echo ""
echo "Running Python whisper tests..."
echo "================================"
python3 test_whisper.py

echo "================================"
echo "Test execution completed!"

echo "Done!"