#!/bin/bash

# WhisperModel Segments Unit Tests
# Tests segment generation, splitting, and word-level timestamp functions
# Created by Amr Aboelela

set -e  # Exit on any error

echo "========================================"
echo "WHISPERMODEL SEGMENTS UNIT TESTS"
echo "Testing segment processing in whisper_model_segments.cpp"
echo "========================================"

# Navigate to cpp directory
cd "$(dirname "$0")"

# Create build directory
if [ ! -d "segments_test_build" ]; then
    mkdir segments_test_build
fi

cd segments_test_build

echo "📋 Segments Test Coverage Overview:"
echo "   • split_segments_by_timestamps() - segment processing"
echo "   • generate_segments() - segment generation logic"
echo "   • generate_with_fallback() - temperature fallback"
echo "   • get_prompt() - prompt construction"
echo "   • generate_word_timestamps() - Arabic word timing"
echo "   • add_word_timestamps() - word-level timing"
echo "   • find_alignment() - token-to-word alignment"
echo "   • Transcription options validation"
echo "   • Arabic word processing and timestamps"
echo "   • Edge cases and error handling"
echo ""

echo "Configuring build with CMake..."
# Copy the segments test CMakeLists to be the main one for this build
cp ../cmak_lists/whisper_model_segments_tests.cmak ./CMakeLists.txt
cmake -DCMAKE_BUILD_TYPE=Release .

echo ""
echo "Building segments test executable..."
make

echo ""
echo "🚀 Running WhisperModel segments tests..."
echo "================================"
./test_whisper_model_segments

echo ""
echo "================================"
echo "📊 Segments test execution completed!"

# Optional: Run with CTest for more detailed output
echo ""
echo "Running with CTest for detailed results..."
make test

echo ""
echo "========================================"
echo "✅ SEGMENTS TESTING COMPLETE"
echo "🎯 Segment Functions Tested: 7"
echo "🔧 Segment Processing: Validated"
echo "⚡ Temperature Fallback: Tested"
echo "🌐 Arabic Word Timestamps: Verified"
echo "📝 Transcription Options: Validated"
echo ""
echo "🏆 Segment processing in whisper_model_segments.cpp is tested!"
echo "========================================"

cd ..
rm -rf segments_test_build

echo "Segments tests done!"