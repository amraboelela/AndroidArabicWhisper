#!/bin/bash

# Script to convert 4-space indentation to 2-space indentation for Android Studio compliance
# Usage: ./fix_indentation.sh

set -e

echo "=== Converting C++ files to 2-space indentation ==="

# List of files to convert (those that likely have 4-space indentation)
FILES=(
    "whisper_model.cpp"
    "include/whisper_model.h"
    "tokenizer.cpp"
    "include/tokenizer.h"
    "audio.cpp"
    "include/audio.h"
    "whisper/whisper_tokenizer.cpp"
    "whisper/whisper_tokenizer.h"
    "whisper/whisper_audio.cpp"
    "whisper/whisper_audio.h"
)

# Function to convert indentation in a file
convert_indentation() {
    local file="$1"
    if [ -f "$file" ]; then
        echo "Converting indentation in $file..."
        # Create backup
        cp "$file" "$file.backup"

        # Convert 4 spaces to 2 spaces at the beginning of lines
        sed 's/^    /  /g' "$file.backup" > "$file.tmp"

        # Handle deeper indentation levels (8 spaces -> 4 spaces, 12 spaces -> 6 spaces, etc.)
        sed 's/^        /    /g' "$file.tmp" > "$file.tmp2"
        sed 's/^            /      /g' "$file.tmp2" > "$file.tmp3"
        sed 's/^                /        /g' "$file.tmp3" > "$file.tmp4"
        sed 's/^                    /          /g' "$file.tmp4" > "$file.tmp5"

        # Move final result back
        mv "$file.tmp5" "$file"

        # Clean up temporary files
        rm -f "$file.tmp" "$file.tmp2" "$file.tmp3" "$file.tmp4" "$file.backup"

        echo "✓ Converted $file"
    else
        echo "⚠ File not found: $file"
    fi
}

# Convert each file
for file in "${FILES[@]}"; do
    convert_indentation "$file"
done

echo "=== Indentation conversion completed ==="
echo "All files now use 2-space indentation consistent with Android Studio guidelines"