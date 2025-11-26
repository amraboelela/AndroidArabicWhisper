#!/bin/bash

# Reset learning rates to default (1e-3) for all training types
# Usage: ./reset_lr.sh

MODEL_DIR="../models"
LR_FILE="$MODEL_DIR/muhaffez_whisper_lr.json"

echo "Resetting learning rates to 1e-3..."

# Create or overwrite the LR file with default values
cat > "$LR_FILE" << EOF
{
  "full": 0.001,
  "augmented": 0.001,
  "curriculum": 0.001
}
EOF

if [ $? -eq 0 ]; then
    echo "✓ Learning rates reset successfully!"
    echo "  - full: 1.0e-03"
    echo "  - augmented: 1.0e-03"
    echo "  - curriculum: 1.0e-03"
    echo ""
    echo "File: $LR_FILE"
else
    echo "❌ Error: Failed to reset learning rates"
    exit 1
fi
