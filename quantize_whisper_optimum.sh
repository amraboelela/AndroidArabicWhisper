#!/bin/bash
# Quantize Whisper model using Optimum CLI for better mobile performance

set -e

MODEL_PATH="onnx/models/custom-whisper-ar-quran"
OUTPUT_INT8="onnx/models/custom-whisper-ar-quran-onnx-int8"
OUTPUT_FP16="onnx/models/custom-whisper-ar-quran-onnx-fp16"

echo "============================================================"
echo "🚀 Quantizing Whisper Model with Optimum"
echo "============================================================"
echo "Source model: $MODEL_PATH"
echo ""

# Create output directories
mkdir -p "$OUTPUT_INT8"
mkdir -p "$OUTPUT_FP16"

# Export to ONNX with INT8 quantization (dynamic)
echo "📦 Step 1: Exporting to ONNX with INT8 quantization..."
echo "This will take several minutes..."
echo ""

optimum-cli export onnx \
  --model "$MODEL_PATH" \
  --task automatic-speech-recognition \
  --optimize O2 \
  --quantize avx2 \
  "$OUTPUT_INT8"

if [ $? -eq 0 ]; then
    echo "✅ INT8 quantization complete!"
    echo "   Output: $OUTPUT_INT8"

    # Show file sizes
    echo ""
    echo "📊 INT8 Model sizes:"
    du -sh "$OUTPUT_INT8"/*.onnx 2>/dev/null || echo "   (models in subdirectories)"
    ls -lh "$OUTPUT_INT8"/*.onnx 2>/dev/null || find "$OUTPUT_INT8" -name "*.onnx" -exec ls -lh {} \;
else
    echo "❌ INT8 quantization failed"
fi

echo ""
echo "============================================================"
echo "📦 Step 2: Exporting to ONNX with FP16 (optional)"
echo "============================================================"
echo ""

# Export to ONNX with FP16
optimum-cli export onnx \
  --model "$MODEL_PATH" \
  --task automatic-speech-recognition \
  --optimize O2 \
  --fp16 \
  "$OUTPUT_FP16"

if [ $? -eq 0 ]; then
    echo "✅ FP16 conversion complete!"
    echo "   Output: $OUTPUT_FP16"

    # Show file sizes
    echo ""
    echo "📊 FP16 Model sizes:"
    du -sh "$OUTPUT_FP16"/*.onnx 2>/dev/null || echo "   (models in subdirectories)"
    ls -lh "$OUTPUT_FP16"/*.onnx 2>/dev/null || find "$OUTPUT_FP16" -name "*.onnx" -exec ls -lh {} \;
else
    echo "❌ FP16 conversion failed"
fi

echo ""
echo "============================================================"
echo "✅ Quantization complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. Copy quantized models to app/src/main/assets/whisper_onnx/"
echo "2. Build and install: ./gradlew assembleDebug"
echo "3. Test on device and compare performance"
echo ""
echo "Choose the format with best speed/accuracy tradeoff:"
echo "  - INT8: Faster, smaller, slightly lower accuracy"
echo "  - FP16: Moderate speed, good accuracy"
echo ""
