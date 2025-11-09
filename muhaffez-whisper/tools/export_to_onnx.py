#!/usr/bin/env python3
"""
Export Muhaffez Whisper model to ONNX format for Android deployment
"""
import sys
import os
import torch
import json

sys.path.append(".")
from tools.encoder_decoder_transformer import EncoderDecoderTransformer

def export_encoder(model, output_path):
    """Export encoder to ONNX"""
    print("Exporting encoder...")

    # Create wrapper for encoder that handles variable-length input
    class EncoderWrapper(torch.nn.Module):
        def __init__(self, original_model):
            super().__init__()
            self.original_model = original_model

        def forward(self, mel_features):
            # The original encode function handles positional embeddings correctly
            return self.original_model.encode(mel_features)

    encoder_wrapper = EncoderWrapper(model)
    encoder_wrapper.eval()

    # Create dummy input with correct size: 30 seconds at 16kHz = 480000 samples
    # With hop_length=160, we get 480000/160 = 3000 frames
    # After conv2 (stride 2), we get 3000/2 = 1500 frames (which fits positional embedding)
    dummy_mel = torch.randn(1, 80, 3000)

    # Export encoder
    torch.onnx.export(
        encoder_wrapper,
        dummy_mel,
        output_path,
        input_names=["input_features"],
        output_names=["encoder_hidden_states"],
        dynamic_axes={
            "input_features": {2: "time"},  # Variable time dimension
            "encoder_hidden_states": {1: "time"}
        },
        opset_version=14,
        do_constant_folding=True
    )

    print(f"✓ Encoder exported to {output_path}")

def export_decoder(model, output_path):
    """Export decoder to ONNX without KV-caching (simpler for initial version)"""
    print("Exporting decoder (without KV-cache for simplicity)...")

    # Create wrapper that takes both inputs
    class DecoderWrapper(torch.nn.Module):
        def __init__(self, original_model):
            super().__init__()
            self.model = original_model

        def forward(self, input_ids, encoder_hidden_states):
            logits, _ = self.model.decode(input_ids, encoder_hidden_states, use_cache=False)
            return logits

    decoder_wrapper = DecoderWrapper(model)
    decoder_wrapper.eval()

    # Create dummy inputs
    # input_ids: (batch=1, seq_len)
    dummy_input_ids = torch.tensor([[1, 100, 200]], dtype=torch.long)
    # encoder_hidden_states: (batch=1, time=1500, d_model=128)
    dummy_encoder_hidden = torch.randn(1, 1500, 128)

    # Export decoder
    torch.onnx.export(
        decoder_wrapper,
        (dummy_input_ids, dummy_encoder_hidden),
        output_path,
        input_names=["input_ids", "encoder_hidden_states"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {1: "seq_len"},
            "encoder_hidden_states": {1: "time"},
            "logits": {1: "seq_len"}
        },
        opset_version=14,
        do_constant_folding=True
    )

    print(f"✓ Decoder exported to {output_path}")
    print("  Note: KV-caching not included in ONNX export (use_cache=False)")

def main():
    # Paths
    model_path = "models/muhaffez_whisper.pt"
    vocab_path = "models/vocabulary.json"
    output_dir = "models/onnx"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load vocabulary
    print("Loading vocabulary...")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"✓ Vocabulary loaded: {len(vocab)} words")

    # Create model
    print("Creating model...")
    model = EncoderDecoderTransformer(
        vocab_size=len(vocab),
        d_model=128,
        n_encoder_layers=4,
        n_decoder_layers=4,
        n_heads=4,
        d_ff=512,
        dropout=0.1
    )

    # Load trained weights
    print(f"Loading model from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print("✓ Model loaded successfully")

    # Export encoder
    encoder_path = os.path.join(output_dir, "encoder_model.onnx")
    export_encoder(model, encoder_path)

    # Export decoder
    decoder_path = os.path.join(output_dir, "decoder_model.onnx")
    export_decoder(model, decoder_path)

    # Copy vocabulary
    import shutil
    vocab_output = os.path.join(output_dir, "vocabulary.json")
    shutil.copy(vocab_path, vocab_output)
    print(f"✓ Vocabulary copied to {vocab_output}")

    print("\n" + "="*60)
    print("✓ ONNX Export Complete!")
    print("="*60)
    print(f"Encoder: {encoder_path}")
    print(f"Decoder: {decoder_path}")
    print(f"Vocabulary: {vocab_output}")
    print("\nNext steps:")
    print(f"1. Copy files from {output_dir}/ to Android assets:")
    print(f"   app/src/main/assets/muhaffez_whisper/")
    print("2. The MuhaffezWhisperHelper.kt is already configured to use these models")

if __name__ == "__main__":
    main()
