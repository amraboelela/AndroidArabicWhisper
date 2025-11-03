#!/usr/bin/env python3
"""
Create a simplified Whisper model that's ONNX-friendly
Remove complex attention masks and use simpler operations
"""

import os
import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import warnings

class SimplifiedWhisperDecoder(nn.Module):
    """Simplified decoder that avoids complex mask operations"""
    def __init__(self, original_decoder, lm_head):
        super().__init__()
        self.embed_tokens = original_decoder.embed_tokens
        self.embed_positions = original_decoder.embed_positions
        self.layers = original_decoder.layers
        self.layer_norm = original_decoder.layer_norm
        self.lm_head = lm_head

    def forward(self, input_ids, encoder_hidden_states):
        # Simple embeddings
        inputs_embeds = self.embed_tokens(input_ids)

        # Position embeddings - simplified
        seq_length = input_ids.shape[1]
        positions = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        position_embeds = self.embed_positions(positions)

        hidden_states = inputs_embeds + position_embeds

        # Create simple causal mask (no complex operations)
        batch_size, seq_len = input_ids.shape
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device) * float('-inf'),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

        # Pass through decoder layers with simplified attention
        for layer in self.layers:
            # Self-attention with causal mask
            residual = hidden_states

            # Layer norm
            hidden_states = layer.self_attn_layer_norm(hidden_states)

            # Self attention (simplified - no complex masking)
            attn_output = layer.self_attn(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
            )
            # Handle tuple or tensor return
            if isinstance(attn_output, tuple):
                hidden_states = attn_output[0]
            else:
                hidden_states = attn_output
            hidden_states = residual + hidden_states

            # Cross-attention with encoder
            residual = hidden_states
            hidden_states = layer.encoder_attn_layer_norm(hidden_states)

            attn_output = layer.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
            )
            # Handle tuple or tensor return
            if isinstance(attn_output, tuple):
                hidden_states = attn_output[0]
            else:
                hidden_states = attn_output
            hidden_states = residual + hidden_states

            # FFN
            residual = hidden_states
            hidden_states = layer.final_layer_norm(hidden_states)
            hidden_states = layer.fc1(hidden_states)
            hidden_states = layer.activation_fn(hidden_states)
            hidden_states = layer.fc2(hidden_states)
            hidden_states = residual + hidden_states

        # Final layer norm
        hidden_states = self.layer_norm(hidden_states)

        # LM head
        logits = self.lm_head(hidden_states)

        return logits

def create_simplified_model(original_model):
    """Create simplified version of Whisper model"""
    print("Creating simplified Whisper model...")

    # Keep encoder as-is (it's simpler)
    encoder = original_model.model.encoder

    # Create simplified decoder
    decoder = SimplifiedWhisperDecoder(
        original_model.model.decoder,
        original_model.proj_out
    )

    print("  ✓ Simplified decoder created")

    return encoder, decoder

def export_simplified_to_onnx(encoder, decoder, output_dir):
    """Export simplified model to ONNX"""
    print("\nExporting simplified model to ONNX...")

    os.makedirs(output_dir, exist_ok=True)

    # Export encoder
    print("\n  Exporting encoder...")
    encoder.eval()

    batch_size = 1
    mel_bins = 80
    sequence_length = 3000

    dummy_input = torch.randn(batch_size, mel_bins, sequence_length)

    encoder_path = os.path.join(output_dir, "encoder_model.onnx")

    torch.onnx.export(
        encoder,
        dummy_input,
        encoder_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_features'],
        output_names=['last_hidden_state'],
        dynamic_axes={
            'input_features': {0: 'batch', 2: 'sequence'},
            'last_hidden_state': {0: 'batch', 1: 'sequence'}
        },
        verbose=False
    )

    size_mb = os.path.getsize(encoder_path) / (1024 * 1024)
    print(f"    ✓ Encoder: {size_mb:.2f} MB")

    # Export decoder
    print("\n  Exporting simplified decoder...")
    decoder.eval()

    decoder_seq_length = 10
    encoder_seq_length = 1500
    hidden_size = 512

    dummy_input_ids = torch.randint(0, 51865, (batch_size, decoder_seq_length))
    dummy_encoder_hidden = torch.randn(batch_size, encoder_seq_length, hidden_size)

    decoder_path = os.path.join(output_dir, "decoder_model.onnx")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            decoder,
            (dummy_input_ids, dummy_encoder_hidden),
            decoder_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input_ids', 'encoder_hidden_states'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch', 1: 'sequence'},
                'encoder_hidden_states': {0: 'batch'},
                'logits': {0: 'batch', 1: 'sequence'}
            },
            verbose=False
        )

    size_mb = os.path.getsize(decoder_path) / (1024 * 1024)
    print(f"    ✓ Decoder: {size_mb:.2f} MB")

    return encoder_path, decoder_path

def main():
    print("="*70)
    print("Creating Simplified ONNX-Compatible Whisper Model")
    print("="*70)

    model_path = "models/custom-whisper-ar-quran"
    output_dir = "models/custom-whisper-ar-quran-onnx-simplified"

    # Load original model
    print(f"\nLoading original model from {model_path}...")
    original_model = WhisperForConditionalGeneration.from_pretrained(model_path)
    processor = WhisperProcessor.from_pretrained(model_path)
    print("  ✓ Model loaded")

    # Create simplified version
    print("\n" + "="*70)
    print("Step 1: Creating Simplified Model")
    print("="*70)
    encoder, decoder = create_simplified_model(original_model)

    # Export to ONNX
    print("\n" + "="*70)
    print("Step 2: Exporting to ONNX")
    print("="*70)

    try:
        encoder_path, decoder_path = export_simplified_to_onnx(encoder, decoder, output_dir)

        # Copy tokenizer files
        print("\n" + "="*70)
        print("Step 3: Copying tokenizer files")
        print("="*70)
        import shutil
        for file in ["config.json", "generation_config.json", "tokenizer_config.json",
                     "vocab.json", "merges.txt", "normalizer.json",
                     "special_tokens_map.json", "added_tokens.json",
                     "preprocessor_config.json", "tokenizer.json"]:
            src = os.path.join(model_path, file)
            if os.path.exists(src):
                dst = os.path.join(output_dir, file)
                shutil.copy(src, dst)
                print(f"  ✓ {file}")

        # Summary
        print("\n" + "="*70)
        print("✓ Simplified Model Exported Successfully!")
        print("="*70)
        print(f"\nSaved to: {output_dir}/")
        print("\nFiles:")
        for file in sorted(os.listdir(output_dir)):
            file_path = os.path.join(output_dir, file)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  - {file} ({size_mb:.2f} MB)")

        print("\nNote: Simplified decoder with basic causal masking")
        print("      Should work with ONNX Runtime")

    except Exception as e:
        print(f"\n✗ Export failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
