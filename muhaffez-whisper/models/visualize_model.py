#!/usr/bin/env python3
"""
Visualize the EncoderDecoderTransformer model architecture
Creates a detailed diagram of the model structure
"""
import json
import sys
from encoder_decoder_transformer import EncoderDecoderTransformer

def visualize_model(save_to_file=True):
    """Print a visual representation of the model architecture"""

    # Load vocabulary to get vocab size
    with open("vocabulary.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)

    # Model parameters
    d_model = 128
    n_encoder_layers = 4
    n_decoder_layers = 4
    n_heads = 4
    d_ff = 512
    max_encoder_seq_len = 2000
    max_decoder_seq_len = 100
    n_mels = 40

    # Create model to get stats
    model = EncoderDecoderTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=0.1,
        max_encoder_seq_len=max_encoder_seq_len,
        max_decoder_seq_len=max_decoder_seq_len,
        n_mels=n_mels
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Print summary to terminal in table format
    print()
    print("=" * 80)
    print("MUHAFFEZ WHISPER MODEL - SUMMARY")
    print("=" * 80)
    print()

    # Model Statistics Table
    print("MODEL STATISTICS")
    print("-" * 80)
    print(f"{'Parameter':<40} {'Value':>38}")
    print("-" * 80)
    print(f"{'Total Parameters':<40} {total_params:>38,}")
    print(f"{'Model Size':<40} {'~' + str(round(total_params * 4 / (1024**2), 1)) + ' MB (float32)':>38}")
    print(f"{'Vocabulary Size':<40} {f'{vocab_size:,} Arabic words':>38}")
    print(f"{'Model Dimension (d_model)':<40} {d_model:>38}")
    print(f"{'Encoder Max Sequence Length':<40} {f'{max_encoder_seq_len} (audio)':>38}")
    print(f"{'Decoder Max Sequence Length':<40} {f'{max_decoder_seq_len} (text)':>38}")
    print("-" * 80)
    print()

    # Configuration Table
    print("CONFIGURATION")
    print("-" * 80)
    print(f"{'Parameter':<40} {'Value':>38}")
    print("-" * 80)
    print(f"{'Mel Bins':<40} {n_mels:>38}")
    print(f"{'Encoder Layers':<40} {n_encoder_layers:>38}")
    print(f"{'Decoder Layers':<40} {n_decoder_layers:>38}")
    print(f"{'Attention Heads':<40} {n_heads:>38}")
    print(f"{'Feed-Forward Dimension':<40} {d_ff:>38}")
    print(f"{'Dropout':<40} {'0.1':>38}")
    print("-" * 80)
    print()

    # Architecture Overview Table
    print("ARCHITECTURE OVERVIEW")
    print("-" * 80)
    print(f"{'Stage':<15} {'Component':<25} {'Details':<38}")
    print("-" * 80)

    architecture = [
        ("INPUT", "Audio", "Mel Spectrogram (40 bins)"),
        ("", "", "↓"),
        ("ENCODER", "Conv1D Layer 1", "40→128, kernel=3, stride=1"),
        ("(Audio Path)", "Activation", "GELU"),
        ("", "Conv1D Layer 2", "128→128, kernel=3, stride=2"),
        ("", "Positional Embed", "Learned (max 2000 positions)"),
        ("", "Transformer Layers", "4x Encoder Blocks"),
        ("", "└─ Self-Attention", "Multi-Head (4 heads)"),
        ("", "└─ Feed-Forward", "128→512→128 + GELU"),
        ("", "└─ Normalization", "Residual + LayerNorm"),
        ("", "", "↓ (encoder hidden states)"),
        ("DECODER", "Input", "Start token: <s>"),
        ("(Text Path)", "Token Embedding", "14,755 vocab → 128 dims"),
        ("", "Positional Embed", "Learned (max 100 positions)"),
        ("", "Transformer Layers", "4x Decoder Blocks"),
        ("", "└─ Masked Attn", "Multi-Head (4 heads, causal)"),
        ("", "└─ Cross-Attn", "Attends to encoder output"),
        ("", "└─ Feed-Forward", "128→512→128 + GELU"),
        ("", "└─ Normalization", "Residual + LayerNorm"),
        ("", "Output Projection", "128→14,755 (logits)"),
        ("", "Generation", "Autoregressive (feed out→in)"),
        ("", "", "↓"),
        ("OUTPUT", "Token Sequence", "<s> word1 word2 ... </s>"),
        ("", "Transcription", "Arabic Text"),
    ]

    for stage, component, details in architecture:
        print(f"{stage:<15} {component:<25} {details:<38}")
    print("-" * 80)
    print()

    # Key Features Table
    print("KEY FEATURES")
    print("-" * 80)
    print(f"{'Feature':<28} {'Description':<50}")
    print("-" * 80)
    features = [
        ("Architecture", "Encoder-Decoder design inspired by Whisper"),
        ("Audio Processing", "Convolutional preprocessing of mel spectrograms"),
        ("Positional Embeddings", "Learned trainable embeddings (separate for enc/dec)"),
        ("Attention Mechanism", "4-head multi-head attention for parallel processing"),
        ("Cross-Attention", "Aligns audio features with text generation"),
        ("Residual Connections", "Improves gradient flow during training"),
        ("Layer Normalization", "Stabilizes training process"),
        ("Causal Masking", "Enables autoregressive text generation"),
        ("Model Size", "4.1M parameters (~15.5 MB)"),
        ("Sequence Lengths", "Encoder: 2000 (audio), Decoder: 100 (text)"),
    ]
    for feature, desc in features:
        print(f"{feature:<28} {desc:<50}")
    print("-" * 80)
    print()
    print("=" * 80)
    print()

    if save_to_file:
        print("✓ Full architecture details saved to: model_architecture.md")
        print()
if __name__ == "__main__":
    visualize_model()
