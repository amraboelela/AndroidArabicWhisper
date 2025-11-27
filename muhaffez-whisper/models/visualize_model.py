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

    # Capture output
    output_lines = []

    def print_line(text=""):
        """Print to console and capture for file"""
        print(text)
        output_lines.append(text)

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
    max_seq_len = 2000
    n_mels = 40

    print("=" * 80)
    print("MUHAFFEZ WHISPER MODEL ARCHITECTURE")
    print("=" * 80)
    print()

    # Model statistics
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

    print(f"Model Statistics:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Model Size: ~{total_params * 4 / (1024**2):.1f} MB (float32)")
    print()

    # Configuration
    print(f"Configuration:")
    print(f"  Vocabulary Size: {vocab_size:,}")
    print(f"  Model Dimension (d_model): {d_model}")
    print(f"  Max Sequence Length: {max_seq_len}")
    print(f"  Mel Bins: {n_mels}")
    print(f"  Encoder Layers: {n_encoder_layers}")
    print(f"  Decoder Layers: {n_decoder_layers}")
    print(f"  Attention Heads: {n_heads}")
    print(f"  Feed-Forward Dimension: {d_ff}")
    print()

    print("=" * 80)
    print("ARCHITECTURE DIAGRAM")
    print("=" * 80)
    print()

    # Input
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│                         INPUT                               │")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│  Audio: Mel Spectrogram (batch, n_mels=40, time_steps)     │")
    print("│  Text:  Token IDs (batch, seq_len)                         │")
    print("└─────────────────────────────────────────────────────────────┘")
    print("                          │")
    print("                          ▼")
    print()

    # Encoder
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│                    ENCODER (Audio Path)                     │")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│                                                             │")
    print("│  ┌───────────────────────────────────────────────────────┐ │")
    print("│  │ Conv1D (kernel=3, stride=1, padding=1)               │ │")
    print("│  │   Input:  (batch, 40, time)                          │ │")
    print("│  │   Output: (batch, 128, time)                         │ │")
    print("│  └───────────────────────────────────────────────────────┘ │")
    print("│                          │                                  │")
    print("│                          ▼                                  │")
    print("│  ┌───────────────────────────────────────────────────────┐ │")
    print("│  │ GELU Activation                                       │ │")
    print("│  └───────────────────────────────────────────────────────┘ │")
    print("│                          │                                  │")
    print("│                          ▼                                  │")
    print("│  ┌───────────────────────────────────────────────────────┐ │")
    print("│  │ Conv1D (kernel=3, stride=2, padding=1)               │ │")
    print("│  │   Output: (batch, 128, time//2)                      │ │")
    print("│  └───────────────────────────────────────────────────────┘ │")
    print("│                          │                                  │")
    print("│                          ▼                                  │")
    print("│  ┌───────────────────────────────────────────────────────┐ │")
    print("│  │ GELU + Transpose                                      │ │")
    print("│  │   Output: (batch, seq_len, 128)                      │ │")
    print("│  └───────────────────────────────────────────────────────┘ │")
    print("│                          │                                  │")
    print("│                          ▼                                  │")
    print("│  ┌───────────────────────────────────────────────────────┐ │")
    print("│  │ Positional Embedding                                  │ │")
    print("│  │   Shape: (2000, 128)                                  │ │")
    print("│  └───────────────────────────────────────────────────────┘ │")
    print("│                          │                                  │")
    print("│                          ▼                                  │")

    for i in range(n_encoder_layers):
        print("│  ┌───────────────────────────────────────────────────────┐ │")
        print(f"│  │ Encoder Layer {i+1}/{n_encoder_layers}                                  │ │")
        print("│  ├───────────────────────────────────────────────────────┤ │")
        print("│  │  ┌─────────────────────────────────────────────────┐ │ │")
        print("│  │  │ LayerNorm                                       │ │ │")
        print("│  │  └─────────────────────────────────────────────────┘ │ │")
        print("│  │  ┌─────────────────────────────────────────────────┐ │ │")
        print("│  │  │ Multi-Head Self-Attention (4 heads)            │ │ │")
        print("│  │  │   Q, K, V: (batch, seq, 128) → 4x(batch,seq,32)│ │ │")
        print("│  │  └─────────────────────────────────────────────────┘ │ │")
        print("│  │  ┌─────────────────────────────────────────────────┐ │ │")
        print("│  │  │ Residual Connection + Dropout                   │ │ │")
        print("│  │  └─────────────────────────────────────────────────┘ │ │")
        print("│  │  ┌─────────────────────────────────────────────────┐ │ │")
        print("│  │  │ LayerNorm                                       │ │ │")
        print("│  │  └─────────────────────────────────────────────────┘ │ │")
        print("│  │  ┌─────────────────────────────────────────────────┐ │ │")
        print("│  │  │ Feed-Forward Network                            │ │ │")
        print("│  │  │   Linear: 128 → 512 (GELU)                     │ │ │")
        print("│  │  │   Linear: 512 → 128                            │ │ │")
        print("│  │  └─────────────────────────────────────────────────┘ │ │")
        print("│  │  ┌─────────────────────────────────────────────────┐ │ │")
        print("│  │  │ Residual Connection + Dropout                   │ │ │")
        print("│  │  └─────────────────────────────────────────────────┘ │ │")
        print("│  └───────────────────────────────────────────────────────┘ │")
        if i < n_encoder_layers - 1:
            print("│                          │                                  │")
            print("│                          ▼                                  │")

    print("│                          │                                  │")
    print("│                          ▼                                  │")
    print("│  ┌───────────────────────────────────────────────────────┐ │")
    print("│  │ LayerNorm (Post-Encoder)                              │ │")
    print("│  └───────────────────────────────────────────────────────┘ │")
    print("│                                                             │")
    print("└─────────────────────────────────────────────────────────────┘")
    print("                          │")
    print("                          │ Encoder Output")
    print("                          │ (batch, seq_len, 128)")
    print("                          │")
    print("                          ▼")
    print()

    # Decoder
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│                    DECODER (Text Path)                      │")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│                                                             │")
    print("│  ┌───────────────────────────────────────────────────────┐ │")
    print("│  │ Token Embedding                                       │ │")
    print(f"│  │   Vocab Size: {vocab_size:,}                          │ │")
    print("│  │   Embedding Dim: 128                                  │ │")
    print("│  └───────────────────────────────────────────────────────┘ │")
    print("│                          │                                  │")
    print("│                          ▼                                  │")
    print("│  ┌───────────────────────────────────────────────────────┐ │")
    print("│  │ Positional Embedding (Learned)                        │ │")
    print("│  │   Shape: (2000, 128)                                  │ │")
    print("│  └───────────────────────────────────────────────────────┘ │")
    print("│                          │                                  │")
    print("│                          ▼                                  │")

    for i in range(n_decoder_layers):
        print("│  ┌───────────────────────────────────────────────────────┐ │")
        print(f"│  │ Decoder Layer {i+1}/{n_decoder_layers}                                  │ │")
        print("│  ├───────────────────────────────────────────────────────┤ │")
        print("│  │  ┌─────────────────────────────────────────────────┐ │ │")
        print("│  │  │ LayerNorm                                       │ │ │")
        print("│  │  └─────────────────────────────────────────────────┘ │ │")
        print("│  │  ┌─────────────────────────────────────────────────┐ │ │")
        print("│  │  │ Masked Self-Attention (4 heads)                │ │ │")
        print("│  │  │   Causal mask for autoregressive generation    │ │ │")
        print("│  │  └─────────────────────────────────────────────────┘ │ │")
        print("│  │  ┌─────────────────────────────────────────────────┐ │ │")
        print("│  │  │ Residual Connection + Dropout                   │ │ │")
        print("│  │  └─────────────────────────────────────────────────┘ │ │")
        print("│  │  ┌─────────────────────────────────────────────────┐ │ │")
        print("│  │  │ LayerNorm                                       │ │ │")
        print("│  │  └─────────────────────────────────────────────────┘ │ │")
        print("│  │  ┌─────────────────────────────────────────────────┐ │ │")
        print("│  │  │ Cross-Attention (4 heads)                      │ │ │")
        print("│  │  │   Q: from decoder, K,V: from encoder           │ │ │")
        print("│  │  └─────────────────────────────────────────────────┘ │ │")
        print("│  │  ┌─────────────────────────────────────────────────┐ │ │")
        print("│  │  │ Residual Connection + Dropout                   │ │ │")
        print("│  │  └─────────────────────────────────────────────────┘ │ │")
        print("│  │  ┌─────────────────────────────────────────────────┐ │ │")
        print("│  │  │ LayerNorm                                       │ │ │")
        print("│  │  └─────────────────────────────────────────────────┘ │ │")
        print("│  │  ┌─────────────────────────────────────────────────┐ │ │")
        print("│  │  │ Feed-Forward Network                            │ │ │")
        print("│  │  │   Linear: 128 → 512 (GELU)                     │ │ │")
        print("│  │  │   Linear: 512 → 128                            │ │ │")
        print("│  │  └─────────────────────────────────────────────────┘ │ │")
        print("│  │  ┌─────────────────────────────────────────────────┐ │ │")
        print("│  │  │ Residual Connection + Dropout                   │ │ │")
        print("│  │  └─────────────────────────────────────────────────┘ │ │")
        print("│  └───────────────────────────────────────────────────────┘ │")
        if i < n_decoder_layers - 1:
            print("│                          │                                  │")
            print("│                          ▼                                  │")

    print("│                          │                                  │")
    print("│                          ▼                                  │")
    print("│  ┌───────────────────────────────────────────────────────┐ │")
    print("│  │ LayerNorm (Post-Decoder)                              │ │")
    print("│  └───────────────────────────────────────────────────────┘ │")
    print("│                          │                                  │")
    print("│                          ▼                                  │")
    print("│  ┌───────────────────────────────────────────────────────┐ │")
    print("│  │ Output Projection                                     │ │")
    print(f"│  │   Linear: 128 → {vocab_size:,}                         │ │")
    print("│  └───────────────────────────────────────────────────────┘ │")
    print("│                                                             │")
    print("└─────────────────────────────────────────────────────────────┘")
    print("                          │")
    print("                          ▼")
    print()

    # Output
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│                         OUTPUT                              │")
    print("├─────────────────────────────────────────────────────────────┤")
    print(f"│  Logits: (batch, seq_len, {vocab_size:,})                     │")
    print("│  Apply Softmax → Token Probabilities                        │")
    print("│  Autoregressive Decoding → Text Transcription              │")
    print("└─────────────────────────────────────────────────────────────┘")
    print()

    print("=" * 80)
    print("KEY FEATURES")
    print("=" * 80)
    print()
    print("✓ Encoder-Decoder Architecture (like Whisper)")
    print("✓ Convolutional audio preprocessing (mel → features)")
    print("✓ Positional embeddings for sequence modeling")
    print("✓ Multi-head attention for parallel processing")
    print("✓ Cross-attention for audio-text alignment")
    print("✓ Residual connections for gradient flow")
    print("✓ Layer normalization for training stability")
    print("✓ Causal masking for autoregressive generation")
    print(f"✓ Compact size: {total_params:,} parameters (~{total_params * 4 / (1024**2):.1f} MB)")
    print()

    print("=" * 80)
    print("TRAINING DETAILS")
    print("=" * 80)
    print()
    print("Loss Function: CrossEntropyLoss with label smoothing (0.1)")
    print("Optimizer: AdamW (weight_decay=0.01)")
    print("Learning Rate: 1e-3 (adaptive decay on plateau)")
    print("Training Modes:")
    print("  • Full: Train on complete audio segments")
    print("  • Curriculum: Train on progressively longer chunks")
    print("  • Augmented: Train with pitch/speed variations")
    print()

if __name__ == "__main__":
    visualize_model()
