# Muhaffez Whisper Model Architecture

## Model Statistics

- **Total Parameters**: 4,074,112
- **Model Size**: ~15.5 MB (float32)
- **Vocabulary Size**: 14,755 Arabic words
- **Model Dimension**: 128
- **Encoder Max Sequence Length**: 2000 (for long audio)
- **Decoder Max Sequence Length**: 100 (for text generation)

## Configuration

| Parameter | Value |
|-----------|-------|
| Mel Bins | 40 |
| Encoder Layers | 4 |
| Decoder Layers | 4 |
| Attention Heads | 4 |
| Feed-Forward Dimension | 512 |
| Model Dimension (d_model) | 128 |
| Dropout | 0.1 |

## Architecture Overview

```
┌──────────────────────────────────────────┐
│              INPUT                       │
├──────────────────────────────────────────┤
│  Audio: Mel Spectrogram (40 bins)       │
└──────────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────┐
│         ENCODER (Audio Path)             │
├──────────────────────────────────────────┤
│  • Conv1D (40→128, kernel=3, stride=1)  │
│  • GELU Activation                       │
│  • Conv1D (128→128, kernel=3, stride=2) │
│  • Positional Embedding                  │
│  • 4x Transformer Encoder Layers         │
│    - Multi-Head Self-Attention (4 heads) │
│    - Feed-Forward Network (128→512→128)  │
│    - Residual + LayerNorm                │
└──────────────────────────────────────────┘
                  │
                  ▼ (encoder hidden states)
┌──────────────────────────────────────────┐
│         DECODER (Text Path)              │
├──────────────────────────────────────────┤
│  Start with: <s> (start token)           │
│  • Token Embedding (14,755 vocab)        │
│  • Positional Embedding (learned)        │
│  • 4x Transformer Decoder Layers         │
│    - Masked Self-Attention (4 heads)     │
│    - Cross-Attention with Encoder        │
│    - Feed-Forward Network (128→512→128)  │
│    - Residual + LayerNorm                │
│  • Output Projection (128→14,755)        │
│  Generate tokens autoregressively ──┐    │
│  (feed each output as next input)   │    │
└─────────────────────────────────────┼────┘
                  │                   │
                  │◄──────────────────┘
                  ▼
┌──────────────────────────────────────────┐
│              OUTPUT                      │
├──────────────────────────────────────────┤
│  Token sequence: <s> word1 word2 ... </s>│
│  → Arabic Text Transcription             │
└──────────────────────────────────────────┘
```

## Encoder Details

### Audio Preprocessing
1. **Conv1D Layer 1**: 40 mel bins → 128 channels
   - Kernel size: 3, Stride: 1, Padding: 1
   - Activation: GELU
2. **Conv1D Layer 2**: Downsampling by factor of 2
   - Kernel size: 3, Stride: 2, Padding: 1
   - Activation: GELU
3. **Positional Embedding**: Learned (trainable tensor, 2000 positions for audio sequences)

### Encoder Layers (4 layers)
Each encoder layer contains:
- **LayerNorm** → Pre-normalization
- **Multi-Head Self-Attention** (4 heads)
  - Head dimension: 32 (128 / 4)
  - Q, K, V projections: Linear(128, 128)
  - Output projection: Linear(128, 128)
- **Residual Connection + Dropout**
- **LayerNorm**
- **Feed-Forward Network**
  - FC1: Linear(128, 512) + GELU
  - FC2: Linear(512, 128)
- **Residual Connection + Dropout**

## Decoder Details

### Text Embedding
- **Token Embedding**: 14,755 vocabulary → 128 dimensions
- **Positional Embedding**: Learned embeddings (100 positions for text generation)

### Decoder Layers (4 layers)
Each decoder layer contains:
- **LayerNorm**
- **Masked Self-Attention** (4 heads, causal mask)
  - Prevents attending to future tokens
  - Enables autoregressive generation
- **Residual Connection + Dropout**
- **LayerNorm**
- **Cross-Attention** (4 heads)
  - Q: from decoder (text)
  - K, V: from encoder (audio)
  - Aligns text generation with audio features
- **Residual Connection + Dropout**
- **LayerNorm**
- **Feed-Forward Network**
  - FC1: Linear(128, 512) + GELU
  - FC2: Linear(512, 128)
- **Residual Connection + Dropout**

### Output Layer
- **Linear Projection**: 128 → 14,755
- **Softmax**: Convert logits to probabilities
- **Greedy/Beam Decoding**: Generate tokens autoregressively

## Key Features

✓ **Encoder-Decoder Architecture** - Inspired by Whisper
✓ **Convolutional Preprocessing** - Efficient mel spectrogram processing
✓ **Learned Positional Embeddings** - Trainable position encodings for both encoder and decoder
✓ **Multi-Head Attention** - 4 heads for parallel processing
✓ **Cross-Attention** - Aligns audio and text representations
✓ **Residual Connections** - Improves gradient flow
✓ **Layer Normalization** - Stabilizes training
✓ **Causal Masking** - Enables autoregressive generation
✓ **Compact Size** - 4.1M parameters (~15.5 MB)
✓ **Optimized Sequence Lengths** - Encoder: 2000 (audio), Decoder: 100 (text)

## Training Configuration

| Setting | Value |
|---------|-------|
| Loss Function | CrossEntropyLoss |
| Label Smoothing | 0.1 |
| Optimizer | AdamW |
| Weight Decay | 0.01 |
| Initial Learning Rate | 1e-3 |
| LR Schedule | Decay on plateau (0.5x) |
| Minimum LR | 1e-7 |
| Gradient Clipping | 1.0 |
| Dropout | 0.1 |

## Training Modes

### 1. Full Training
- Trains on complete audio segments
- Direct audio → full transcription
- Best for learning complete utterances

### 2. Curriculum Training
- Progressively trains on longer chunks
- Stage 1: 1.3s → 1 word
- Stage 2: 2.6s → 2 words
- Stage N: Full segment → Full text
- Helps with alignment and gradual learning

### 3. Augmented Training
- Includes pitch-shifted variations (±2, ±4 semitones)
- Includes speed variations (±10%, ±20%)
- Improves robustness to voice variations
- 8x more training samples

## Model Comparison

| Model | Parameters | Size | Use Case |
|-------|------------|------|----------|
| **Muhaffez Whisper** | 4.1M | ~15.5 MB | Quran transcription |
| Whisper Tiny | 39M | ~150 MB | General purpose |
| Whisper Base | 74M | ~290 MB | General purpose |
| Whisper Small | 244M | ~950 MB | General purpose |
| Whisper Medium | 769M | ~3 GB | General purpose |
| Whisper Large | 1550M | ~6 GB | General purpose |

**Muhaffez Whisper is ~19x smaller than Whisper Base** while being specifically optimized for Quran Arabic.
