# Transformer Decoder Model for Quranic Arabic

This directory contains a decoder-only transformer model optimized for Quranic Arabic text generation.

## Model Overview

### Architecture
- **Type**: Decoder-only Transformer
- **Dimension (d_model)**: 800
- **Attention Layers**: 1 layer with 1 head
- **FFN Layers**: 1 layer (expands to 3200, then projects back to 800)
- **Max Sequence Length**: 512 tokens
- **Total Parameters**: 190,213,564 (~190M parameters)

### Vocabulary
- **Size**: 50,364 tokens
- **Type**: Quranic Arabic subwords (from Tarteel AI Whisper model)
- **Tokenizer**: WhisperTokenizerFast (trained on Quran)
- **Source**: `tarteel-ai/whisper-tiny-ar-quran`

## Files

### Model Files
- `transformer_decoder.onnx` (726 MB) - ONNX format model for inference
- `transformer_decoder.pt` (726 MB) - PyTorch state dict
- `model_info.json` - Model architecture information

### Code Files
- `create_transformer_model.py` - Script to create and export the model
- `use_tarteel_tokenizer.py` - Script to extract and save the Tarteel tokenizer
- `example_usage.py` - Example usage scripts

### Tokenizer Files (in `tokenizer/` directory)
- `tokenizer.json` - Tokenizer vocabulary and merges
- `tokenizer_config.json` - Tokenizer configuration
- `vocab_mapping.json` - Vocabulary mapping and special tokens

## Model Details

### Layer Breakdown
1. **Token Embedding Layer**: Maps token IDs to 800-dimensional vectors (50,364 × 800 = 40,291,200 parameters)
2. **Positional Encoding**: Learnable position embeddings (512 × 800 = 409,600 parameters)
3. **Self-Attention Layer**:
   - Query, Key, Value projections: 3 × (800 × 800) = 1,920,000 parameters
   - Single attention head (no multi-head split)
   - Causal masking for autoregressive generation
4. **Feed-Forward Network**:
   - Hidden layer: 800 → 3200 (2,560,000 parameters)
   - Output layer: 3200 → 800 (2,560,000 parameters)
   - GELU activation
5. **Layer Normalization**: 2 layers (after attention and FFN)
6. **Output Projection**: 800 → 50,364 vocabulary (40,291,200 parameters)

### Special Tokens
- `<|endoftext|>` (ID: 50257) - Used for PAD, BOS, and EOS
- Unknown token (ID: 50256)

## Usage

### Loading the ONNX Model

```python
import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession("onnx_claude/transformer_decoder.onnx")

# Prepare input
input_ids = np.array([[3555, 38251, 21984]], dtype=np.int64)  # "بسم الله"

# Run inference
outputs = session.run(None, {"input_ids": input_ids})
logits = outputs[0]  # Shape: (batch_size, seq_length, vocab_size)

# Get next token prediction
next_token_logits = logits[0, -1, :]
next_token_id = np.argmax(next_token_logits)
```

### Loading the PyTorch Model

```python
import torch
from create_transformer_model import SimpleTransformerDecoder

# Create model instance
model = SimpleTransformerDecoder(vocab_size=50364, d_model=800, max_seq_length=512)

# Load weights
model.load_state_dict(torch.load("onnx_claude/transformer_decoder.pt"))
model.eval()

# Prepare input
input_ids = torch.tensor([[3555, 38251, 21984]])  # "بسم الله"

# Run inference
with torch.no_grad():
    logits = model(input_ids)  # Shape: (batch_size, seq_length, vocab_size)

# Generate text
generated = model.generate(input_ids, max_new_tokens=50)
```

### Using the Tokenizer

```python
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("onnx_claude/tokenizer")

# Encode text
text = "بسم الله الرحمن الرحيم"
token_ids = tokenizer.encode(text)
print(f"Tokens: {token_ids}")

# Decode tokens
decoded_text = tokenizer.decode(token_ids)
print(f"Decoded: {decoded_text}")
```

## Model Characteristics

### Strengths
- **Quranic Arabic specialization**: Vocabulary specifically trained on Quran text
- **Compact architecture**: Single attention layer and FFN layer for efficiency
- **ONNX compatibility**: Can be used with ONNX Runtime for deployment
- **Autoregressive generation**: Causal masking enables text generation

### Limitations
- **Untrained weights**: Model architecture is created but weights are randomly initialized
- **Single layer**: Limited capacity compared to deeper models
- **Single attention head**: May not capture diverse attention patterns
- **Requires training**: Model needs to be trained on Quranic text for useful generation

## Training Recommendations

To train this model on Quranic Arabic text:

1. **Dataset**: Use complete Quran text with tafsir or translations
2. **Objective**: Causal language modeling (predict next token)
3. **Optimizer**: AdamW with learning rate ~1e-4
4. **Batch size**: 8-32 sequences
5. **Sequence length**: 128-512 tokens
6. **Training steps**: ~100K steps minimum
7. **Regularization**: Weight decay (0.01), gradient clipping (1.0)

Example training loop:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

model = SimpleTransformerDecoder(vocab_size=50364, d_model=800)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

for batch in train_dataloader:
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    # Forward pass
    logits = model(input_ids)

    # Compute loss
    loss = criterion(logits.view(-1, vocab_size), labels.view(-1))

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

## Deployment

### Android Deployment (ONNX Runtime)
1. Add ONNX Runtime dependency to `build.gradle`
2. Copy `transformer_decoder.onnx` to assets folder
3. Load model in Java/Kotlin using OrtEnvironment

### C++ Inference
Use ONNX Runtime C++ API:
```cpp
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "transformer");
Ort::SessionOptions session_options;
Ort::Session session(env, "transformer_decoder.onnx", session_options);
```

## References

- Tokenizer source: [Tarteel AI Whisper Quran](https://huggingface.co/tarteel-ai/whisper-tiny-ar-quran)
- ONNX format: [ONNX Documentation](https://onnx.ai/)
- Transformer architecture: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## License

Created by Amr Aboelela for Quranic Arabic text generation research.

---

**Note**: This model has randomly initialized weights and requires training on Quranic Arabic text before it can generate meaningful output.
