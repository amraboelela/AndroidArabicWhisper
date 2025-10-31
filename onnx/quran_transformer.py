#!/usr/bin/env python3
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DecoderOnlyTransformer(nn.Module):
    """
    Audio-to-text decoder-only transformer for Quran transcription
    - Dimension: 800
    - 1 attention layer
    - 1 head
    - 1 FFN output projection layer

    Input format: Audio: a1, a2, ..., an Text: t1, t2, ..., tm
    - Audio embeddings (a1, a2, ...): mel spectrum features, 800 dim, 10 per second
    - Text tokens (t1, t2, ...): vocabulary indices to predict
    - Audio length: 1-5 seconds (10-50 audio frames)
    """

    def __init__(self, vocab_size, d_model=800, max_seq_len=512, max_audio_len=100):
        super(DecoderOnlyTransformer, self).__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_audio_len = max_audio_len

        # Audio embedding projection (in case audio features need transformation)
        # If audio is already 800-dim mel spectrum, this is identity-like
        self.audio_projection = nn.Linear(d_model, d_model)

        # Token embedding for text
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional embedding (for both audio and text positions)
        self.positional_embedding = nn.Embedding(max_seq_len + max_audio_len, d_model)

        # Modality embeddings to distinguish audio vs text
        self.modality_embedding = nn.Embedding(2, d_model)  # 0=audio, 1=text

        # Single-head self-attention
        self.attention = SingleHeadAttention(d_model)

        # Layer normalization after attention
        self.ln1 = nn.LayerNorm(d_model)

        # Feed-forward network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),  # Expansion
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)   # Projection back
        )

        # Layer normalization after FFN
        self.ln2 = nn.LayerNorm(d_model)

        # Output projection to vocabulary
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.positional_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.modality_embedding.weight, mean=0.0, std=0.02)

        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, audio_features=None, text_ids=None, labels=None):
        """
        Forward pass with audio and text inputs

        Args:
            audio_features: (batch_size, audio_len, d_model) - mel spectrum features
                           audio_len is typically 10-50 (1-5 seconds at 10 fps)
            text_ids: (batch_size, text_len) - text token indices
            labels: (batch_size, text_len) - target tokens for loss computation

        Returns:
            logits: (batch_size, text_len, vocab_size) - output logits for text
            loss: optional, if labels provided
        """
        batch_size = audio_features.shape[0] if audio_features is not None else text_ids.shape[0]

        embeddings_list = []
        position_offset = 0

        # Process audio embeddings if provided
        if audio_features is not None:
            audio_len = audio_features.shape[1]

            # Project audio features
            audio_embeds = self.audio_projection(audio_features)  # (batch_size, audio_len, d_model)

            # Add positional embeddings for audio
            audio_positions = torch.arange(0, audio_len, dtype=torch.long, device=audio_features.device)
            audio_positions = audio_positions.unsqueeze(0).expand(batch_size, audio_len)
            audio_pos_embeds = self.positional_embedding(audio_positions)

            # Add modality embedding (0 = audio)
            audio_modality = torch.zeros((batch_size, audio_len), dtype=torch.long, device=audio_features.device)
            audio_mod_embeds = self.modality_embedding(audio_modality)

            # Combine audio embeddings
            audio_embeds = audio_embeds + audio_pos_embeds + audio_mod_embeds
            embeddings_list.append(audio_embeds)
            position_offset = audio_len

        # Process text tokens if provided
        if text_ids is not None:
            text_len = text_ids.shape[1]

            # Token embeddings
            text_embeds = self.token_embedding(text_ids)  # (batch_size, text_len, d_model)

            # Add positional embeddings for text (continuing from audio positions)
            text_positions = torch.arange(position_offset, position_offset + text_len,
                                         dtype=torch.long, device=text_ids.device)
            text_positions = text_positions.unsqueeze(0).expand(batch_size, text_len)
            text_pos_embeds = self.positional_embedding(text_positions)

            # Add modality embedding (1 = text)
            text_modality = torch.ones((batch_size, text_len), dtype=torch.long, device=text_ids.device)
            text_mod_embeds = self.modality_embedding(text_modality)

            # Combine text embeddings
            text_embeds = text_embeds + text_pos_embeds + text_mod_embeds
            embeddings_list.append(text_embeds)

        # Concatenate audio and text embeddings
        x = torch.cat(embeddings_list, dim=1)  # (batch_size, audio_len + text_len, d_model)

        # Self-attention with residual connection and layer norm
        attn_output = self.attention(x)
        x = self.ln1(x + attn_output)

        # Feed-forward network with residual connection and layer norm
        ffn_output = self.ffn(x)
        x = self.ln2(x + ffn_output)

        # Output projection to vocabulary (only for text portion)
        if audio_features is not None:
            audio_len = audio_features.shape[1]
            x = x[:, audio_len:, :]  # Take only text portion

        logits = self.output_projection(x)  # (batch_size, text_len, vocab_size)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Flatten for cross-entropy
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                labels.reshape(-1),
                ignore_index=-100
            )

        return logits if loss is None else (logits, loss)

    def generate(self, audio_features, max_new_tokens=50, temperature=1.0):
        """
        Generate text tokens from audio autoregressively

        Args:
            audio_features: (batch_size, audio_len, d_model) - mel spectrum features
            max_new_tokens: number of text tokens to generate
            temperature: sampling temperature

        Returns:
            generated: (batch_size, max_new_tokens) - generated text tokens
        """
        self.eval()
        batch_size = audio_features.shape[0]

        # Start with <s> token
        text_ids = torch.ones((batch_size, 1), dtype=torch.long, device=audio_features.device)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get logits for current sequence
                logits = self.forward(audio_features=audio_features, text_ids=text_ids)

                # Get logits for last token
                logits = logits[:, -1, :] / temperature  # (batch_size, vocab_size)

                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

                # Append to sequence
                text_ids = torch.cat([text_ids, next_token], dim=1)

                # Stop if we generate </s> for all sequences
                if (next_token == 2).all():  # 2 is </s> token
                    break

        return text_ids


class SingleHeadAttention(nn.Module):
    """Single-head self-attention with causal masking"""

    def __init__(self, d_model):
        super(SingleHeadAttention, self).__init__()

        self.d_model = d_model

        # Query, Key, Value projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # Dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        Forward pass with causal masking

        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        # Compute Q, K, V
        Q = self.q_proj(x)  # (batch_size, seq_len, d_model)
        K = self.k_proj(x)  # (batch_size, seq_len, d_model)
        V = self.v_proj(x)  # (batch_size, seq_len, d_model)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_model)
        # (batch_size, seq_len, seq_len)

        # Apply causal mask (prevent attending to future tokens)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask.unsqueeze(0), float('-inf'))

        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, V)  # (batch_size, seq_len, d_model)

        # Output projection
        output = self.out_proj(output)

        return output


def create_model_from_vocabulary(vocab_file="vocabulary.json"):
    """Create model using vocabulary from JSON file"""

    # Load vocabulary
    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    vocab_size = len(vocab)

    print(f"Creating decoder-only transformer:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Model dimension: 800")
    print(f"  Attention heads: 1")
    print(f"  Layers: 1 attention + 1 FFN")

    # Create model
    model = DecoderOnlyTransformer(vocab_size=vocab_size, d_model=800)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / (1024**2):.2f} MB (float32)")

    return model, vocab


def main():
    """Main function to create and test the audio-to-text model"""

    # Create model
    model, vocab = create_model_from_vocabulary()

    # Test forward pass with audio + text
    print("\n" + "="*60)
    print("Testing forward pass with audio + text...")
    print("="*60)

    batch_size = 2
    audio_len = 30  # 3 seconds at 10 fps
    text_len = 10

    # Create dummy audio features (mel spectrum, already 800-dim)
    audio_features = torch.randn(batch_size, audio_len, 800)

    # Create dummy text tokens
    text_ids = torch.randint(3, len(vocab), (batch_size, text_len))  # Start from 3 (after special tokens)

    # Create labels (same as text_ids for teacher forcing)
    labels = text_ids.clone()

    print(f"\nInput shapes:")
    print(f"  Audio features: {audio_features.shape} (batch, audio_len, d_model)")
    print(f"  Text IDs: {text_ids.shape} (batch, text_len)")
    print(f"  Labels: {labels.shape} (batch, text_len)")

    # Forward pass
    logits, loss = model(audio_features=audio_features, text_ids=text_ids, labels=labels)

    print(f"\nOutput shapes:")
    print(f"  Logits: {logits.shape} (batch, text_len, vocab_size)")
    print(f"  Loss: {loss.item():.4f}")

    # Test generation from audio
    print("\n" + "="*60)
    print("Testing generation from audio...")
    print("="*60)

    # Single audio sample (2 seconds)
    audio_features_single = torch.randn(1, 20, 800)  # 2 seconds at 10 fps

    print(f"\nInput audio: {audio_features_single.shape} (1, 20, 800)")
    print(f"  Duration: 2 seconds (20 frames at 10 fps)")

    # Generate text
    generated = model.generate(audio_features_single, max_new_tokens=10)

    print(f"\nGenerated tokens: {generated[0].tolist()}")
    print(f"Generated words: {[vocab[idx] for idx in generated[0].tolist()]}")

    # Show prompt format
    print("\n" + "="*60)
    print("Prompt Format Example:")
    print("="*60)
    print("\nTraining format:")
    print("  Audio: a1, a2, a3, ..., an  (mel spectrum, 800-dim each, 10 per second)")
    print("  Text: t1, t2, t3, ..., tm   (vocabulary tokens)")
    print("\nFor 3-second audio clip:")
    print("  Audio frames: 30 (a1...a30)")
    print("  Text tokens: variable length (e.g., 5-15 words)")
    print("\nExample batch:")
    print("  Batch size: 2")
    print("  Audio: (2, 30, 800)")
    print("  Text: (2, 10) -> vocabulary indices")

    # Save model architecture
    print("\n" + "="*60)
    print("Model Architecture:")
    print("="*60)
    print(model)

    # Optionally save model
    # torch.save(model.state_dict(), "quran_audio_transformer.pt")
    # print("\nModel saved to quran_audio_transformer.pt")


if __name__ == "__main__":
    main()
