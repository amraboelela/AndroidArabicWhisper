#!/usr/bin/env python3
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ImprovedDecoderTransformer(nn.Module):
    """
    Improved audio-to-text decoder transformer
    - Multiple layers (4 layers instead of 1)
    - Multiple attention heads (4 heads instead of 1)
    - Better regularization with dropout
    - Dimension: 1600
    """

    def __init__(self, vocab_size, d_model=1600, n_layers=4, n_heads=4,
                 d_ff=6400, dropout=0.1, max_seq_len=512):
        super(ImprovedDecoderTransformer, self).__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Audio projection
        self.audio_projection = nn.Linear(d_model, d_model)

        # Token embedding for text
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional embedding
        self.positional_embedding = nn.Embedding(max_seq_len, d_model)

        # Modality embeddings (audio vs text)
        self.modality_embedding = nn.Embedding(2, d_model)

        # Stack of transformer decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.ln_final = nn.LayerNorm(d_model)

        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.positional_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.modality_embedding.weight, mean=0.0, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, audio_features=None, text_ids=None, labels=None):
        """Forward pass with audio and text inputs"""
        batch_size = audio_features.shape[0] if audio_features is not None else text_ids.shape[0]

        embeddings_list = []
        position_offset = 0

        # Process audio
        if audio_features is not None:
            audio_len = audio_features.shape[1]
            audio_embeds = self.audio_projection(audio_features)

            # Add positional + modality embeddings
            audio_positions = torch.arange(0, audio_len, dtype=torch.long, device=audio_features.device)
            audio_positions = audio_positions.unsqueeze(0).expand(batch_size, audio_len)
            audio_pos_embeds = self.positional_embedding(audio_positions)

            audio_modality = torch.zeros((batch_size, audio_len), dtype=torch.long, device=audio_features.device)
            audio_mod_embeds = self.modality_embedding(audio_modality)

            audio_embeds = audio_embeds + audio_pos_embeds + audio_mod_embeds
            audio_embeds = self.dropout(audio_embeds)
            embeddings_list.append(audio_embeds)
            position_offset = audio_len

        # Process text
        if text_ids is not None:
            text_len = text_ids.shape[1]
            text_embeds = self.token_embedding(text_ids)

            # Add positional + modality embeddings
            text_positions = torch.arange(position_offset, position_offset + text_len,
                                         dtype=torch.long, device=text_ids.device)
            text_positions = text_positions.unsqueeze(0).expand(batch_size, text_len)
            text_pos_embeds = self.positional_embedding(text_positions)

            text_modality = torch.ones((batch_size, text_len), dtype=torch.long, device=text_ids.device)
            text_mod_embeds = self.modality_embedding(text_modality)

            text_embeds = text_embeds + text_pos_embeds + text_mod_embeds
            text_embeds = self.dropout(text_embeds)
            embeddings_list.append(text_embeds)

        # Concatenate audio and text
        x = torch.cat(embeddings_list, dim=1)

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)

        # Final layer norm
        x = self.ln_final(x)

        # Get text portion for output
        if audio_features is not None:
            audio_len = audio_features.shape[1]
            x = x[:, audio_len:, :]

        # Output projection
        logits = self.output_projection(x)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                labels.reshape(-1),
                ignore_index=-100
            )

        return logits if loss is None else (logits, loss)

    def generate(self, audio_features, max_new_tokens=50, temperature=1.0):
        """Generate text from audio"""
        self.eval()
        batch_size = audio_features.shape[0]
        text_ids = torch.ones((batch_size, 1), dtype=torch.long, device=audio_features.device)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.forward(audio_features=audio_features, text_ids=text_ids)
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                text_ids = torch.cat([text_ids, next_token], dim=1)

                if (next_token == 2).all():
                    break

        return text_ids


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer with multi-head attention"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        """Forward pass"""
        # Self-attention with residual
        attn_output = self.self_attn(x)
        x = self.ln1(x + attn_output)

        # FFN with residual
        ffn_output = self.ffn(x)
        x = self.ln2(x + ffn_output)

        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking"""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass"""
        batch_size, seq_len, d_model = x.shape

        # Project and reshape for multi-head attention
        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, V)

        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.out_proj(output)

        return output
