#!/usr/bin/env python3
"""
Encoder-Decoder Transformer for Audio-to-Text (Whisper-style)

Changes / improvements:
 - forward() returns logits only (training loop should compute loss)
 - optional key-padding masks supported for encoder/decoder (batching)
 - weight tying between token embedding and output projection
 - generate() accepts bos_id/eos_id and uses temperature properly
 - small doc/comments and safety checks
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderDecoderTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_encoder_layers=4, n_decoder_layers=4,
                 n_heads=8, d_ff=2048, dropout=0.1, max_seq_len=512):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Audio Encoder projection (assumes input last-dim == 128)
        self.audio_projection = nn.Linear(128, d_model)

        # Positional embeddings
        self.encoder_pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.decoder_pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Encoder / decoder stacks
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_encoder_layers)
        ])
        self.encoder_ln = nn.LayerNorm(d_model)

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_decoder_layers)
        ])
        self.decoder_ln = nn.LayerNorm(d_model)

        # Output projection (tied to token_embedding.weight later)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()
        # Weight tying (output logits weight tied to token embeddings)
        self.output_projection.weight = self.token_embedding.weight

    def _init_weights(self):
        # Initialize embeddings and linear weights like many transformer implementations
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.encoder_pos_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.decoder_pos_embedding.weight, mean=0.0, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # --------------------
    # Encoding / Decoding
    # --------------------
    def encode(self, audio_features, src_key_padding_mask: torch.Tensor = None):
        """
        audio_features: (batch, audio_len, 800)
        src_key_padding_mask: (batch, audio_len) boolean mask, True for valid tokens (or False for pad).
                              Convention used below is bool mask where True means valid (not masked).
        Returns: encoder_output (batch, audio_len, d_model)
        """
        batch_size, audio_len, _ = audio_features.shape

        # Project audio to d_model and scale (scale optional)
        x = self.audio_projection(audio_features) / math.sqrt(self.d_model)

        # Positional embeddings
        positions = torch.arange(audio_len, device=audio_features.device).unsqueeze(0).expand(batch_size, audio_len)
        pos_embeds = self.encoder_pos_embedding(positions)
        x = x + pos_embeds
        x = self.dropout(x)

        # Pass through encoder layers (pass mask if provided)
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)

        x = self.encoder_ln(x)
        return x

    def decode(self, text_ids, encoder_output, tgt_key_padding_mask: torch.Tensor = None, memory_key_padding_mask: torch.Tensor = None):
        """
        text_ids: (batch, tgt_len)
        encoder_output: (batch, src_len, d_model)
        tgt_key_padding_mask: (batch, tgt_len) bool mask for decoder inputs (True for valid tokens)
        memory_key_padding_mask: (batch, src_len) bool mask for encoder outputs (True for valid tokens)
        Returns: logits (batch, tgt_len, vocab_size)
        """
        batch_size, tgt_len = text_ids.shape

        # Token embeddings
        x = self.token_embedding(text_ids)  # (batch, tgt_len, d_model)

        # Positional embeddings
        positions = torch.arange(tgt_len, device=text_ids.device).unsqueeze(0).expand(batch_size, tgt_len)
        pos_embeds = self.decoder_pos_embedding(positions)
        x = x + pos_embeds
        x = self.dropout(x)

        # Decoder layers with cross-attention (pass masks)
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)

        x = self.decoder_ln(x)
        logits = self.output_projection(x)  # (batch, tgt_len, vocab_size)
        return logits

    def forward(self, audio_features=None, text_ids=None, encoder_output=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Forward returns logits only (trainer computes loss).
        Provide either audio_features (to encode) or encoder_output (precomputed).
        Masks are boolean tensors with True indicating valid (non-padding) tokens.
        """
        if encoder_output is None:
            assert audio_features is not None, "Provide audio_features or encoder_output"
            encoder_output = self.encode(audio_features, src_key_padding_mask=src_key_padding_mask)

        assert text_ids is not None, "text_ids required for decoding"
        logits = self.decode(text_ids, encoder_output, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        return logits

    # --------------------
    # Generation (greedy)
    # --------------------
    def generate(self, audio_features, max_new_tokens=50, temperature=1.0, min_tokens=1, bos_id: int = 1, eos_id: int = 2, use_sampling=True):
        """
        Generation using cached encoder output with optional sampling.

        audio_features: (batch, audio_len, 128)
        use_sampling: if True, sample from distribution; if False, use greedy argmax
        Returns: generated_ids (batch, seq_len)
        """
        assert audio_features is not None
        self.eval()
        batch_size = audio_features.shape[0]
        device = audio_features.device

        with torch.no_grad():
            encoder_output = self.encode(audio_features)  # (batch, src_len, d_model)

            # Start tokens
            text_ids = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)

            for step in range(max_new_tokens):
                logits = self.decode(text_ids, encoder_output)  # (batch, cur_len, vocab_size)
                next_logits = logits[:, -1, :] / max(1e-8, temperature)

                # prevent early EOS for first min_tokens
                if step < min_tokens:
                    # set eos logit to -inf
                    next_logits[:, eos_id] = float("-inf")

                # Sample or greedy
                if use_sampling:
                    probs = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)
                else:
                    next_token = torch.argmax(next_logits, dim=-1, keepdim=True)  # (batch, 1)

                text_ids = torch.cat([text_ids, next_token], dim=1)

                # stop if all sequences produced eos
                if (next_token == eos_id).all():
                    break

        return text_ids  # includes bos_id as first token

# -------------------------
# Encoder / Decoder Layers
# -------------------------
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, causal=False)
        self.ln1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, src_key_padding_mask: torch.Tensor = None):
        # Self-attention (support optional padding mask)
        attn_output = self.self_attn(x, x, key_padding_mask=src_key_padding_mask)
        x = self.ln1(x + attn_output)

        ffn_output = self.ffn(x)
        x = self.ln2(x + ffn_output)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, causal=True)
        self.ln1 = nn.LayerNorm(d_model)

        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout, causal=False)
        self.ln2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, tgt_key_padding_mask: torch.Tensor = None, memory_key_padding_mask: torch.Tensor = None):
        # causal self-attention
        attn_output = self.self_attn(x, x, key_padding_mask=tgt_key_padding_mask)
        x = self.ln1(x + attn_output)

        # cross-attention: queries from decoder, keys/values from encoder
        cross_output = self.cross_attn(x, encoder_output, key_padding_mask=memory_key_padding_mask)
        x = self.ln2(x + cross_output)

        ffn_output = self.ffn(x)
        x = self.ln3(x + ffn_output)
        return x


# -------------------------
# MultiHeadAttention
# -------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, causal=False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.causal = causal

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, key_padding_mask: torch.Tensor = None):
        """
        query: (batch, q_len, d_model)
        key_value: (batch, kv_len, d_model)
        key_padding_mask: optional (batch, kv_len) bool tensor where True indicates valid token.
        Returns: (batch, q_len, d_model)
        """
        batch_size, q_len, _ = query.shape
        kv_len = key_value.shape[1]

        # project
        Q = self.q_proj(query).view(batch_size, q_len, self.n_heads, self.d_k).transpose(1, 2)  # (batch, heads, q_len, d_k)
        K = self.k_proj(key_value).view(batch_size, kv_len, self.n_heads, self.d_k).transpose(1, 2)  # (batch, heads, kv_len, d_k)
        V = self.v_proj(key_value).view(batch_size, kv_len, self.n_heads, self.d_k).transpose(1, 2)  # (batch, heads, kv_len, d_k)

        # scaled dot-product
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, heads, q_len, kv_len)

        # causal mask (for causal self-attention)
        if self.causal:
            # only valid when kv_len == q_len (self-attention). Create upper triangular mask.
            causal_mask = torch.triu(torch.ones((q_len, kv_len), device=query.device), diagonal=1).bool()  # True for masked positions
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # key padding mask: key_padding_mask shape (batch, kv_len) with True for VALID tokens.
        # We want to mask out invalid positions => convert to mask where True = valid => invert for masked_fill
        if key_padding_mask is not None:
            # allow passing mask as boolean where True means valid; convert to mask of invalid positions
            if key_padding_mask.dtype != torch.bool:
                key_padding_mask = key_padding_mask.to(torch.bool)
            # expand to (batch, 1, 1, kv_len)
            mask = ~key_padding_mask.unsqueeze(1).unsqueeze(1)  # True where we want to mask
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)  # (batch, heads, q_len, d_k)
        out = out.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        out = self.out_proj(out)
        return out
