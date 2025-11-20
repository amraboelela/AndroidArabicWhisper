#!/usr/bin/env python3
"""
Encoder-Decoder Transformer for Audio-to-Text (Whisper Tiny-compatible encoder)

Encoder matches Whisper Tiny architecture exactly for weight transfer
Decoder is standard transformer decoder (can be smaller/custom)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderDecoderTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=384, n_encoder_layers=6, n_decoder_layers=4,
                 n_heads=6, d_ff=1536, dropout=0.1, max_seq_len=1500, n_mels=40):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_mels = n_mels

        # -------------------------
        # Whisper Tiny-compatible encoder
        # -------------------------
        self.conv1 = nn.Conv1d(n_mels, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)

        # Learned positional embeddings
        self.positional_embedding = nn.Parameter(torch.empty(max_seq_len, d_model))

        # Encoder blocks (Whisper-style)
        self.blocks = nn.ModuleList([
            WhisperEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_encoder_layers)
        ])
        self.ln_post = nn.LayerNorm(d_model)

        # -------------------------
        # Decoder (customizable)
        # -------------------------
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_pos_embedding = nn.Embedding(max_seq_len, d_model)

        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_decoder_layers)
        ])
        self.decoder_ln = nn.LayerNorm(d_model)

        # Output projection (weight-tied to token_embedding)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        self.output_projection.weight = self.token_embedding.weight

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    # -------------------------
    # Weight initialization
    # -------------------------
    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.decoder_pos_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.positional_embedding, mean=0.0, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # -------------------------
    # Encoder
    # -------------------------
    def encode(self, mel_features):
        """
        Whisper Tiny-compatible encoder
        mel_features: (batch, n_mels, time)
        Returns: encoder_output (batch, time//2, d_model)
        """
        x = F.gelu(self.conv1(mel_features))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)  # (batch, time//2, d_model)
        seq_len = x.shape[1]
        # Scale embeddings by sqrt(d_model) for gradient stability
        x = x * math.sqrt(self.d_model) + self.positional_embedding[:seq_len]

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x

    # -------------------------
    # Decoder
    # -------------------------
    def decode(self, text_ids, encoder_output, tgt_key_padding_mask: torch.Tensor = None,
               memory_key_padding_mask: torch.Tensor = None, past_kvs=None, use_cache=False):
        """
        Args:
            past_kvs: list of (past_self_kv, past_cross_kv) for each layer
            use_cache: whether to return KV cache
        Returns:
            logits: (B, T, vocab_size)
            present_kvs: list of (self_kv, cross_kv) for each layer if use_cache else None
        """
        batch_size, tgt_len = text_ids.shape
        # Scale token embeddings by sqrt(d_model) for gradient stability
        x = self.token_embedding(text_ids) * math.sqrt(self.d_model)

        # Calculate correct position offset when using cache
        if past_kvs is not None and len(past_kvs) > 0:
            # Position is past_length + current position
            past_length = past_kvs[0][0][0].shape[2]  # Get K tensor shape: (B, n_heads, past_len, d_k)
            offset = past_length
        else:
            offset = 0

        # Clamp positions to avoid index out of bounds
        positions = torch.arange(offset, offset + tgt_len, device=text_ids.device).clamp(max=self.decoder_pos_embedding.num_embeddings - 1)
        positions = positions.unsqueeze(0).expand(batch_size, tgt_len)
        x = x + self.decoder_pos_embedding(positions)
        x = self.dropout(x)

        present_kvs = [] if use_cache else None

        for i, layer in enumerate(self.decoder_layers):
            past_self_kv = past_kvs[i][0] if past_kvs is not None else None
            past_cross_kv = past_kvs[i][1] if past_kvs is not None else None

            x, (present_self_kv, present_cross_kv) = layer(
                x, encoder_output,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                past_self_kv=past_self_kv,
                past_cross_kv=past_cross_kv,
                use_cache=use_cache
            )

            if use_cache:
                present_kvs.append((present_self_kv, present_cross_kv))

        x = self.decoder_ln(x)
        logits = self.output_projection(x)
        return logits, present_kvs

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, mel_features=None, text_ids=None, encoder_output=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        if encoder_output is None:
            assert mel_features is not None, "Provide mel_features or encoder_output"
            encoder_output = self.encode(mel_features)
        assert text_ids is not None, "text_ids required for decoding"
        logits, _ = self.decode(text_ids, encoder_output, tgt_key_padding_mask=tgt_key_padding_mask,
                                memory_key_padding_mask=memory_key_padding_mask, use_cache=False)
        return logits

    # -------------------------
    # Generation (greedy or sampling)
    # -------------------------
    def generate(self, mel_features, max_new_tokens=50, temperature=1.0, min_tokens=1,
                 bos_id: int = 1, eos_id: int = 2, use_sampling=True, audio_duration_seconds=None):
        """
        Generate with KV-cache for faster autoregressive decoding
        """
        assert mel_features is not None
        self.eval()
        batch_size = mel_features.shape[0]
        device = mel_features.device

        with torch.no_grad():
            # Encode once (this is cached across all decoding steps)
            encoder_output = self.encode(mel_features)

            # Initialize with BOS token
            text_ids = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)

            # KV cache: list of (self_kv, cross_kv) for each layer
            past_kvs = None

            for step in range(max_new_tokens):
                # Decode only the last token, using cached KV from previous steps
                logits, past_kvs = self.decode(
                    text_ids[:, -1:],  # Only pass the last token
                    encoder_output,
                    past_kvs=past_kvs,
                    use_cache=True
                )

                next_logits = logits[:, -1, :] / max(1e-8, temperature)
                if step < min_tokens:
                    next_logits[:, eos_id] = float("-inf")

                if use_sampling:
                    probs = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

                text_ids = torch.cat([text_ids, next_token], dim=1)

                if (next_token == eos_id).all():
                    break

        return text_ids


# -------------------------
# Whisper Encoder Block
# -------------------------
class WhisperEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, causal=False)
        self.attn_ln = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.mlp_ln = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(self.attn_ln(x), self.attn_ln(x))
        x = x + attn_out
        x = x + self.mlp(self.mlp_ln(x))
        return x


# -------------------------
# Transformer Decoder Layer
# -------------------------
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

    def forward(self, x, encoder_output, tgt_key_padding_mask=None, memory_key_padding_mask=None,
                past_self_kv=None, past_cross_kv=None, use_cache=False):
        """
        Args:
            past_self_kv: cached (K, V) for self-attention
            past_cross_kv: cached (K, V) for cross-attention
            use_cache: whether to return new KV cache
        Returns:
            x: output
            (present_self_kv, present_cross_kv): new cache if use_cache else (None, None)
        """
        # Self-attention with cache
        self_attn_out, present_self_kv = self.self_attn(
            self.ln1(x), self.ln1(x),
            key_padding_mask=tgt_key_padding_mask,
            past_kv=past_self_kv,
            use_cache=use_cache
        )
        x = x + self_attn_out

        # Cross-attention with cache
        cross_attn_out, present_cross_kv = self.cross_attn(
            self.ln2(x), encoder_output,
            key_padding_mask=memory_key_padding_mask,
            past_kv=past_cross_kv,
            use_cache=use_cache
        )
        x = x + cross_attn_out
        x = x + self.ffn(self.ln3(x))

        return x, (present_self_kv, present_cross_kv)


# -------------------------
# MultiHeadAttention
# -------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, causal=False):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.causal = causal

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, key_padding_mask=None, past_kv=None, use_cache=False):
        """
        Args:
            query: (B, Q, D)
            key_value: (B, KV, D)
            key_padding_mask: (B, KV) optional
            past_kv: tuple of (past_key, past_value) each (B, n_heads, past_len, d_k)
            use_cache: whether to return current K, V for caching
        Returns:
            out: (B, Q, D)
            present_kv: tuple of (K, V) if use_cache else None
        """
        batch_size, q_len, _ = query.shape
        kv_len = key_value.shape[1]

        Q = self.q_proj(query).view(batch_size, q_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(key_value).view(batch_size, kv_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(key_value).view(batch_size, kv_len, self.n_heads, self.d_k).transpose(1, 2)

        # Concatenate past K, V if provided
        if past_kv is not None:
            past_k, past_v = past_kv
            K = torch.cat([past_k, K], dim=2)  # (B, n_heads, past_len + kv_len, d_k)
            V = torch.cat([past_v, V], dim=2)
            kv_len = K.shape[2]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if self.causal:
            causal_mask = torch.triu(torch.ones((q_len, kv_len), device=query.device), diagonal=kv_len - q_len + 1).bool()
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        if key_padding_mask is not None:
            # key_padding_mask: (B, K) where True = mask out (pad token)
            # Expand to (B, 1, 1, K) for broadcasting over heads and queries
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        out = self.out_proj(out)

        present_kv = (K, V) if use_cache else None
        return out, present_kv
