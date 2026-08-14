"""Transformer encoders and a decoder-only LM built from PyTorch primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


AttentionVariant = Literal["absolute", "alibi", "local", "disentangled"]


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    max_sequence_length: int = 32
    embedding_dim: int = 64
    num_heads: int = 2
    num_layers: int = 4
    feedforward_dim: int = 100
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.embedding_dim % self.num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if self.vocab_size < 2:
            raise ValueError("vocab_size must include at least padding and unknown tokens")
        if self.max_sequence_length < 1:
            raise ValueError("max_sequence_length must be positive")


def _alibi_slopes(num_heads: int) -> torch.Tensor:
    """Slopes from the ALiBi reference algorithm, including non-powers of two."""

    def slopes_for_power_of_two(heads: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(heads) - 3)))
        return [start * (start**index) for index in range(heads)]

    if math.log2(num_heads).is_integer():
        return torch.tensor(slopes_for_power_of_two(num_heads))
    lower_power = 2 ** math.floor(math.log2(num_heads))
    base = slopes_for_power_of_two(lower_power)
    extra = slopes_for_power_of_two(2 * lower_power)[0::2][: num_heads - lower_power]
    return torch.tensor(base + extra)


class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product self-attention with optional causal masking."""

    def __init__(self, config: TransformerConfig, *, causal: bool = False) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads
        self.causal = causal
        self.qkv = nn.Linear(config.embedding_dim, 3 * config.embedding_dim)
        self.output = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

    def _attention_bias(self, sequence_length: int, device: torch.device) -> torch.Tensor | None:
        del sequence_length, device
        return None

    def _structural_mask(self, sequence_length: int, device: torch.device) -> torch.Tensor:
        mask = torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device)
        return mask.tril() if self.causal else mask

    def forward(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, embedding_dim = hidden.shape
        query, key, value = self.qkv(hidden).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(
                batch_size, sequence_length, self.num_heads, self.head_dim
            ).transpose(1, 2)

        query, key, value = map(split_heads, (query, key, value))
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        bias = self._attention_bias(sequence_length, hidden.device)
        if bias is not None:
            scores = scores + bias.to(dtype=scores.dtype)

        allowed = self._structural_mask(sequence_length, hidden.device)[None, None, :, :]
        if attention_mask is not None:
            allowed = allowed & attention_mask[:, None, None, :].bool()
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        context = self.attention_dropout(weights) @ value
        context = context.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, embedding_dim
        )
        return self.residual_dropout(self.output(context)), weights


class AlibiSelfAttention(MultiHeadSelfAttention):
    """Bidirectional ALiBi adaptation for encoder self-attention."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        self.register_buffer("slopes", _alibi_slopes(config.num_heads), persistent=False)

    def _attention_bias(self, sequence_length: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(sequence_length, device=device)
        distance = (positions[:, None] - positions[None, :]).abs()
        return -self.slopes[:, None, None] * distance[None, :, :]


class LocalSelfAttention(MultiHeadSelfAttention):
    """Dense attention constrained to a symmetric local receptive field."""

    def __init__(self, config: TransformerConfig, window_size: int) -> None:
        super().__init__(config)
        if window_size < 0:
            raise ValueError("window_size must be non-negative")
        self.window_size = window_size

    def _structural_mask(self, sequence_length: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(sequence_length, device=device)
        return (positions[:, None] - positions[None, :]).abs() <= self.window_size


class DisentangledSelfAttention(nn.Module):
    """DeBERTa-inspired content/content and content/position attention."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads
        self.max_sequence_length = config.max_sequence_length
        self.content_query = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.content_key = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.value = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.relative_embedding = nn.Embedding(
            2 * config.max_sequence_length - 1, config.embedding_dim
        )
        self.position_query = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)
        self.position_key = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)
        self.output = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = tensor.shape
        return tensor.view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, embedding_dim = hidden.shape
        query = self._split_heads(self.content_query(hidden))
        key = self._split_heads(self.content_key(hidden))
        value = self._split_heads(self.value(hidden))

        positions = torch.arange(sequence_length, device=hidden.device)
        relative_ids = positions[None, :] - positions[:, None]
        relative_ids = relative_ids + self.max_sequence_length - 1
        relative = self.relative_embedding(relative_ids)
        position_query = self.position_query(relative).view(
            sequence_length, sequence_length, self.num_heads, self.head_dim
        )
        position_key = self.position_key(relative).view(
            sequence_length, sequence_length, self.num_heads, self.head_dim
        )

        content_to_content = query @ key.transpose(-2, -1)
        content_to_position = torch.einsum("bhid,ijhd->bhij", query, position_key)
        position_to_content = torch.einsum("ijhd,bhjd->bhij", position_query, key)
        scores = (content_to_content + content_to_position + position_to_content) / math.sqrt(
            3 * self.head_dim
        )
        if attention_mask is not None:
            scores = scores.masked_fill(
                ~attention_mask[:, None, None, :].bool(), torch.finfo(scores.dtype).min
            )
        weights = F.softmax(scores, dim=-1)
        context = self.attention_dropout(weights) @ value
        context = context.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, embedding_dim
        )
        return self.residual_dropout(self.output(context)), weights


class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.embedding_dim, config.feedforward_dim),
            nn.ReLU(),
            nn.Linear(config.feedforward_dim, config.embedding_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.layers(hidden)


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig, attention: nn.Module) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.embedding_dim)
        self.attention = attention
        self.feedforward_norm = nn.LayerNorm(config.embedding_dim)
        self.feedforward = FeedForward(config)

    def forward(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attended, weights = self.attention(self.attention_norm(hidden), attention_mask)
        hidden = hidden + attended
        hidden = hidden + self.feedforward(self.feedforward_norm(hidden))
        return hidden, weights


class TransformerEncoder(nn.Module):
    """Stacked bidirectional Transformer encoder with masked mean pooling."""

    def __init__(
        self,
        config: TransformerConfig,
        *,
        variant: AttentionVariant = "absolute",
        local_window_size: int = 4,
    ) -> None:
        super().__init__()
        self.config = config
        self.variant = variant
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=0)
        self.position_embedding = (
            nn.Embedding(config.max_sequence_length, config.embedding_dim)
            if variant in {"absolute", "local"}
            else None
        )
        self.embedding_dropout = nn.Dropout(config.dropout)

        def make_attention() -> nn.Module:
            if variant == "absolute":
                return MultiHeadSelfAttention(config)
            if variant == "alibi":
                return AlibiSelfAttention(config)
            if variant == "local":
                return LocalSelfAttention(config, local_window_size)
            if variant == "disentangled":
                return DisentangledSelfAttention(config)
            raise ValueError(f"Unknown attention variant: {variant}")

        self.blocks = nn.ModuleList(
            TransformerBlock(config, make_attention()) for _ in range(config.num_layers)
        )
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        _, sequence_length = input_ids.shape
        if sequence_length > self.config.max_sequence_length:
            raise ValueError("Input is longer than max_sequence_length")
        if attention_mask is None:
            attention_mask = input_ids.ne(0)

        hidden = self.token_embedding(input_ids)
        if self.position_embedding is not None:
            positions = torch.arange(sequence_length, device=input_ids.device)
            hidden = hidden + self.position_embedding(positions)[None, :, :]
        hidden = self.embedding_dropout(hidden)

        attention_maps: list[torch.Tensor] = []
        for block in self.blocks:
            hidden, weights = block(hidden, attention_mask)
            if return_attention:
                attention_maps.append(weights)
        hidden = self.final_norm(hidden)

        float_mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        pooled = (hidden * float_mask).sum(dim=1) / float_mask.sum(dim=1).clamp_min(1.0)
        return (pooled, attention_maps) if return_attention else pooled


class SpeechClassifier(nn.Module):
    def __init__(
        self, encoder: TransformerEncoder, hidden_dim: int = 100, num_classes: int = 3
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(encoder.config.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(input_ids, attention_mask))


class TransformerDecoderLM(nn.Module):
    """Decoder-only, causal language model; no encoder or cross-attention is used."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.max_sequence_length, config.embedding_dim)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config, MultiHeadSelfAttention(config, causal=True))
            for _ in range(config.num_layers)
        )
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.apply(TransformerEncoder._initialize_weights)
        self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[
        torch.Tensor, torch.Tensor | None, list[torch.Tensor]
    ]:
        _, sequence_length = input_ids.shape
        if sequence_length > self.config.max_sequence_length:
            raise ValueError("Input is longer than max_sequence_length")
        positions = torch.arange(sequence_length, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        hidden = self.embedding_dropout(hidden)

        attention_maps: list[torch.Tensor] = []
        for block in self.blocks:
            hidden, weights = block(hidden)
            if return_attention:
                attention_maps.append(weights)
        logits = self.lm_head(self.final_norm(hidden))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        if return_attention:
            return logits, loss, attention_maps
        return logits, loss
