import unittest

import torch

from transformers_from_scratch.models import (
    LocalSelfAttention,
    MultiHeadSelfAttention,
    SpeechClassifier,
    TransformerConfig,
    TransformerDecoderLM,
    TransformerEncoder,
)


def small_config() -> TransformerConfig:
    return TransformerConfig(
        vocab_size=31,
        max_sequence_length=8,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
        feedforward_dim=24,
        dropout=0.0,
    )


class ModelTest(unittest.TestCase):
    def test_encoder_variants_shape_and_backward(self) -> None:
        for variant in ("absolute", "alibi", "local", "disentangled"):
            with self.subTest(variant=variant):
                encoder = TransformerEncoder(small_config(), variant=variant, local_window_size=1)
                model = SpeechClassifier(encoder, hidden_dim=12)
                input_ids = torch.tensor([[2, 3, 4, 0], [5, 6, 7, 8]])
                mask = input_ids.ne(0)
                logits = model(input_ids, mask)
                self.assertEqual(logits.shape, (2, 3))
                logits.sum().backward()
                self.assertIsNotNone(encoder.token_embedding.weight.grad)

    def test_padding_does_not_change_encoder_pooling(self) -> None:
        encoder = TransformerEncoder(small_config(), variant="absolute").eval()
        short = torch.tensor([[2, 3, 4]])
        padded = torch.tensor([[2, 3, 4, 0, 0]])
        with torch.no_grad():
            short_output = encoder(short, short.ne(0))
            padded_output = encoder(padded, padded.ne(0))
        self.assertTrue(torch.allclose(short_output, padded_output, atol=1e-5))

    def test_attention_probabilities_sum_to_one(self) -> None:
        attention = MultiHeadSelfAttention(small_config()).eval()
        hidden = torch.randn(2, 5, 16)
        _, weights = attention(hidden)
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(2, 4, 5), atol=1e-6))

    def test_causal_attention_cannot_look_ahead(self) -> None:
        attention = MultiHeadSelfAttention(small_config(), causal=True).eval()
        _, weights = attention(torch.randn(1, 5, 16))
        self.assertEqual(torch.count_nonzero(torch.triu(weights, diagonal=1)).item(), 0)

    def test_local_attention_respects_window(self) -> None:
        attention = LocalSelfAttention(small_config(), window_size=1).eval()
        _, weights = attention(torch.randn(1, 5, 16))
        positions = torch.arange(5)
        outside_window = (positions[:, None] - positions[None, :]).abs() > 1
        self.assertEqual(
            torch.count_nonzero(weights.masked_select(outside_window[None, None])).item(), 0
        )

    def test_decoder_loss_and_causal_maps(self) -> None:
        model = TransformerDecoderLM(small_config()).eval()
        input_ids = torch.randint(2, 31, (2, 6))
        logits, loss, maps = model(input_ids, input_ids, return_attention=True)
        self.assertEqual(logits.shape, (2, 6, 31))
        self.assertIsNotNone(loss)
        self.assertEqual(loss.ndim, 0)
        self.assertEqual(len(maps), 2)
        self.assertEqual(torch.count_nonzero(torch.triu(maps[0], diagonal=1)).item(), 0)


if __name__ == "__main__":
    unittest.main()
