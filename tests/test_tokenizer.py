import tempfile
import unittest
from pathlib import Path

from transformers_from_scratch.tokenizer import WordTokenizer


class WordTokenizerTest(unittest.TestCase):
    def test_vocabulary_is_frequency_then_lexicographic(self) -> None:
        tokenizer = WordTokenizer("pear apple pear banana apple pear")
        self.assertEqual(tokenizer.itos, ["<pad>", "<unk>", "pear", "apple", "banana"])

    def test_tokenizer_handles_unknowns_and_round_trip(self) -> None:
        tokenizer = WordTokenizer("Hello, world!")
        self.assertEqual(
            tokenizer.encode("HELLO unseen"), [tokenizer.stoi["hello"], tokenizer.unk_id]
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "tokenizer.json"
            tokenizer.save(destination)
            restored = WordTokenizer.load(destination)
        self.assertEqual(restored.itos, tokenizer.itos)
        self.assertEqual(restored.encode("Hello, world!"), tokenizer.encode("Hello, world!"))


if __name__ == "__main__":
    unittest.main()
