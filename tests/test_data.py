import tempfile
import unittest
from pathlib import Path

import torch

from transformers_from_scratch.data import (
    SpeechClassificationDataset,
    classification_collator,
    read_classification_rows,
)
from transformers_from_scratch.tokenizer import WordTokenizer


class DataTest(unittest.TestCase):
    def test_tsv_parser_preserves_tabs_inside_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.tsv"
            path.write_text("0\tfirst\tsegment\n1\tsecond segment\n", encoding="utf-8")
            self.assertEqual(
                read_classification_rows(path),
                [(0, "first\tsegment"), (1, "second segment")],
            )

    def test_classification_collator_returns_padding_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.tsv"
            path.write_text("0\tone two\n1\tone\n", encoding="utf-8")
            tokenizer = WordTokenizer("one two")
            dataset = SpeechClassificationDataset(tokenizer, path)
            collate = classification_collator(block_size=4, pad_id=tokenizer.pad_id)
            input_ids, mask, labels = collate([dataset[0], dataset[1]])
        self.assertEqual(input_ids.shape, (2, 4))
        self.assertEqual(mask.shape, (2, 4))
        self.assertEqual(mask.sum(dim=1).tolist(), [2, 1])
        self.assertTrue(torch.equal(labels, torch.tensor([0, 1])))


if __name__ == "__main__":
    unittest.main()
