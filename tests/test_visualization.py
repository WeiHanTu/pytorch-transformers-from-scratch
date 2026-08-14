import tempfile
import unittest
from pathlib import Path

import torch

from transformers_from_scratch.visualization import save_attention_overview


class VisualizationTest(unittest.TestCase):
    def test_attention_overview_is_written(self) -> None:
        attention_maps = [
            torch.softmax(torch.randn(1, 2, 3, 3), dim=-1),
            torch.softmax(torch.randn(1, 2, 3, 3), dim=-1),
        ]
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "attention.png"
            result = save_attention_overview(attention_maps, ["one", "two", "three"], destination)
            self.assertEqual(result, destination)
            self.assertTrue(destination.is_file())
            self.assertGreater(destination.stat().st_size, 1_000)


if __name__ == "__main__":
    unittest.main()
