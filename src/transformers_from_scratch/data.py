"""Dataset parsing and batching for speech classification and language modeling."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .tokenizer import WordTokenizer


CLASS_NAMES = {
    0: "Barack Obama",
    1: "George W. Bush",
    2: "George H. W. Bush",
}

EXPECTED_FILES = (
    "train_CLS.tsv",
    "test_CLS.tsv",
    "train_LM.txt",
    "test_LM_obama.txt",
    "test_LM_wbush.txt",
    "test_LM_hbush.txt",
)


def validate_data_directory(data_dir: str | Path, required: Sequence[str] = EXPECTED_FILES) -> Path:
    path = Path(data_dir)
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing dataset files in {path}: {joined}. See DATASET.md for setup and licensing."
        )
    return path


def read_classification_rows(path: str | Path) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            if not line:
                continue
            try:
                raw_label, text = line.split("\t", maxsplit=1)
                label = int(raw_label)
            except ValueError as error:
                raise ValueError(f"Malformed row {line_number} in {path}") from error
            if label not in CLASS_NAMES:
                raise ValueError(f"Unknown label {label} on row {line_number} in {path}")
            if text.strip():
                rows.append((label, text))
    return rows


def build_training_corpus(data_dir: str | Path) -> str:
    """Build vocabulary from training splits only, avoiding test-set leakage."""
    path = validate_data_directory(data_dir, ("train_CLS.tsv", "train_LM.txt"))
    classification_text = "\n".join(
        text for _, text in read_classification_rows(path / "train_CLS.tsv")
    )
    language_model_text = (path / "train_LM.txt").read_text(encoding="utf-8")
    return f"{classification_text}\n{language_model_text}"


class SpeechClassificationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, tokenizer: WordTokenizer, path: str | Path) -> None:
        self.tokenizer = tokenizer
        self.rows = read_classification_rows(path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        label, text = self.rows[index]
        input_ids = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        return input_ids, torch.tensor(label, dtype=torch.long)


class LanguageModelingDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, tokenizer: WordTokenizer, text: str, block_size: int) -> None:
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        self.block_size = block_size
        if len(self.data) <= block_size:
            raise ValueError("Language-modeling text must contain more tokens than block_size")

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[index : index + self.block_size + 1]
        return chunk[:-1], chunk[1:]


def classification_collator(
    *, block_size: int, pad_id: int
) -> Callable[
    [list[tuple[torch.Tensor, torch.Tensor]]],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    def collate(
        batch: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequences, labels = zip(*batch)
        truncated = [sequence[:block_size] for sequence in sequences]
        input_ids = pad_sequence(truncated, batch_first=True, padding_value=pad_id)
        if input_ids.shape[1] < block_size:
            input_ids = torch.nn.functional.pad(
                input_ids, (0, block_size - input_ids.shape[1]), value=pad_id
            )
        attention_mask = input_ids.ne(pad_id)
        return input_ids, attention_mask, torch.stack(labels)

    return collate
