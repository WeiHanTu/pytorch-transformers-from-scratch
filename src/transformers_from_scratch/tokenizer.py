"""A small, deterministic word-level tokenizer with no downloaded resources."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


TOKEN_PATTERN = re.compile(r"\w+(?:['’]\w+)?|[^\w\s]", flags=re.UNICODE)


class WordTokenizer:
    """Tokenize words and punctuation and map them to a stable vocabulary.

    Vocabulary entries are sorted by descending frequency and then
    lexicographically. This makes token IDs identical across processes.
    """

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(self, text: str, *, lowercase: bool = True, min_frequency: int = 1) -> None:
        if min_frequency < 1:
            raise ValueError("min_frequency must be at least 1")
        self.lowercase = lowercase
        counts = Counter(self.tokenize(text))
        vocabulary = sorted(
            (token for token, count in counts.items() if count >= min_frequency),
            key=lambda token: (-counts[token], token),
        )
        self.itos = [self.PAD_TOKEN, self.UNK_TOKEN, *vocabulary]
        self.stoi = {token: index for index, token in enumerate(self.itos)}

    @property
    def pad_id(self) -> int:
        return self.stoi[self.PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.stoi[self.UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def tokenize(self, text: str) -> list[str]:
        if self.lowercase:
            text = text.lower()
        return TOKEN_PATTERN.findall(text)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(token, self.unk_id) for token in self.tokenize(text)]

    def decode(self, indices: list[int]) -> str:
        return " ".join(
            self.itos[index] if 0 <= index < len(self.itos) else self.UNK_TOKEN
            for index in indices
        )

    def save(self, path: str | Path) -> None:
        payload = {"lowercase": self.lowercase, "vocabulary": self.itos}
        Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "WordTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls("", lowercase=payload["lowercase"])
        tokenizer.itos = payload["vocabulary"]
        tokenizer.stoi = {token: index for index, token in enumerate(tokenizer.itos)}
        return tokenizer
