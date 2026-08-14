"""Reproducible training and evaluation loops used by the command-line interface."""

from __future__ import annotations

import json
import math
import platform
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import (
    LanguageModelingDataset,
    SpeechClassificationDataset,
    build_training_corpus,
    classification_collator,
    validate_data_directory,
)
from .models import SpeechClassifier, TransformerConfig, TransformerDecoderLM, TransformerEncoder
from .tokenizer import WordTokenizer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return device
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


@torch.no_grad()
def classification_accuracy(
    model: SpeechClassifier,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> float:
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    for input_ids, attention_mask, labels in loader:
        logits = model(input_ids.to(device), attention_mask.to(device))
        labels = labels.to(device)
        correct += logits.argmax(dim=-1).eq(labels).sum().item()
        total += labels.numel()
    model.train(was_training)
    return correct / total


def run_classification(
    *,
    data_dir: str | Path,
    variants: Iterable[str],
    device_name: str = "auto",
    seed: int = 42,
    epochs: int = 15,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    block_size: int = 32,
    local_window_size: int = 4,
    attention_sentence: str | None = None,
    attention_output_dir: str | Path | None = None,
) -> dict[str, Any]:
    path = validate_data_directory(data_dir, ("train_CLS.tsv", "test_CLS.tsv", "train_LM.txt"))
    device = select_device(device_name)
    tokenizer = WordTokenizer(build_training_corpus(path))
    train_dataset = SpeechClassificationDataset(tokenizer, path / "train_CLS.tsv")
    test_dataset = SpeechClassificationDataset(tokenizer, path / "test_CLS.tsv")
    collate = classification_collator(block_size=block_size, pad_id=tokenizer.pad_id)
    config = TransformerConfig(vocab_size=tokenizer.vocab_size, max_sequence_length=block_size)
    results: dict[str, Any] = {}

    for variant in variants:
        seed_everything(seed)
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate,
            generator=generator,
        )
        evaluation_train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate
        )
        encoder = TransformerEncoder(
            config, variant=variant, local_window_size=local_window_size
        )
        model = SpeechClassifier(encoder).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        history = []

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            examples = 0
            for input_ids, attention_mask, labels in train_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(input_ids, attention_mask)
                loss = nn.functional.cross_entropy(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item() * labels.numel()
                examples += labels.numel()
            train_accuracy = classification_accuracy(model, evaluation_train_loader, device)
            epoch_result = {
                "epoch": epoch,
                "loss": total_loss / examples,
                "train_accuracy": train_accuracy,
            }
            history.append(epoch_result)
            print(
                f"[{variant}] epoch {epoch:02d}/{epochs}: "
                f"loss={epoch_result['loss']:.4f}, train_accuracy={train_accuracy:.4f}",
                flush=True,
            )

        test_accuracy = classification_accuracy(model, test_loader, device)
        results[variant] = {
            "train_accuracy": history[-1]["train_accuracy"],
            "test_accuracy": test_accuracy,
            "parameters": count_parameters(model),
            "history": history,
        }
        if attention_sentence is not None and attention_output_dir is not None:
            from .visualization import save_attention_overview

            tokens = tokenizer.tokenize(attention_sentence)[:block_size]
            input_ids = torch.tensor(
                [tokenizer.encode(attention_sentence)[:block_size]],
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            model.eval()
            with torch.no_grad():
                _, attention_maps = model.encoder(
                    input_ids, attention_mask, return_attention=True
                )
            figure_path = Path(attention_output_dir) / f"{variant}-attention.png"
            save_attention_overview(attention_maps, tokens, figure_path)
            results[variant]["attention_figure"] = str(figure_path)
            print(f"[{variant}] attention_figure={figure_path}", flush=True)
        print(f"[{variant}] test_accuracy={test_accuracy:.4f}", flush=True)

    return {
        "task": "speech_classification",
        "device": str(device),
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "block_size": block_size,
        "vocab_size": tokenizer.vocab_size,
        "variants": results,
    }


@torch.no_grad()
def language_model_perplexity(
    model: TransformerDecoderLM,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> float:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    token_count = 0
    for batch_index, (input_ids, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        targets = targets.to(device)
        _, loss = model(input_ids.to(device), targets)
        assert loss is not None
        total_loss += loss.item() * targets.numel()
        token_count += targets.numel()
    model.train(was_training)
    if token_count == 0:
        raise ValueError("Cannot compute perplexity from an empty data loader")
    return math.exp(total_loss / token_count)


def run_language_model(
    *,
    data_dir: str | Path,
    device_name: str = "auto",
    seed: int = 42,
    max_steps: int = 500,
    eval_interval: int = 100,
    eval_batches: int = 200,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    block_size: int = 32,
) -> dict[str, Any]:
    path = validate_data_directory(data_dir)
    device = select_device(device_name)
    seed_everything(seed)
    tokenizer = WordTokenizer(build_training_corpus(path))
    train_text = (path / "train_LM.txt").read_text(encoding="utf-8")
    train_dataset = LanguageModelingDataset(tokenizer, train_text, block_size)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    config = TransformerConfig(vocab_size=tokenizer.vocab_size, max_sequence_length=block_size)
    model = TransformerDecoderLM(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    iterator = iter(train_loader)
    history = []

    for step in range(1, max_steps + 1):
        try:
            input_ids, targets = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            input_ids, targets = next(iterator)
        input_ids, targets = input_ids.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(input_ids, targets)
        assert loss is not None
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % eval_interval == 0 or step == max_steps:
            perplexity = language_model_perplexity(
                model, train_loader, device, max_batches=eval_batches
            )
            history.append({"step": step, "train_perplexity": perplexity})
            print(
                f"[decoder] step {step:04d}/{max_steps}: "
                f"train_perplexity={perplexity:.4f}",
                flush=True,
            )

    test_files = {
        "obama": "test_LM_obama.txt",
        "g_w_bush": "test_LM_wbush.txt",
        "ghw_bush": "test_LM_hbush.txt",
    }
    test_perplexity = {}
    for name, filename in test_files.items():
        dataset = LanguageModelingDataset(
            tokenizer, (path / filename).read_text(encoding="utf-8"), block_size
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        test_perplexity[name] = language_model_perplexity(model, loader, device)
        print(f"[decoder] {name}_perplexity={test_perplexity[name]:.4f}", flush=True)

    return {
        "task": "causal_language_modeling",
        "device": str(device),
        "seed": seed,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "block_size": block_size,
        "vocab_size": tokenizer.vocab_size,
        "parameters": count_parameters(model),
        "train_perplexity": history[-1]["train_perplexity"],
        "test_perplexity": test_perplexity,
        "history": history,
    }


def environment_metadata() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "platform": platform.platform(),
    }


def write_results(payload: dict[str, Any], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
