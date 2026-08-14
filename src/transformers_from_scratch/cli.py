"""Command-line entry point for repeatable experiments."""

from __future__ import annotations

import argparse
from typing import Any

from .experiments import (
    environment_metadata,
    run_classification,
    run_language_model,
    write_results,
)


VARIANTS = ("absolute", "alibi", "local", "disentangled")


def _shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", default="speechesdataset")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--output", help="Optional JSON output path")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transformers-scratch",
        description=(
            "Train compact Transformer models implemented from first principles in PyTorch."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    classification = subparsers.add_parser("classify", help="Train speech classifiers")
    _shared_arguments(classification)
    classification.add_argument("--variant", choices=(*VARIANTS, "all"), default="absolute")
    classification.add_argument("--epochs", type=int, default=15)
    classification.add_argument("--local-window-size", type=int, default=4)
    classification.add_argument(
        "--attention-sentence",
        help="Optional sentence whose trained encoder attention should be rendered",
    )
    classification.add_argument(
        "--attention-dir",
        help="Directory for the attention overview; requires --attention-sentence",
    )

    language_model = subparsers.add_parser("language-model", help="Train the causal decoder LM")
    _shared_arguments(language_model)
    language_model.add_argument("--max-steps", type=int, default=500)
    language_model.add_argument("--eval-interval", type=int, default=100)
    language_model.add_argument("--eval-batches", type=int, default=200)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "classify":
        if bool(args.attention_sentence) != bool(args.attention_dir):
            parser.error("--attention-sentence and --attention-dir must be provided together")
        variants = VARIANTS if args.variant == "all" else (args.variant,)
        result = run_classification(
            data_dir=args.data_dir,
            variants=variants,
            device_name=args.device,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            block_size=args.block_size,
            local_window_size=args.local_window_size,
            attention_sentence=args.attention_sentence,
            attention_output_dir=args.attention_dir,
        )
    else:
        result = run_language_model(
            data_dir=args.data_dir,
            device_name=args.device,
            seed=args.seed,
            max_steps=args.max_steps,
            eval_interval=args.eval_interval,
            eval_batches=args.eval_batches,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            block_size=args.block_size,
        )
    result["environment"] = environment_metadata()
    if args.output:
        write_results(result, args.output)
    return result


if __name__ == "__main__":
    main()
