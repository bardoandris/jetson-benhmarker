"""Command-line entry point for validating benchmark setup."""

import argparse
import pathlib
import sys
import typing

from huggingface_hub.errors import HfHubHTTPError
from transformers import AutoConfig

from testerframework import ENGINE_PACKAGE_NAMES


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a dataset directory and an inference backend/model "
            "reference. This command does not run inference."
        )
    )
    parser.add_argument(
        "--dataset-folder",
        required=True,
        type=pathlib.Path,
        help="Path to the dataset root directory.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Model reference: Hugging Face model ID or local model directory "
            "for transformer; local GGUF file for llama_cpp_python."
        ),
    )
    parser.add_argument(
        "--engine",
        required=True,
        choices=ENGINE_PACKAGE_NAMES,
        help="Inference engine package to validate.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow Transformers to load custom configuration code from a model.",
    )
    return parser


def _validate_transformer_model(model: str, trust_remote_code: bool) -> str:
    config = AutoConfig.from_pretrained(
        model,
        trust_remote_code=trust_remote_code,
    )
    model_type = getattr(config, "model_type", None)
    if isinstance(model_type, str):
        return model_type
    return type(config).__name__


def main(argv: typing.Optional[typing.Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.dataset_folder.is_dir():
        parser.error(f"dataset folder is not a directory: {args.dataset_folder}")

    if args.engine == "transformer":
        try:
            model_type = _validate_transformer_model(
                args.model,
                args.trust_remote_code,
            )
        except (HfHubHTTPError, OSError, ValueError, ImportError) as error:
            parser.error(f"could not load Transformers model config: {error}")

        print(
            f"Setup validated: dataset directory={args.dataset_folder}; "
            f"engine=transformer; model={args.model}; model_type={model_type}."
        )
        print(
            "Validation only: dataset contents are not inspected and no "
            "inference or benchmark metrics are produced."
        )
        return 0

    if args.engine == "llama_cpp_python":
        model_path = pathlib.Path(args.model)
        if not model_path.is_file() or model_path.suffix.lower() != ".gguf":
            parser.error(
                "llama_cpp_python requires --model to name an existing local "
                "GGUF file."
            )
        parser.error(
            "the llama_cpp_python package does not yet contain a runnable "
            "tester; only its local GGUF reference was validated."
        )

    if args.engine == "llama_serve":
        parser.error(
            "the llama_serve package does not yet contain a runnable tester "
            "or a defined model/endpoint argument contract."
        )

    parser.error(f"no setup validator is registered for engine: {args.engine}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
