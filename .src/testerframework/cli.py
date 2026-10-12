"""Command-line entry point for dataset inference and setup validation."""

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
            "Run image-level MVTec AD classification with a Transformers "
            "vision-language model, or validate setup only."
        )
    )
    parser.add_argument(
        "--dataset-folder",
        required=True,
        type=pathlib.Path,
        help="MVTec AD dataset root containing <category>/test/<label> image folders.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Hugging Face model ID or local model directory for transformer; "
            "local GGUF path for llama_cpp_python."
        ),
    )
    parser.add_argument(
        "--engine",
        required=True,
        choices=ENGINE_PACKAGE_NAMES,
        help="Inference engine package to run.",
    )
    parser.add_argument(
        "--category",
        help="One MVTec AD category directory name, e.g. bottle or capsule.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path(".dpl/results"),
        help="Directory for the category result JSON (default: .dpl/results).",
    )
    parser.add_argument(
        "--log-dir",
        type=pathlib.Path,
        default=pathlib.Path(".dpl/logs"),
        help="Directory for structured tester event logs (default: .dpl/logs).",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device map (default: auto).",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="Model weight dtype; verify hardware support before overriding auto.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8,
        help="Maximum generated label tokens per image (default: 8).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow Transformers to execute custom code supplied by a model.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the dataset directory/model config without loading weights or inference.",
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

    if args.engine != "transformer":
        if args.engine == "llama_cpp_python":
            model_path = pathlib.Path(args.model)
            if not model_path.is_file() or model_path.suffix.lower() != ".gguf":
                parser.error(
                    "llama_cpp_python requires --model to name an existing local "
                    "GGUF file."
                )
            parser.error(
                "the llama_cpp_python package does not yet contain a runnable "
                "MVTec tester."
            )
        parser.error(
            "the llama_serve package does not yet contain a runnable tester "
            "or a defined model/endpoint argument contract."
        )

    if args.validate_only:
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
        print("Validation only: dataset contents are not inspected.")
        return 0

    if not args.category:
        parser.error("--category is required unless --validate-only is set.")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1.")

    from .transformer.transformer_testers import transformer_mvtec_tester

    tester = transformer_mvtec_tester(
        model_path=args.model,
        dataset_folder=args.dataset_folder,
        category=args.category,
        output_dir=args.output_dir,
        device_map=args.device_map,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        trust_remote_code=args.trust_remote_code,
        log_dir=args.log_dir,
    )
    try:
        result = tester.start_test()
    except (HfHubHTTPError, FileNotFoundError, OSError, ValueError, ImportError, RuntimeError) as error:
        parser.error(f"Transformers category run failed: {error}")

    print(
        f"Completed {result['sample_count']} images for category "
        f"{result['category']}: accuracy={result['accuracy']:.4f}, "
        f"mean response={result['mean_response_seconds']:.4f}s, "
        f"p95 response={result['p95_response_seconds']:.4f}s."
    )
    print(f"Results written to {result['result_file']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
