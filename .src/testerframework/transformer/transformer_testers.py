"""Transformers-backed benchmark tester implementations."""

import json
import pathlib
import re
import statistics
import time
import typing
import uuid
from datetime import datetime, timezone

import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

from testerframework.base_test.base_tester import base_tester

class transformer_tester(base_tester):
    """Run a basic text-generation benchmark with a Hugging Face causal LM."""

    def __init__(
        self,
        model_path: str,
        prompt: str,
        max_new_tokens: int = 32,
        device_map: typing.Union[
            str, typing.Dict[str, typing.Union[str, int]]
        ] = "auto",
        trust_remote_code: bool = True,
        log_dir: typing.Union[str, pathlib.Path] = ".dpl/logs",
    ) -> None:
        super().__init__(name="transformers", log_dir=log_dir)
        if not model_path:
            raise ValueError("model_path must not be empty.")
        if not prompt:
            raise ValueError("prompt must not be empty.")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1.")

        self.model_path = model_path
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code

    def _run_test(self) -> typing.Dict[str, typing.Any]:
        self._logger.add_metadata(
            backend="transformers",
            model_path=self.model_path,
            max_new_tokens=self.max_new_tokens,
            device_map=self.device_map,
        )
        self._logger.info("model_loading_started")
        load_started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=self.trust_remote_code,
                device_map=self.device_map,
            )
        
        model.eval()
        load_seconds = time.perf_counter() - load_started
        self._logger.add_metadata(model_load_seconds=load_seconds)
        self._logger.info("model_loading_completed")

        inputs = tokenizer(self.prompt, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]
        input_token_count = int(input_ids.shape[-1])

        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        generation_started = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate( # type: ignore
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        generation_seconds = time.perf_counter() - generation_started

        generated_token_count = int(output_ids.shape[-1]) - input_token_count
        generated_text = tokenizer.decode(
            output_ids[0, input_token_count:],
            skip_special_tokens=True,
        )
        tokens_per_second = (
            generated_token_count / generation_seconds
            if generated_token_count > 0 and generation_seconds > 0
            else 0.0
        )
        self._logger.add_metadata(
            input_tokens=input_token_count,
            generated_tokens=generated_token_count,
            generation_seconds=generation_seconds,
            tokens_per_second=tokens_per_second,
        )
        self._logger.info("generation_completed")

        return {
            "model_path": self.model_path,
            "input_tokens": input_token_count,
            "generated_tokens": generated_token_count,
            "model_load_seconds": load_seconds,
            "generation_seconds": generation_seconds,
            "tokens_per_second": tokens_per_second,
            "generated_text": generated_text,
        }


class transformer_mvtec_tester(base_tester):
    """Classify images in one MVTec AD test category with a vision-language model."""

    IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

    def __init__(
        self,
        model_path: str,
        dataset_folder: typing.Union[str, pathlib.Path],
        category: str,
        output_dir: typing.Union[str, pathlib.Path] = ".dpl/results",
        device_map: typing.Union[
            str, typing.Dict[str, typing.Union[str, int]]
        ] = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 8,
        trust_remote_code: bool = False,
        log_dir: typing.Union[str, pathlib.Path] = ".dpl/logs",
    ) -> None:
        super().__init__(name="transformer-mvtec", log_dir=log_dir)
        if not model_path:
            raise ValueError("model_path must not be empty.")
        if (
            not category
            or category in {".", ".."}
            or pathlib.Path(category).name != category
        ):
            raise ValueError("category must be a single directory name.")
        if dtype not in {"auto", "float16", "bfloat16", "float32"}:
            raise ValueError("dtype must be auto, float16, bfloat16, or float32.")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1.")

        self.model_path = model_path
        self.dataset_folder = pathlib.Path(dataset_folder)
        self.category = category
        self.output_dir = pathlib.Path(output_dir)
        self.device_map = device_map
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.trust_remote_code = trust_remote_code

    def _get_test_images(self) -> typing.List[typing.Tuple[pathlib.Path, bool]]:
        category_root = self.dataset_folder / self.category
        test_root = category_root / "test"
        if not self.dataset_folder.is_dir():
            raise FileNotFoundError(
                f"Dataset directory does not exist: {self.dataset_folder}"
            )
        if not test_root.is_dir():
            raise FileNotFoundError(
                f"Expected MVTec test directory does not exist: {test_root}"
            )

        samples: typing.List[typing.Tuple[pathlib.Path, bool]] = []
        for label_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
            is_anomalous = label_dir.name != "good"
            for image_path in sorted(label_dir.iterdir()):
                if (
                    image_path.is_file()
                    and image_path.suffix.lower() in self.IMAGE_EXTENSIONS
                ):
                    samples.append((image_path, is_anomalous))
        if not samples:
            raise ValueError(
                f"No supported test images found under {test_root}. "
                "Expected test/good and test/<defect_type> folders."
            )
        if not any(not is_anomalous for _, is_anomalous in samples):
            raise ValueError(
                f"No normal ('good') test images found under {test_root}."
            )
        if not any(is_anomalous for _, is_anomalous in samples):
            raise ValueError(
                f"No anomalous test images found under {test_root}; "
                "expected at least one defect-type directory."
            )
        return samples

    @staticmethod
    def _parse_prediction(response: str, image_path: pathlib.Path) -> bool:
        label = response.strip().casefold()
        if label == "normal":
            return False
        if label == "anomalous":
            return True
        raise ValueError(
            f"Model returned an invalid label for {image_path}: {response!r}. "
            "Expected exactly 'normal' or 'anomalous'."
        )

    @staticmethod
    def _percentile_95(values: typing.Sequence[float]) -> float:
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        rank = 0.95 * (len(ordered) - 1)
        lower = int(rank)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = rank - lower
        return ordered[lower] + fraction * (ordered[upper] - ordered[lower])

    def _run_test(self) -> typing.Dict[str, typing.Any]:
        samples = self._get_test_images()
        self._logger.add_metadata(
            backend="transformers",
            model_path=self.model_path,
            dataset_folder=str(self.dataset_folder),
            category=self.category,
            sample_count=len(samples),
            device_map=self.device_map,
            dtype=self.dtype,
        )
        self._logger.info("model_loading_started")
        load_started = time.perf_counter()
        processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )
        load_options: typing.Dict[str, typing.Any] = {
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.dtype != "auto":
            load_options["dtype"] = getattr(torch, self.dtype)
        model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            **load_options,
        )
        model.eval()
        model_load_seconds = time.perf_counter() - load_started
        self._logger.add_metadata(model_load_seconds=model_load_seconds)
        self._logger.info("model_loading_completed")

        predictions: typing.List[typing.Dict[str, typing.Any]] = []
        preprocessing_times: typing.List[float] = []
        inference_times: typing.List[float] = []
        prompt = (
            f"Inspect this {self.category} product image for a manufacturing "
            "defect. Reply with exactly one word: normal or anomalous."
        )

        for index, (image_path, expected_anomaly) in enumerate(samples, start=1):
            preprocessing_started = time.perf_counter()
            with Image.open(image_path) as source_image:
                with source_image.convert("RGB") as image:
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]
                    model_inputs = processor.apply_chat_template(
                        conversation,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    ).to(model.device)
            preprocessing_seconds = time.perf_counter() - preprocessing_started

            if model.device.type == "cuda":
                torch.cuda.synchronize(model.device)
            inference_started = time.perf_counter()
            with torch.inference_mode():
                output_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            if model.device.type == "cuda":
                torch.cuda.synchronize(model.device)
            inference_seconds = time.perf_counter() - inference_started

            input_token_count = model_inputs["input_ids"].shape[-1]
            response = processor.batch_decode(
                output_ids[:, input_token_count:],
                skip_special_tokens=True,
            )[0]
            predicted_anomaly = self._parse_prediction(response, image_path)
            preprocessing_times.append(preprocessing_seconds)
            inference_times.append(inference_seconds)
            predictions.append(
                {
                    "image": str(image_path.relative_to(self.dataset_folder)),
                    "expected": "anomalous" if expected_anomaly else "normal",
                    "predicted": "anomalous" if predicted_anomaly else "normal",
                    "correct": predicted_anomaly == expected_anomaly,
                    "response": response,
                    "preprocessing_seconds": preprocessing_seconds,
                    "inference_seconds": inference_seconds,
                }
            )
            self._logger.info(
                "image_evaluated",
                extra={
                    "metadata": {
                        "index": index,
                        "image": str(image_path.relative_to(self.dataset_folder)),
                        "correct": predicted_anomaly == expected_anomaly,
                    }
                },
            )

        correct_count = sum(result["correct"] for result in predictions)
        accuracy = correct_count / len(predictions)
        mean_inference_seconds = statistics.fmean(inference_times)
        total_response_times = [
            prediction["preprocessing_seconds"] + prediction["inference_seconds"]
            for prediction in predictions
        ]
        safe_model_name = re.sub(
            r"[^A-Za-z0-9_.-]+", "_", self.model_path
        ).strip("._") or "model"
        result_path = self.output_dir / (
            f"{self.category}_{safe_model_name}_"
            f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S.%fZ}_"
            f"{uuid.uuid4().hex}.json"
        )
        result = {
            "engine": "transformer",
            "model": self.model_path,
            "dataset": str(self.dataset_folder),
            "category": self.category,
            "device_map": self.device_map,
            "dtype": self.dtype,
            "model_load_seconds": model_load_seconds,
            "sample_count": len(predictions),
            "correct_count": correct_count,
            "accuracy": accuracy,
            "mean_preprocessing_seconds": statistics.fmean(preprocessing_times),
            "mean_inference_seconds": mean_inference_seconds,
            "mean_response_seconds": statistics.fmean(total_response_times),
            "p95_response_seconds": self._percentile_95(total_response_times),
            "predictions": predictions,
            "result_file": str(result_path),
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with result_path.open("w", encoding="utf-8") as output_file:
            json.dump(result, output_file, indent=2, sort_keys=True, allow_nan=False)
            output_file.write("\n")
        self._logger.add_metadata(
            result_file=str(result_path),
            accuracy=accuracy,
            mean_response_seconds=result["mean_response_seconds"],
            p95_response_seconds=result["p95_response_seconds"],
        )
        self._logger.info("category_evaluation_completed")
        return result