import pathlib
import time
import typing

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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