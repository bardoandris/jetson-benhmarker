# Copilot instructions for Jetson Benchmarker

## Project purpose

- This project aims to compare inference applications such as Transformers and Ollama on an NVIDIA Jetson, primarily for visual anomaly detection using the MVTec AD dataset.
- The intended workload uses open-weight models, including multimodal vision-language models, to assess images and detect anomalies.
- The planned benchmark output is JSON with batch accuracy, mean response time, 95th-percentile response time, and inference/preprocessing time when those timings can be separated.
- Keep intended functionality distinct from implemented functionality: the current Transformers tester is a basic causal text-generation prototype. Do not describe it as an MVTec AD image evaluation or assume the dataset pipeline, anomaly scoring, or cross-runtime comparison already exists.

## Repository shape

- The application code lives under `.src/`, not under a conventional top-level `src/` package.
- The entry point is `.src/benchmarker.py`, which dispatches to `.src/testerframework/cli.py`; `.src/py/main.py` is a thin launcher.
- The CLI in `.src/testerframework/cli.py` runs image-level inference for one MVTec AD category with a Transformers vision-language model; `--validate-only` checks the dataset directory/model config without loading weights or inferring.
- Benchmark implementations are organized under `.src/testerframework/`:
  - `.src/testerframework/base_test/base_tester.py` provides the shared run lifecycle and JSON event logging.
  - Backend-specific testers belong in their own subpackages (for example, `.src/testerframework/transformer/`).
  - `.src/testerframework/transformer/transformer_testers.py` includes both a causal text-generation prototype and a category-level MVTec tester that prompts a vision-language model for strict `normal`/`anomalous` image labels.
- `.dpl/` is intended for deployment/runtime files and generated outputs such as logs. Keep generated artifacts out of source changes unless explicitly required.
- `.doc/` is for software and benchmark-results documentation; `.dep/` is a repository-level dependency area.

## Build, test, and lint commands

- There is no project-level test runner, lint config, or build system checked into this repository right now (`pytest`, `tox`, `ruff`, `black`, `mypy`, `Makefile`, `pyproject.toml`, and similar config files are not present).
- Focused MVTec tester tests:
  - Full focused module: `PYTHONPATH=.src python -m unittest testerframework.transformer.test_transformer_mvtec -v`
  - Single test: `PYTHONPATH=.src python -m unittest testerframework.transformer.test_transformer_mvtec.TransformerMvtecTesterTests.test_run_writes_predictions_and_metrics -v`
- The repository still has no configured lint/build system. For syntax checks, target the files being changed rather than assuming all experimental backend stubs compile.

## High-level architecture

The current codebase is a small Python benchmark harness, while the README describes a broader planned image-anomaly benchmark:

- The current Transformers MVTec flow evaluates the selected category's `test/good` and `test/<defect>` images, calculates image-level binary accuracy and preprocessing/inference/response latency summaries, and writes per-image predictions plus aggregate metrics under `.dpl/results/`.
- The existing shared `base_tester.start_test()` lifecycle creates a per-run JSON log, executes the implementation, and records completion or failure.
- Tester subclasses implement `_run_test()` and use `self._logger.add_metadata(...)` plus standard logger methods to record backend/model details and benchmark events.
- Per-run event logs are written under `.dpl/logs`. Treat these structured event logs separately from the planned benchmark summary containing batch-level accuracy and latency aggregates.
- Keep inference-backend-specific loading and execution isolated from shared lifecycle and evaluation logic. Only introduce shared evaluation components when they are supported by multiple backends or required by the benchmark contract.

## Key conventions

- Treat `.src/` as the effective import root when running Python commands: `PYTHONPATH=.src ...` is the safe pattern in this repo.
- New benchmark implementations should subclass `base_tester` and override `_run_test()`, rather than reimplementing the logging lifecycle in each backend.
- Preserve the metadata/logging contract: metadata must be JSON-serializable; retain base fields such as `test_name`, `started_at`, and final `status`.
- For comparable backend results, use the same dataset split, image inputs, prompts/task instructions, and accuracy/latency definitions. Record model, backend, device/runtime, and workload details as metadata where available.
- MVTec image-level ground truth is derived from test subfolder names (`good` is normal; defect folders are anomalous); do not present this as pixel-level mask evaluation.
- Measure inference and preprocessing separately when practical; use the README's required mean and 95th-percentile response-time metrics for benchmark summaries.
- Keep backend-specific code inside its own subpackage instead of scattering model loading or execution into the top-level runner.
- Do not commit generated output under `.dpl/` unless the task explicitly requires it.

## Working style in this repo

- The repository is still small and prototype-like; prefer surgical edits that follow the existing benchmark abstraction rather than introducing a new framework layer.
- Keep changes aligned with the repository’s current conventions: `.src`-rooted layout, backend subpackages, and JSON metadata logging.
