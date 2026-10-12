# Copilot instructions for Jetson Benchmarker

## Project purpose

- This project aims to compare inference applications such as Transformers and Ollama on an NVIDIA Jetson, primarily for visual anomaly detection using the MVTec AD dataset.
- The intended workload uses open-weight models, including multimodal vision-language models, to assess images and detect anomalies.
- The planned benchmark output is JSON with batch accuracy, mean response time, 95th-percentile response time, and inference/preprocessing time when those timings can be separated.
- Keep intended functionality distinct from implemented functionality: the current Transformers tester is a basic causal text-generation prototype. Do not describe it as an MVTec AD image evaluation or assume the dataset pipeline, anomaly scoring, or cross-runtime comparison already exists.

## Repository shape

- The application code lives under `.src/`, not under a conventional top-level `src/` package.
- The entry point is `.src/benchmarker.py`, which dispatches to `.src/testerframework/cli.py`; `.src/py/main.py` is a thin launcher.
- The current CLI validates an existing dataset directory and the Transformers model configuration only. It does not inspect dataset samples, load model weights, run inference, or compute benchmark metrics.
- Benchmark implementations are organized under `.src/testerframework/`:
  - `.src/testerframework/base_test/base_tester.py` provides the shared run lifecycle and JSON event logging.
  - Backend-specific testers belong in their own subpackages (for example, `.src/testerframework/transformer/`).
  - `.src/testerframework/transformer/transformer_test.py` currently loads a Hugging Face causal language model, generates text, and measures model-load and generation time, token counts, and throughput.
- `.dpl/` is intended for deployment/runtime files and generated outputs such as logs. Keep generated artifacts out of source changes unless explicitly required.
- `.doc/` is for software and benchmark-results documentation; `.dep/` is a repository-level dependency area.

## Build, test, and lint commands

- There is no project-level test runner, lint config, or build system checked into this repository right now (`pytest`, `tox`, `ruff`, `black`, `mypy`, `Makefile`, `pyproject.toml`, and similar config files are not present).
- For quick validation in this repo, use Python syntax smoke checks against the `.src` tree rather than assuming a full test suite exists:
  - `PYTHONPATH=.src python -m compileall .src`
  - `PYTHONPATH=.src python -c "import benchmarker"`
- There are no discovered unit tests to run individually through a framework; if you need a single targeted validation for a specific benchmark backend, run the backend script or a short Python snippet that imports the relevant module directly.

## High-level architecture

The current codebase is a small Python benchmark harness, while the README describes a broader planned image-anomaly benchmark:

- The intended flow is to evaluate inference backends on the same MVTec AD image workload, calculate accuracy and latency statistics, and produce machine-readable JSON results suitable for comparing backends on Jetson hardware.
- The existing shared `base_tester.start_test()` lifecycle creates a per-run JSON log, executes the implementation, and records completion or failure.
- Tester subclasses implement `_run_test()` and use `self._logger.add_metadata(...)` plus standard logger methods to record backend/model details and benchmark events.
- Per-run event logs are written under `.dpl/logs`. Treat these structured event logs separately from the planned benchmark summary containing batch-level accuracy and latency aggregates.
- Keep inference-backend-specific loading and execution isolated from shared lifecycle and evaluation logic. Only introduce shared evaluation components when they are supported by multiple backends or required by the benchmark contract.

## Key conventions

- Treat `.src/` as the effective import root when running Python commands: `PYTHONPATH=.src ...` is the safe pattern in this repo.
- New benchmark implementations should subclass `base_tester` and override `_run_test()`, rather than reimplementing the logging lifecycle in each backend.
- Preserve the metadata/logging contract: metadata must be JSON-serializable; retain base fields such as `test_name`, `started_at`, and final `status`.
- For comparable backend results, use the same dataset split, image inputs, prompts/task instructions, and accuracy/latency definitions. Record model, backend, device/runtime, and workload details as metadata where available.
- Measure inference and preprocessing separately when practical; use the README's required mean and 95th-percentile response-time metrics for benchmark summaries.
- Keep backend-specific code inside its own subpackage instead of scattering model loading or execution into the top-level runner.
- Do not commit generated output under `.dpl/` unless the task explicitly requires it.

## Working style in this repo

- The repository is still small and prototype-like; prefer surgical edits that follow the existing benchmark abstraction rather than introducing a new framework layer.
- Keep changes aligned with the repository’s current conventions: `.src`-rooted layout, backend subpackages, and JSON metadata logging.
