# Jetson Benchmarker

In this repository i aim to demonstrate the performance of different inference apps (transformers, ollama, etc). In anomaly detection mainly using the [MVTech AD Dataset](https://www.mvtec.com/research-teaching/datasets/mvtec-ad).

The benchmark will ask the models (open weight models downloaded from Hugging Face) to detect anomalies within the dataset, and produces a JSON file indicating batch accuracy, mean response time, 95th percentile response time, inference vs preprocessing time if applicable.

## Command-line setup validation

The current command-line interface validates the dataset folder and model configuration; it does not yet run image inference, inspect dataset contents, or produce anomaly metrics. The Transformer engine accepts a Hugging Face model ID or a local model directory. It loads the model configuration only, not the model weights.

Run it from the repository root:

```bash
PYTHONPATH=.src python .src/benchmarker.py \
  --dataset-folder /path/to/mvtec_ad \
  --model google/vit-base-patch16-224 \
  --engine transformer
```

For an offline/local Transformers model, pass its directory as `--model`:

```bash
PYTHONPATH=.src python .src/benchmarker.py \
  --dataset-folder /path/to/mvtec_ad \
  --model /path/to/local/model \
  --engine transformer
```

The supported engine names correspond to packages under `.src/testerframework/`:

- `transformer`: validates a Hugging Face model configuration from a Hub ID or local directory.
- `llama_cpp_python`: expects an existing local `.gguf` file, but exits with an explicit unsupported message because this package does not yet contain a runnable tester.
- `llama_serve`: is listed as a package, but has no runnable tester or defined endpoint argument contract yet.

`--dataset-folder` must identify an existing directory. The current validator does not verify that it contains a correctly structured MVTec AD dataset. Add `--trust-remote-code` only for model configurations that require custom Hugging Face code.

## Repository structure
This repository is strictured using the following directories:
### `.src`
This contains the main swftware project, written mainly in python

### `.dep`

### `.doc`

The documentations of:
-  `sw`: the software itself
-  `results` the findings form the tests ran by the software

### `.dpl`
 The deployment directory, here runtime files can be found, like input files, output files