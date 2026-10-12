# Jetson Benchmarker

In this repository i aim to demonstrate the performance of different inference apps (transformers, ollama, etc). In anomaly detection mainly using the [MVTech AD Dataset](https://www.mvtec.com/research-teaching/datasets/mvtec-ad).

The benchmark will ask the models (open weight models downloaded from Hugging Face) to detect anomalies within the dataset, and produces a JSON file indicating batch accuracy, mean response time, 95th percentile response time, inference vs preprocessing time if applicable.

## Run an MVTec AD category test

The `transformer` engine can run image-level normal/anomalous classification for one MVTec AD category. On the remote Jetson, provide the path where you extracted MVTec AD and a Hugging Face model ID or local model directory for an instruction-tuned vision-language model supported by `AutoModelForImageTextToText` and `AutoProcessor` (including a chat template that accepts images).

The dataset root should have the standard MVTec AD layout:

```text
<dataset-root>/<category>/test/good/*.png
<dataset-root>/<category>/test/<defect-type>/*.png
```

The `good` folder is treated as normal; every other test subfolder is treated as anomalous. This first implementation measures image-level binary classification, not pixel-level defect segmentation. It requires at least one normal and one anomalous image in the selected category.

From the repository root on the Jetson, set the variables to the actual paths/model you chose, then run:

```bash
DATASET_ROOT="/absolute/path/to/mvtec_anomaly_detection"
MODEL_ID="your-org/your-instruction-tuned-vision-language-model"
CATEGORY="bottle"

PYTHONPATH=.src python .src/benchmarker.py \
  --dataset-folder "$DATASET_ROOT" \
  --category "$CATEGORY" \
  --model "$MODEL_ID" \
  --engine transformer \
  --device-map auto \
  --dtype auto
```

For a model already stored on the Jetson, set `MODEL_ID` to its absolute local directory instead. The model is loaded and evaluated on each image; this is an actual inference run, not just a config validation. Ensure the Jetson environment has a Transformers-compatible, Jetson/CUDA-compatible PyTorch installation and Pillow. Do not assume that the workstation's CUDA/PyTorch wheels are suitable for Jetson.

By default, a JSON result containing per-image predictions, accuracy, mean preprocessing/inference/response time, and p95 response time is written under `.dpl/results/`. Structured lifecycle/event logs are written under `.dpl/logs/`. Override these with `--output-dir` and `--log-dir`. `--max-new-tokens` defaults to 8; the model must answer each image with exactly `normal` or `anomalous`, otherwise the run fails rather than scoring an ambiguous response. `--trust-remote-code` is opt-in; use it only when the chosen model requires custom code.

To check only the dataset directory and Hugging Face model configuration without loading weights or processing images, add `--validate-only`. The `llama_cpp_python` and `llama_serve` packages do not yet have runnable MVTec testers; the CLI reports that explicitly.

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