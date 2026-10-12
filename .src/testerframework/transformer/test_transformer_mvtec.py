import json
import pathlib
import tempfile
import unittest
from unittest import mock
from contextlib import redirect_stdout
from io import StringIO

import torch

from testerframework.transformer.transformer_testers import transformer_mvtec_tester


class _FakeBatch(dict):
    def to(self, device):
        return self


class _FakeProcessor:
    def apply_chat_template(self, conversation, **kwargs):
        image = conversation[0]["content"][0]["image"]
        pixel_value = image.getpixel((0, 0))[0]
        return _FakeBatch(
            input_ids=torch.tensor([[1, 2]]),
            pixel_values=torch.tensor([pixel_value]),
        )

    def batch_decode(self, output_ids, **kwargs):
        return ["anomalous" if int(output_ids[0, -1]) == 9 else "normal"]


class _FakeModel:
    device = torch.device("cpu")

    def eval(self):
        return self

    def generate(self, input_ids, pixel_values, **kwargs):
        labels = torch.where(
            pixel_values > 128,
            torch.tensor([9]),
            torch.tensor([8]),
        ).unsqueeze(1)
        return torch.cat((input_ids, labels), dim=1)


class TransformerMvtecTesterTests(unittest.TestCase):
    def test_run_writes_predictions_and_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            category_root = root / "mvtec" / "bottle" / "test"
            normal_dir = category_root / "good"
            anomaly_dir = category_root / "broken"
            normal_dir.mkdir(parents=True)
            anomaly_dir.mkdir(parents=True)

            from PIL import Image

            Image.new("RGB", (2, 2), (0, 0, 0)).save(normal_dir / "normal.png")
            Image.new("RGB", (2, 2), (255, 0, 0)).save(anomaly_dir / "anomaly.png")

            tester = transformer_mvtec_tester(
                model_path="test/multimodal-model",
                dataset_folder=root / "mvtec",
                category="bottle",
                output_dir=root / "results",
                log_dir=root / "logs",
            )
            with mock.patch(
                "testerframework.transformer.transformer_testers.AutoProcessor.from_pretrained",
                return_value=_FakeProcessor(),
            ), mock.patch(
                "testerframework.transformer.transformer_testers.AutoModelForImageTextToText.from_pretrained",
                return_value=_FakeModel(),
            ):
                result = tester.start_test()

            self.assertEqual(result["sample_count"], 2)
            self.assertEqual(result["correct_count"], 2)
            self.assertEqual(result["accuracy"], 1.0)
            self.assertGreaterEqual(result["mean_response_seconds"], 0.0)
            self.assertGreaterEqual(result["p95_response_seconds"], 0.0)
            self.assertTrue(pathlib.Path(result["result_file"]).is_file())
            saved_result = json.loads(
                pathlib.Path(result["result_file"]).read_text(encoding="utf-8")
            )
            self.assertEqual(saved_result["accuracy"], 1.0)
            self.assertEqual(
                {prediction["predicted"] for prediction in saved_result["predictions"]},
                {"normal", "anomalous"},
            )

    def test_rejects_missing_category_test_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            tester = transformer_mvtec_tester(
                model_path="test/multimodal-model",
                dataset_folder=temporary,
                category="bottle",
                output_dir=pathlib.Path(temporary) / "results",
                log_dir=pathlib.Path(temporary) / "logs",
            )
            with self.assertRaises(FileNotFoundError):
                tester._get_test_images()

    def test_rejects_non_label_model_response(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "Expected exactly"):
                transformer_mvtec_tester._parse_prediction(
                    "This image looks defective.",
                    pathlib.Path("bottle/test/broken/image.png"),
                )

    def test_cli_runs_category_and_prints_result_path(self):
        from testerframework.cli import main

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            normal_dir = root / "mvtec" / "bottle" / "test" / "good"
            anomaly_dir = root / "mvtec" / "bottle" / "test" / "broken"
            normal_dir.mkdir(parents=True)
            anomaly_dir.mkdir(parents=True)

            from PIL import Image

            Image.new("RGB", (2, 2), (0, 0, 0)).save(normal_dir / "normal.png")
            Image.new("RGB", (2, 2), (255, 0, 0)).save(anomaly_dir / "anomaly.png")

            output = StringIO()
            with mock.patch(
                "testerframework.transformer.transformer_testers.AutoProcessor.from_pretrained",
                return_value=_FakeProcessor(),
            ), mock.patch(
                "testerframework.transformer.transformer_testers.AutoModelForImageTextToText.from_pretrained",
                return_value=_FakeModel(),
            ), redirect_stdout(output):
                status = main(
                    [
                        "--dataset-folder",
                        str(root / "mvtec"),
                        "--category",
                        "bottle",
                        "--model",
                        "test/multimodal-model",
                        "--engine",
                        "transformer",
                        "--output-dir",
                        str(root / "results"),
                        "--log-dir",
                        str(root / "logs"),
                    ]
                )

            self.assertEqual(status, 0)
            self.assertIn("accuracy=1.0000", output.getvalue())
            self.assertIn(str(root / "results"), output.getvalue())


if __name__ == "__main__":
    unittest.main()
