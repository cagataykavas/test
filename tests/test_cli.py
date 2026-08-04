import json
import tempfile
import unittest
from pathlib import Path

from struct_xai.cli import run


class CliTests(unittest.TestCase):
    def test_cli_writes_explainable_report(self) -> None:
        payload = {
            "model_id": "model",
            "example_id": "example",
            "target_token": " target",
            "distractor_token": " distractor",
            "target_logits": [0.0, 1.0, 2.0],
            "distractor_logits": [0.5, 0.4, 0.3],
            "ablations": [
                {
                    "feature": "cue",
                    "target_logits": [0.0, 0.3, 0.5],
                    "distractor_logits": [0.5, 0.4, 0.3],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.json"
            output_path = Path(tmp) / "report.json"
            input_path.write_text(json.dumps(payload), encoding="utf-8")

            exit_code = run([str(input_path), "--output", str(output_path), "--pretty"])

            self.assertEqual(exit_code, 0)
            report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(report["schema_version"], "1.0")
            self.assertEqual(report["trajectory"]["decision_layer"], 1)
            self.assertEqual(report["ablations"][0]["feature"], "cue")
            self.assertIn("support_effect", report["definitions"])
            self.assertTrue(report["limitations"])

    def test_cli_returns_nonzero_for_missing_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.json"
            input_path.write_text("{}", encoding="utf-8")
            self.assertEqual(run([str(input_path)]), 2)


if __name__ == "__main__":
    unittest.main()
