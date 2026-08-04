import unittest

from struct_xai.ablation import compare_ablation
from struct_xai.metrics import summarize_trajectory


def make_summary(example_id: str, target: list[float], distractor: list[float]):
    return summarize_trajectory(
        model_id="model",
        example_id=example_id,
        target_token=" target",
        distractor_token=" distractor",
        target_logits=target,
        distractor_logits=distractor,
        layer_numbers=[0, 1, 2, 3],
    )


class AblationTests(unittest.TestCase):
    def test_positive_effect_means_feature_supported_target(self) -> None:
        base = make_summary("base", [0.0, 1.0, 2.0, 3.0], [0.5, 0.5, 0.5, 0.5])
        ablated = make_summary("ablated", [0.0, 0.4, 0.8, 1.0], [0.5, 0.5, 0.5, 0.5])

        result = compare_ablation(base, ablated, feature="cue")

        self.assertGreater(result.mean_support_effect, 0.0)
        self.assertEqual(result.max_absolute_effect_layer, 3)
        self.assertAlmostEqual(result.final_support_effect, 2.0)
        self.assertEqual(result.decision_layer_shift, 1)

    def test_misaligned_layers_are_rejected(self) -> None:
        base = make_summary("base", [0.0, 1.0, 2.0, 3.0], [0.5, 0.5, 0.5, 0.5])
        ablated = summarize_trajectory(
            model_id="model",
            example_id="ablated",
            target_token=" target",
            distractor_token=" distractor",
            target_logits=[0.0, 1.0, 2.0, 3.0],
            distractor_logits=[0.5, 0.5, 0.5, 0.5],
            layer_numbers=[1, 2, 3, 4],
        )

        with self.assertRaisesRegex(ValueError, "same layers"):
            compare_ablation(base, ablated, feature="cue")


if __name__ == "__main__":
    unittest.main()
