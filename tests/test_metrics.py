import unittest

import numpy as np

from struct_xai.metrics import (
    count_sign_flips,
    first_sustained_positive_layer,
    stable_positive_from_layer,
    summarize_trajectory,
    target_distractor_gap,
)


class GapMetricTests(unittest.TestCase):
    def test_target_distractor_gap(self) -> None:
        result = target_distractor_gap([1.0, 2.5], [0.5, 1.0])
        np.testing.assert_allclose(result, [0.5, 1.5])

    def test_gap_rejects_mismatched_lengths(self) -> None:
        with self.assertRaisesRegex(ValueError, "same shape"):
            target_distractor_gap([1.0], [1.0, 2.0])

    def test_first_sustained_layer_uses_external_layer_numbers(self) -> None:
        layer = first_sustained_positive_layer(
            [-0.2, 0.1, 0.3, -0.1],
            min_consecutive=2,
            layer_numbers=[4, 6, 8, 10],
        )
        self.assertEqual(layer, 6)

    def test_stable_layer_is_not_a_trivial_final_layer(self) -> None:
        self.assertEqual(
            stable_positive_from_layer([-0.1, 0.2, 0.3], min_remaining=2),
            1,
        )
        self.assertIsNone(stable_positive_from_layer([-0.1, 0.2, -0.3], min_remaining=2))

    def test_sign_flips_ignore_dead_zone(self) -> None:
        self.assertEqual(count_sign_flips([-1.0, 0.01, 1.0, -1.0], threshold=0.1), 2)

    def test_summary_keeps_raw_evidence_traceable(self) -> None:
        summary = summarize_trajectory(
            model_id="model",
            example_id="example",
            target_token=" target",
            distractor_token=" distractor",
            target_logits=[0.0, 0.8, 1.4, 1.2],
            distractor_logits=[0.2, 0.4, 0.6, 0.7],
            layer_numbers=[2, 4, 6, 8],
            top_tokens=["a", "b", "target", "target"],
        )

        self.assertEqual(summary.decision_layer, 4)
        self.assertEqual(summary.stable_from_layer, 4)
        self.assertEqual(summary.peak_layer, 6)
        self.assertAlmostEqual(summary.final_gap, 0.5)
        self.assertEqual(summary.layers[2].top_token, "target")


if __name__ == "__main__":
    unittest.main()
