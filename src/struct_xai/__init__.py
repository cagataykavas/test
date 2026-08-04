"""Public API for the Struct-XAI analysis core."""

from struct_xai.ablation import AblationSummary, LayerEffect, compare_ablation
from struct_xai.metrics import (
    count_sign_flips,
    first_sustained_positive_layer,
    stable_positive_from_layer,
    summarize_trajectory,
    target_distractor_gap,
)
from struct_xai.report import build_explainable_report, write_explainable_report
from struct_xai.schema import LayerEvidence, TrajectorySummary

__all__ = [
    "AblationSummary",
    "LayerEffect",
    "LayerEvidence",
    "TrajectorySummary",
    "build_explainable_report",
    "compare_ablation",
    "count_sign_flips",
    "first_sustained_positive_layer",
    "stable_positive_from_layer",
    "summarize_trajectory",
    "target_distractor_gap",
    "write_explainable_report",
]

__version__ = "0.1.0"
