"""Shim: old `from evaluation import ...` → `segpos.eval.metrics`."""

from segpos.eval.metrics import (
    confusion_matrix,
    evaluate_error_sources,
    pos_accuracy,
    print_tiny_confusion_matrix,
    segmentation_accuracy,
    spanish_segmentation_accuracy,
)

__all__ = [
    "confusion_matrix",
    "evaluate_error_sources",
    "pos_accuracy",
    "print_tiny_confusion_matrix",
    "segmentation_accuracy",
    "spanish_segmentation_accuracy",
]
