"""Tests package for TinyRecursiveModels."""

from .evaluation import (
    evaluate_model,
    evaluate_model_all_difficulties,
    compare_models,
    print_comparison_table,
    plot_comparison,
    get_summary_stats,
    quick_evaluate,
    generate_puzzle,
    count_prediction_validity,
    EvaluationResult,
)

__all__ = [
    'evaluate_model',
    'evaluate_model_all_difficulties',
    'compare_models',
    'print_comparison_table',
    'plot_comparison',
    'get_summary_stats',
    'quick_evaluate',
    'generate_puzzle',
    'count_prediction_validity',
    'EvaluationResult',
]
