"""
Data loading: single import point for evaluation harness + methods.
"""
import sys, os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BP_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'bp')

from benchpress.evaluation_harness import (
    M_FULL, OBSERVED, N_MODELS, N_BENCH, MODEL_IDS, BENCH_IDS,
    MODEL_NAMES, BENCH_NAMES, MODEL_REASONING, MODEL_PROVIDERS, BENCH_CATS,
    MODEL_IDX, BENCH_IDX,
)
from benchpress.all_methods import predict_benchpress_scores
