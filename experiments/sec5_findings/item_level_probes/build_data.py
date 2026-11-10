#!/usr/bin/env python
"""Build the Open LLM Leaderboard v2 model-by-question answer matrix.

Source: the per-question results released with Zhang, Dorner and Hardt (2025),
https://github.com/socialfoundations/benchmark-prediction at commit
SOURCE_COMMIT (`data/scores/<benchmark>.jbl` and `<benchmark>_models.jbl`).
Columns of each source matrix concatenate that benchmark's subtasks in the
order of `data_utils.OpenLLM.get_meta_infos.task2subtasks`.

Output: one compressed npz with
  answers          uint8 (n_models, n_questions), 1 = correct
  question_column  int16 (n_questions,), subtask column index of each question
  column_ids       str   (37,), "<benchmark>/<subtask>"
  model_ids        str   (n_models,), sorted
plus `<out>.manifest.json` with source and output sha256.
"""

import argparse
import hashlib
import json
import os
import subprocess

import joblib
import numpy as np

SOURCE_REPO = 'https://github.com/socialfoundations/benchmark-prediction'
SOURCE_COMMIT = '5b7a23e32ee738c69b4c408cb9746429a7fd0043'

# (source file stem, [(subtask, n_questions), ...]) in source column order.
BENCHMARKS = [
    ('bbh', [
        ('boolean_expressions', 250), ('causal_judgement', 187),
        ('date_understanding', 250), ('disambiguation_qa', 250),
        ('formal_fallacies', 250), ('geometric_shapes', 250),
        ('hyperbaton', 250), ('logical_deduction_five_objects', 250),
        ('logical_deduction_seven_objects', 250),
        ('logical_deduction_three_objects', 250),
        ('movie_recommendation', 250), ('navigate', 250),
        ('object_counting', 250), ('penguins_in_a_table', 146),
        ('reasoning_about_colored_objects', 250), ('ruin_names', 250),
        ('salient_translation_error_detection', 250), ('snarks', 178),
        ('sports_understanding', 250), ('temporal_sequences', 250),
        ('tracking_shuffled_objects_five_objects', 250),
        ('tracking_shuffled_objects_seven_objects', 250),
        ('tracking_shuffled_objects_three_objects', 250),
        ('web_of_lies', 250),
    ]),
    ('gpqa', [('diamond', 198), ('extended', 546), ('main', 448)]),
    ('ifeval', [('ifeval', 541)]),
    ('openllm_math', [
        ('geometry_hard', 132), ('intermediate_algebra_hard', 280),
        ('num_theory_hard', 154), ('prealgebra_hard', 193),
        ('precalculus_hard', 135),
    ]),
    ('mmlu_pro', [('mmlu_pro', 12032)]),
    ('musr', [
        ('murder_mysteries', 250), ('object_placements', 256),
        ('team_allocation', 250),
    ]),
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def source_commit(source_dir):
    out = subprocess.run(['git', '-C', source_dir, 'rev-parse', 'HEAD'],
                         check=True, capture_output=True, text=True)
    return out.stdout.strip()


def load_benchmark(source_dir, stem, subtasks):
    """Return {model_id: answer row} for one benchmark, rows deduplicated."""
    scores_dir = os.path.join(source_dir, 'data', 'scores')
    scores = np.asarray(joblib.load(os.path.join(scores_dir, f'{stem}.jbl')))
    models = list(joblib.load(os.path.join(scores_dir, f'{stem}_models.jbl')))
    if scores.shape != (len(models), sum(n for _, n in subtasks)):
        raise ValueError(f'{stem}: shape {scores.shape} does not match '
                         f'{len(models)} models x subtask sizes')
    values = np.unique(scores)
    if not set(values.tolist()) <= {0, 1}:
        raise ValueError(f'{stem}: non-binary answers {values[:10]}')
    rows = {}
    for model, row in zip(models, scores.astype(np.uint8)):
        if model in rows and not np.array_equal(rows[model], row):
            raise ValueError(f'{stem}: duplicate model {model} with different rows')
        rows.setdefault(model, row)
    return rows


def build(source_dir):
    per_bench = [load_benchmark(source_dir, stem, subtasks)
                 for stem, subtasks in BENCHMARKS]
    model_ids = sorted(set.intersection(*(set(r) for r in per_bench)))
    answers = np.concatenate(
        [np.stack([rows[m] for m in model_ids]) for rows in per_bench], axis=1)
    column_ids, question_column = [], []
    for stem, subtasks in BENCHMARKS:
        for subtask, n in subtasks:
            question_column += [len(column_ids)] * n
            column_ids.append(f'{stem}/{subtask}')
    return {
        'answers': answers,
        'question_column': np.asarray(question_column, dtype=np.int16),
        'column_ids': np.asarray(column_ids),
        'model_ids': np.asarray(model_ids),
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--source-dir', required=True,
                    help=f'git checkout of {SOURCE_REPO} at {SOURCE_COMMIT}')
    ap.add_argument('--out', default=os.path.join(here, 'results', 'data', 'oll_items.npz'))
    args = ap.parse_args()

    commit = source_commit(args.source_dir)
    if commit != SOURCE_COMMIT:
        raise SystemExit(f'source checkout is at {commit}, need {SOURCE_COMMIT}')
    data = build(args.source_dir)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **data)

    scores_dir = os.path.join(args.source_dir, 'data', 'scores')
    manifest = {
        'source_repo': SOURCE_REPO,
        'source_commit': commit,
        'source_sha256': {
            name: sha256_file(os.path.join(scores_dir, name))
            for stem, _ in BENCHMARKS
            for name in (f'{stem}.jbl', f'{stem}_models.jbl')
        },
        'n_models': int(data['answers'].shape[0]),
        'n_questions': int(data['answers'].shape[1]),
        'n_columns': int(len(data['column_ids'])),
        'output_sha256': sha256_file(args.out),
    }
    with open(args.out + '.manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'source_sha256'}))


if __name__ == '__main__':
    main()
