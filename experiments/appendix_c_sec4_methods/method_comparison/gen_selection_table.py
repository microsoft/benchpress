#!/usr/bin/env python3
"""Appendix C.2 model-selection comparison (tab:model_selection).

Reports the adopted BenchPress configuration under the two selection protocols:
the held-out-fold selection behind results.json, and the nested selection behind
results_nested.json. Both rows are ranked among the configurations that predict
every held-out cell, so the comparison is against the same population the main
text draws its full-coverage claims from.
"""
import os

from benchpress.io_utils import load_json

SEC4_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'sec4_building_benchpress', 'method_comparison',
)
MANIFEST_PATH = os.path.normpath(os.path.join(SEC4_DIR, 'manifest.json'))
NESTED_PATH = os.path.normpath(os.path.join(SEC4_DIR, 'results_nested.json'))

BENCHPRESS_TRANSFORM = 'logit'
BENCHPRESS_METHOD = 'Bias ALS'
BENCHPRESS_HP = {'rank': 2, 'lam': 0.1}


def main():
    completed = load_json(MANIFEST_PATH)['completed']
    nested_rows = load_json(NESTED_PATH)['pair_rows']

    population = [r for r in completed if r['coverage'] >= 1.0]
    if not population:
        raise RuntimeError(f'no full-coverage configuration in {MANIFEST_PATH}')

    held_out = next(
        r for r in completed
        if r['transform'] == BENCHPRESS_TRANSFORM
        and r['method'] == BENCHPRESS_METHOD
        and r['hp'] == BENCHPRESS_HP)
    nested = next(
        r for r in nested_rows
        if r['transform'] == BENCHPRESS_TRANSFORM
        and r['method'] == BENCHPRESS_METHOD)
    if nested['modal_hp'] != BENCHPRESS_HP:
        raise RuntimeError(
            f'nested selection picked {nested["modal_hp"]} for '
            f'{BENCHPRESS_TRANSFORM} x {BENCHPRESS_METHOD}, not {BENCHPRESS_HP}; '
            'the appendix claim that the choice is unchanged no longer holds')

    n = len(population)
    print(r'\begin{tabular}{@{}lrrrr@{}}')
    print(r'\toprule')
    print(r'Selection protocol & $\mathsf{MedAPE}$ (\%) $\downarrow$ & Rank & '
          r'$\mathsf{MedAE}$ $\downarrow$ & Rank \\')
    print(r'\midrule')
    for label, row in (('Held-out folds', held_out),
                       ('Nested validation', nested)):
        medape, medae = row['medape_median'], row['medae_median']
        rank_medape = sum(1 for r in population
                          if r['medape_median'] < medape) + 1
        rank_medae = sum(1 for r in population
                         if r['medae_median'] < medae) + 1
        print(f'{label} & {medape:.2f} & {rank_medape} / {n} & '
              f'{medae:.2f} & {rank_medae} / {n} \\\\')
    print(r'\bottomrule')
    print(r'\end{tabular}')


if __name__ == '__main__':
    main()
