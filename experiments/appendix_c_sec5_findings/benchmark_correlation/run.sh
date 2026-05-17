#!/bin/bash
# Table 3: Pairwise benchmark OLS analysis
# Generates: pairwise_ols_stats.json

cd "$(dirname "$0")"
python run_pairwise_ols.py
