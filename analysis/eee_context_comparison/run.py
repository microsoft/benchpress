#!/usr/bin/env python
"""Compare BenchPress and EEE context on identical audited target cells."""

import argparse
import os

import numpy as np

from common import load_filtered_eee
from evaluate import run_evaluation
from inventory import write_candidate_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eee-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mapping")
    parser.add_argument(
        "--phase", choices=["inventory", "evaluate"], required=True)
    args = parser.parse_args()

    output = os.path.abspath(os.path.expanduser(args.output))
    eee = load_filtered_eee(args.eee_dir)
    print(
        f"EEE filtered matrix {eee['values'].shape} "
        f"({int(np.isfinite(eee['values']).sum())} observed cells)",
        flush=True,
    )
    if args.phase == "inventory":
        write_candidate_report(eee, output)
        return
    if not args.mapping:
        raise SystemExit("--mapping is required for --phase evaluate")
    run_evaluation(eee, os.path.expanduser(args.mapping), output)


if __name__ == "__main__":
    main()
