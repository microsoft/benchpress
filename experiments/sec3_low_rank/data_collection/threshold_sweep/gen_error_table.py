#!/usr/bin/env python3
"""Generate the appendix filtering-threshold robustness table."""

import json
import os


HERE = os.path.dirname(__file__)
RESULTS_PATH = os.path.join(HERE, "error_results.json")
GRID = [
    (10, 8),
    (10, 12),
    (10, 16),
    (15, 8),
    (15, 12),
    (15, 16),
    (20, 8),
    (20, 12),
    (20, 16),
]


def styled(value: str, selected: bool) -> str:
    if not selected:
        return value
    return rf"\cBP\textbf{{{value}}}"


def main() -> None:
    with open(RESULTS_PATH, encoding="utf-8") as f:
        results = json.load(f)

    configs = {
        (row["t_m"], row["t_b"]): row
        for row in results["configs"]
    }
    unfiltered = configs[(0, 0)]
    adopted = (
        results["protocol"]["adopted"]["t_m"],
        results["protocol"]["adopted"]["t_b"],
    )

    print(r"\begin{tabular}{ccrrrr}")
    print(r"\toprule")
    print(r"\multicolumn{2}{c}{Min.\ obs.\ per} & \#Obs. & Fill & MedAE & MedAPE \\")
    print(r"\cmidrule(lr){1-2}")
    print(r"Model & Bench. & & & $\downarrow$ & $\downarrow$ \\")
    print(r"\midrule")
    native = unfiltered["native"]
    observations = f"{unfiltered['n_observations']:,}".replace(",", "{,}")
    print(
        r"\multicolumn{2}{c}{Canonicalized, no density filter}"
        f" & {observations}"
        f" & {100 * unfiltered['fill_rate']:.1f}\\%"
        f" & {native['medae_median']:.2f}"
        f" & {native['medape_median']:.2f}\\% \\\\"
    )
    print(r"\midrule")

    for thresholds in GRID:
        row = configs.get(thresholds)
        if row is None:
            print(
                f"{thresholds[0]} & {thresholds[1]}"
                r" & \multicolumn{4}{c}{Empty after filtering} \\"
            )
            continue

        selected = thresholds == adopted
        native = row["native"]
        values = [
            str(thresholds[0]),
            str(thresholds[1]),
            f"{row['n_observations']:,}".replace(",", "{,}"),
            f"{100 * row['fill_rate']:.1f}\\%",
            f"{native['medae_median']:.2f}",
            f"{native['medape_median']:.2f}\\%",
        ]
        print(" & ".join(styled(value, selected) for value in values) + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
