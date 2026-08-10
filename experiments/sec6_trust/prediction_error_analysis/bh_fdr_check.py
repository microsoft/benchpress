#!/usr/bin/env python3
"""Benjamini-Hochberg FDR correction across the 32 §6.1 hypothesis tests.

We report 16 hypotheses (7 benchmark-side, 9 model-side) under two error
metrics (MedAPE, MedAE), giving 32 tests. This script applies the
Benjamini-Hochberg procedure over all 32 headline p-values and reports which
survive, backing the multiple-testing statement in `app:reliability_stat_tests`.

Provenance: the 32 headline p-values are exactly the ones displayed in the two
main-text tables, `tab:predictability_factors` (benchmark H1-H7) and
`tab:model_hypotheses` (model H1-H9), produced by the sibling
`benchmark_analysis/gen_table.py` and `model_analysis/gen_table.py`. Values shown
in those tables as `p<0.001` are recorded here as a representative sub-1e-3 value;
the alpha=0.01 conclusion depends only on their rank below the 0.004 cutoff, not
on their exact magnitude (see the assertion at the end).
"""

SUB_1EM3 = 5e-4  # table shows "p<0.001"; exact value is immaterial at alpha=0.01

# (family, hypothesis, metric, p). "supported" factors are those the paper
# reports as jointly rejected under BOTH metrics.
TESTS = [
    # benchmark-side (tab:predictability_factors)
    ("bench", "H1", "medape", 0.150), ("bench", "H1", "medae", 0.036),
    ("bench", "H2", "medape", SUB_1EM3), ("bench", "H2", "medae", 0.054),
    ("bench", "H3", "medape", SUB_1EM3), ("bench", "H3", "medae", SUB_1EM3),
    ("bench", "H4", "medape", SUB_1EM3), ("bench", "H4", "medae", SUB_1EM3),
    ("bench", "H5", "medape", SUB_1EM3), ("bench", "H5", "medae", SUB_1EM3),
    ("bench", "H6", "medape", 0.209), ("bench", "H6", "medae", 0.035),
    ("bench", "H7", "medape", 0.832), ("bench", "H7", "medae", 0.725),
    # model-side (tab:model_hypotheses)
    ("model", "H1", "medape", 0.101), ("model", "H1", "medae", 0.263),
    ("model", "H2", "medape", SUB_1EM3), ("model", "H2", "medae", 0.003),
    ("model", "H3", "medape", SUB_1EM3), ("model", "H3", "medae", SUB_1EM3),
    ("model", "H4", "medape", 0.389), ("model", "H4", "medae", 0.042),
    ("model", "H5", "medape", SUB_1EM3), ("model", "H5", "medae", 0.004),
    ("model", "H6", "medape", 0.033), ("model", "H6", "medae", 0.309),
    ("model", "H7", "medape", 0.011), ("model", "H7", "medae", 0.094),
    ("model", "H8", "medape", SUB_1EM3), ("model", "H8", "medae", SUB_1EM3),
    ("model", "H9", "medape", 0.002), ("model", "H9", "medae", 0.002),
]

# Factors the paper reports as jointly supported (rejected under both metrics).
SUPPORTED = {("bench", "H3"), ("bench", "H4"), ("bench", "H5"),
             ("model", "H2"), ("model", "H3"), ("model", "H5"),
             ("model", "H8"), ("model", "H9")}

ALPHA = 0.01


def benjamini_hochberg(pvals, alpha):
    """Return the set of indices rejected by BH at level alpha."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    k_max = 0
    for rank, idx in enumerate(order, start=1):
        if pvals[idx] <= rank / m * alpha:
            k_max = rank
    return set(order[:k_max]), k_max


def main():
    pvals = [t[3] for t in TESTS]
    rejected, k = benjamini_hochberg(pvals, ALPHA)
    m = len(TESTS)
    cutoff = max((pvals[i] for i in rejected), default=0.0)

    print(f"{m} tests, Benjamini-Hochberg at alpha={ALPHA}: reject {k} tests "
          f"(largest surviving p = {cutoff:g}).")

    # Every metric of every supported factor must survive.
    supported_survive = all(
        i in rejected
        for i, (fam, h, _m, _p) in enumerate(TESTS)
        if (fam, h) in SUPPORTED
    )
    # No factor outside SUPPORTED should become jointly significant under BH.
    newly = sorted({
        (fam, h) for i, (fam, h, _m, _p) in enumerate(TESTS)
        if i in rejected and (fam, h) not in SUPPORTED
    } & {  # only count as "new" if BOTH metrics of that factor are rejected
        (fam, h) for fam, h in {(t[0], t[1]) for t in TESTS}
        if all(i in rejected for i, (f2, h2, _m2, _p2) in enumerate(TESTS)
               if (f2, h2) == (fam, h))
    })

    print(f"All 8 supported factors significant under both metrics after BH: "
          f"{supported_survive}")
    print(f"Factors newly jointly-significant only under BH: "
          f"{newly if newly else 'none'}")

    assert k == 17, f"expected 17 rejections, got {k}"
    assert supported_survive, "a supported factor failed BH correction"
    assert not newly, f"unexpected newly-significant factors: {newly}"
    print("OK: matches app:reliability_stat_tests.")


if __name__ == "__main__":
    main()
