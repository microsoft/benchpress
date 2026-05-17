#!/usr/bin/env python3
"""Shared LaTeX table formatting helpers."""


def format_hyperparameters(hp):
    """Format method hyperparameters for LaTeX tables."""
    if not hp:
        return "---"
    parts = []
    for key, value in hp.items():
        if key in ("top_k", "k"):
            parts.append(f"$k{{=}}{value}$")
        elif key == "min_r2":
            parts.append(f"$R^2_{{\\min}}{{=}}{value}$")
        elif key == "rank":
            parts.append(f"$r{{=}}{value}$")
        elif key == "lam":
            parts.append(f"$\\lambda{{=}}{value}$")
        elif key == "lr":
            parts.append(f"lr${{=}}{value}$")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)
