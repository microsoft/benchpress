"""Helpers for reproducible experiment artifacts.

Experiment scripts should prefer the checked-in artifact when present. If it is
missing, they should run the documented upstream command before reading it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def _as_paths(paths: str | os.PathLike | Sequence[str | os.PathLike]) -> list[Path]:
    if isinstance(paths, (str, os.PathLike)):
        return [Path(paths)]
    return [Path(path) for path in paths]


def ensure_artifacts(
    paths: str | os.PathLike | Sequence[str | os.PathLike],
    command: Sequence[str],
    *,
    cwd: str | os.PathLike | None = None,
    description: str = "artifact",
) -> bool:
    """Ensure ``paths`` exist, running ``command`` if any are missing.

    Returns ``True`` if the command was run. Raises ``FileNotFoundError`` if the
    command completes but the expected artifact is still missing.
    """
    expected = _as_paths(paths)
    missing = [path for path in expected if not path.exists()]
    if not missing:
        return False

    cmd = [sys.executable if part == "{python}" else str(part) for part in command]
    cwd_path = Path(cwd) if cwd is not None else None
    print(
        f"[benchpress] Missing {description}: "
        + ", ".join(str(path) for path in missing),
        file=sys.stderr,
    )
    print(
        "[benchpress] Running upstream command: " + " ".join(cmd),
        file=sys.stderr,
    )
    subprocess.run(cmd, cwd=str(cwd_path) if cwd_path else None, check=True)

    still_missing = [path for path in expected if not path.exists()]
    if still_missing:
        raise FileNotFoundError(
            f"upstream command finished but {description} is still missing: "
            + ", ".join(str(path) for path in still_missing)
        )
    return True


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_method_comparison_results() -> None:
    root = repo_root()
    exp_dir = root / "experiments" / "sec4_building_benchpress" / "method_comparison"
    ensure_artifacts(
        [exp_dir / "results.json", exp_dir / "manifest.json"],
        ["{python}", exp_dir / "run.py", "--merge"],
        description="Section 4.2 method-comparison results",
    )


def ensure_default_predictions() -> None:
    root = repo_root()
    out_dir = root / "benchpress" / "evaluation" / "default_predictions" / "benchpress_default"
    ensure_artifacts(
        [
            out_dir / "predictions.npz",
            out_dir / "metadata.json",
            out_dir / "by_benchmark.json",
            out_dir / "by_model.json",
        ],
        ["{python}", "-m", "benchpress.default_predictions"],
        description="default BenchPress prediction artifacts",
    )
