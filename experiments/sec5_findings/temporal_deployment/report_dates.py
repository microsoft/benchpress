"""Source-verifiable report-date intervals for score-matrix cells.

Each observed cell of the evaluation matrix cites one ``reference_url``. This
module assigns that cited document an interval ``[lower, upper]`` that contains
the date on which the document reported the cell's score, using only the URL,
the score-matrix JSON, and the checked-in ``report_date_sources.json``:

- ``x_status_post``: the post time decoded from the X status ID (exact).
- ``arxiv_version``: a versioned arXiv URL is dated by that version (exact).
- ``arxiv_latest_version``: an unversioned arXiv URL shows some version between
  v1 and the latest version, so the interval is ``[v1 date, latest date]``.
- ``launch_<kind>``: the official launch document of model L is dated at L's
  release date. Announcements, system cards, and model cards are exact unless a
  score record citing the URL belongs to a model released after L, which proves
  a later revision; revised documents and technical reports get only the lower
  bound.
- ``undated_source``: leaderboards, aggregators, product pages, repositories,
  and other live or undated pages, plus missing URLs, get no bound. A live page
  adds rows over time, so the latest model it lists bounds only its newest row.

A cell is ``after_target`` when ``lower > cutoff``, ``on_or_before_target`` when
``upper <= cutoff``, and ``undatable`` otherwise.
"""

from __future__ import annotations

import calendar
import datetime
import functools
import hashlib
import json
import os
import re

from benchpress.build_benchmark_matrix import DATA
from benchpress.build_benchmark_matrix.build_benchmark_matrix import _load_raw
from benchpress.evaluation_harness import BENCH_IDX, MODEL_IDX
from benchpress.io_utils import load_json

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_DATE_SOURCES_PATH = os.path.join(HERE, "report_date_sources.json")
REPORT_DATE_CLASSES = ("after_target", "on_or_before_target", "undatable")
MATRIX_AUDIT_STATUSES = ("verified", "verified_third_party")
X_EPOCH_MS = 1288834974657
X_STATUS_RE = re.compile(r"^https?://(?:www\.)?(?:x|twitter)\.com/[^/]+/status/(\d+)")
ARXIV_RE = re.compile(r"^https?://arxiv\.org/(?:abs|html|pdf)/(\d{4}\.\d{4,5})(?:v(\d+))?")


def date_interval(value) -> tuple[str | None, str | None]:
    """Return the first and last calendar day of a YYYY, YYYY-MM, or YYYY-MM-DD date."""
    if not isinstance(value, str) or not value.strip():
        return None, None
    parts = [int(part) for part in value.strip().split("-")]
    if len(parts) == 3:
        day = datetime.date(*parts).isoformat()
        return day, day
    if len(parts) == 2:
        last = calendar.monthrange(parts[0], parts[1])[1]
        return (datetime.date(parts[0], parts[1], 1).isoformat(),
                datetime.date(parts[0], parts[1], last).isoformat())
    if len(parts) == 1:
        return f"{parts[0]:04d}-01-01", f"{parts[0]:04d}-12-31"
    raise ValueError(f"Unparseable date {value!r}")


def _document_interval(url: str | None, sources: dict, release_dates: dict,
                       cohort_latest_start: dict) -> dict:
    if not url:
        return {"rule": "undated_source", "report_date_lower": None, "report_date_upper": None}
    x_match = X_STATUS_RE.match(url)
    if x_match:
        posted_ms = (int(x_match.group(1)) >> 22) + X_EPOCH_MS
        day = datetime.datetime.fromtimestamp(
            posted_ms / 1000, datetime.timezone.utc,
        ).date().isoformat()
        return {"rule": "x_status_post", "report_date_lower": day, "report_date_upper": day}
    arxiv_match = ARXIV_RE.match(url)
    if arxiv_match:
        paper = sources["arxiv_versions"]["papers"][arxiv_match.group(1)]
        if arxiv_match.group(2) is None:
            return {
                "rule": "arxiv_latest_version",
                "report_date_lower": paper["v1_date"],
                "report_date_upper": paper["latest_date"],
            }
        version = int(arxiv_match.group(2))
        version_dates = {1: paper["v1_date"], int(paper["latest_version"]): paper["latest_date"]}
        if version not in version_dates:
            raise KeyError(f"{url}: no recorded date for arXiv version v{version}")
        return {
            "rule": "arxiv_version",
            "report_date_lower": version_dates[version],
            "report_date_upper": version_dates[version],
        }
    launch = sources["launch_documents"].get(url)
    if launch is not None:
        launch_lower, launch_upper = date_interval(release_dates[launch["model_id"]])
        if launch_lower is None:
            raise ValueError(f"{url}: launch model {launch['model_id']} has no release date")
        revised = cohort_latest_start[url] > launch_upper
        exact = launch["kind"] in ("announcement", "system_card", "model_card") and not revised
        return {
            "rule": f"launch_{launch['kind']}",
            "launch_model_id": launch["model_id"],
            "document_revised": bool(revised),
            "report_date_lower": launch_lower,
            "report_date_upper": launch_upper if exact else None,
        }
    return {"rule": "undated_source", "report_date_lower": None, "report_date_upper": None}


@functools.cache
def matrix_cell_report_dates() -> dict[tuple[str, str], dict]:
    """Report-date interval for every observed evaluation-matrix cell.

    Cell URLs follow ``evaluation_harness.M_FULL``: the last ``DATA`` record for
    a (model, benchmark) pair sets both the score and the cited document.
    """
    sources = load_json(REPORT_DATE_SOURCES_PATH)
    raw = _load_raw()
    release_dates = {model["id"]: model.get("release_date") for model in raw["models"]}
    cohort_latest_start = {}
    for record in raw["scores"]:
        if record.get("audit_status") not in MATRIX_AUDIT_STATUSES:
            continue
        url = record.get("reference_url")
        start, _ = date_interval(release_dates.get(record["model_id"]))
        if url and start and start > cohort_latest_start.get(url, ""):
            cohort_latest_start[url] = start

    cell_urls = {
        (model_id, benchmark_id): url
        for model_id, benchmark_id, _, url in DATA
        if model_id in MODEL_IDX and benchmark_id in BENCH_IDX
    }
    documents = {}
    cells = {}
    for (model_id, benchmark_id), url in sorted(cell_urls.items()):
        if url not in documents:
            documents[url] = _document_interval(url, sources, release_dates, cohort_latest_start)
            lower = documents[url]["report_date_lower"]
            upper = documents[url]["report_date_upper"]
            if lower is not None and upper is not None and lower > upper:
                raise ValueError(f"{url}: report-date lower bound {lower} exceeds upper bound {upper}")
        cells[(model_id, benchmark_id)] = {
            "model_id": model_id,
            "benchmark_id": benchmark_id,
            "reference_url": url,
            **documents[url],
        }
    return cells


def report_date_class(cell: dict, cutoff_date: str) -> str:
    lower = cell["report_date_lower"]
    upper = cell["report_date_upper"]
    if lower is not None and lower > cutoff_date:
        return "after_target"
    if upper is not None and upper <= cutoff_date:
        return "on_or_before_target"
    return "undatable"


def report_date_table_sha256(cells: dict[tuple[str, str], dict]) -> str:
    rows = [cells[key] for key in sorted(cells)]
    text = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()
