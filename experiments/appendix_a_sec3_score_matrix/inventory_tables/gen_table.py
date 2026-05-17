#!/usr/bin/env python3
"""Generate Appendix A.1 benchmark/model inventory tables.

Usage:
    cd ~/Documents/submission/benchpress/github
    python experiments/appendix_a_sec3_score_matrix/inventory_tables/gen_table.py
    python experiments/appendix_a_sec3_score_matrix/inventory_tables/gen_table.py --write-overleaf
"""
from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from benchpress.build_benchmark_matrix import load_score_matrix


ROOT = Path(__file__).resolve().parents[3]
OVERLEAF_APPENDIX = ROOT.parent / "overleaf" / "arxiv" / "appendix.tex"

SOURCE = "github/experiments/appendix_a_sec3_score_matrix/inventory_tables/gen_table.py"

BROAD_CATEGORY_ORDER = [
    "Agentic & tool use",
    "Math",
    "Coding",
    "Multimodal & vision",
    "Long context",
    "Instruction following",
    "Knowledge & QA",
    "Reasoning",
    "Hallucination & factuality",
    "Science",
    "Other",
]

BROAD_CATEGORY_MAP = {
    "Agentic": "Agentic & tool use",
    "Agentic search": "Agentic & tool use",
    "Search Agent": "Agentic & tool use",
    "Tool Use": "Agentic & tool use",
    "Tool use": "Agentic & tool use",
    "Math": "Math",
    "Math/Vision": "Math",
    "Coding": "Coding",
    "Repository Code": "Coding",
    "Multimodal": "Multimodal & vision",
    "Vision": "Multimodal & vision",
    "Video/Multimodal": "Multimodal & vision",
    "Long Context": "Long context",
    "Long-context": "Long context",
    "Instruction Following": "Instruction following",
    "Instruction following": "Instruction following",
    "Knowledge": "Knowledge & QA",
    "QA": "Knowledge & QA",
    "Reasoning": "Reasoning",
    "Reasoning & Knowledge": "Reasoning",
    "Hallucination": "Hallucination & factuality",
    "Factuality": "Hallucination & factuality",
    "Science": "Science",
}

PROVIDER_ORDER = [
    "OpenAI",
    "Google",
    "Anthropic",
    "Alibaba",
    "DeepSeek",
    "Meta",
    "Zhipu AI",
    "Moonshot AI",
    "xAI",
    "MiniMax",
    "Cohere",
    "ByteDance",
    "Mistral",
]


def latex_escape(value: object) -> str:
    if value is None:
        return "---"
    if isinstance(value, float) and math.isnan(value):
        return "---"
    text = str(value)
    if not text:
        return "---"
    placeholders = {
        "τ²": "@@TAU2@@",
        "τ³": "@@TAU3@@",
        "τ": "@@TAU@@",
    }
    for src, placeholder in placeholders.items():
        text = text.replace(src, placeholder)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    text = "".join(replacements.get(ch, ch) for ch in text)
    return (
        text
        .replace("@@TAU2@@", r"$\tau^2$")
        .replace("@@TAU3@@", r"$\tau^3$")
        .replace("@@TAU@@", r"$\tau$")
    )


def href(url: object) -> str:
    if url is None:
        return ""
    if isinstance(url, float) and math.isnan(url):
        return ""
    text = str(url).strip()
    if not text:
        return ""
    text = text.replace("%", r"\%").replace("#", r"\#")
    return rf"\href{{{text}}}{{\faExternalLink*}}"


def fmt_int(value: object) -> str:
    if value is None:
        return "---"
    if isinstance(value, float) and math.isnan(value):
        return "---"
    return f"{int(value):,}".replace(",", "{,}")


def fmt_params_millions(value: object) -> str:
    if value is None:
        return "---"
    if isinstance(value, float) and math.isnan(value):
        return "---"
    billions = float(value) / 1000.0
    if abs(billions - round(billions)) < 1e-9:
        return str(int(round(billions)))
    return f"{billions:.1f}".rstrip("0").rstrip(".")


def fmt_bool(value: object) -> str:
    return r"\cmark" if bool(value) else r"\xmark"


def fmt_release(value: object) -> str:
    if value is None:
        return "---"
    if isinstance(value, float) and math.isnan(value):
        return "---"
    text = str(value)
    return text[:7] if len(text) >= 7 else text


def broad_category(category: str) -> str:
    return BROAD_CATEGORY_MAP.get(category, "Other")


def category_sort_key(row: pd.Series) -> tuple[int, str]:
    broad = broad_category(str(row["category"]))
    try:
        rank = BROAD_CATEGORY_ORDER.index(broad)
    except ValueError:
        rank = len(BROAD_CATEGORY_ORDER)
    return rank, str(row["name"]).lower()


def provider_sort_key(provider: str) -> tuple[int, str]:
    try:
        return PROVIDER_ORDER.index(provider), provider
    except ValueError:
        return len(PROVIDER_ORDER), provider


def parse_existing_model_links() -> dict[str, str]:
    if not OVERLEAF_APPENDIX.exists():
        return {}
    text = OVERLEAF_APPENDIX.read_text()
    if r"\label{tab:models}" not in text:
        return {}
    block = text.split(r"\label{tab:models}", 1)[1].split(r"\end{table*}", 1)[0]
    links: dict[str, str] = {}
    for line in block.splitlines():
        if r"\href{" not in line or "&" not in line:
            continue
        parts = [p.strip() for p in line.split("&")]
        if len(parts) < 8:
            continue
        match = re.search(r"\\href\{(?P<url>[^}]+)\}\{\\faExternalLink\*\}", parts[-1])
        if not match:
            continue
        name = re.sub(r"\s+", " ", parts[1]).strip()
        links[name] = match.group("url").replace(r"\%", "%").replace(r"\#", "#")
    return links


def normalize_model_name(name: str) -> str:
    name = re.sub(r"\([^)]*\)", "", name)
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def render_benchmark_table(bench_df: pd.DataFrame) -> str:
    rows = bench_df.copy()
    rows["broad_category"] = rows["category"].map(lambda x: broad_category(str(x)))
    rows = rows.sort_values(by=["broad_category", "name"], key=lambda s: s.map(str.lower))
    rows = rows.sort_values(
        by=["broad_category", "name"],
        key=lambda s: s.map(lambda x: BROAD_CATEGORY_ORDER.index(x) if x in BROAD_CATEGORY_ORDER else str(x).lower())
        if s.name == "broad_category"
        else s.map(str.lower),
        kind="stable",
    )

    counts = Counter(rows["broad_category"])
    lines = [
        rf"% Source: {SOURCE}",
        r"% ── Benchmark table ──────────────────────────────────────────────",
        r"{\scriptsize",
        r"\setlength{\tabcolsep}{1.2pt}",
        r"\begin{longtable}{p{0.13\textwidth} p{0.40\textwidth} p{0.19\textwidth} r c}",
        rf"\caption{{\textbf{{Benchmark inventory.}} All {len(rows)} benchmarks in the adopted score matrix. Categories are grouped to match the main-text summary.}}",
        r"\label{tab:benchmarks} \\",
        r"\toprule",
        r"\textbf{Category} & \textbf{Benchmark} & \textbf{Metric} & \textbf{Items} & \textbf{Link} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"\textbf{Category} & \textbf{Benchmark} & \textbf{Metric} & \textbf{Items} & \textbf{Link} \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]

    first_group = True
    for category in BROAD_CATEGORY_ORDER:
        group = rows[rows["broad_category"] == category]
        if group.empty:
            continue
        if not first_group:
            lines.append(r"\cmidrule(lr){1-5}")
        first_group = False
        for i, (_, row) in enumerate(group.iterrows()):
            cat_cell = (
                rf"\multirow{{{counts[category]}}}{{*}}{{\parbox[t]{{0.13\textwidth}}{{{latex_escape(category)} ({counts[category]})}}}}"
                if i == 0
                else ""
            )
            lines.append(
                f"{cat_cell} & {latex_escape(row['name'])} & {latex_escape(row['metric'])} "
                f"& {fmt_int(row['num_problems'])} & {href(row['source_url'])} \\\\"
            )

    lines.extend([r"\end{longtable}", r"}% end \small"])
    return "\n".join(lines)


def split_provider_groups(models_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    counts = Counter(models_df["provider"])
    providers = sorted(counts, key=provider_sort_key)
    target = math.ceil(len(models_df) / 2)
    left: list[str] = []
    total = 0
    for provider in providers:
        if total < target:
            left.append(provider)
            total += counts[provider]
        else:
            break
    right = [p for p in providers if p not in left]
    return left, right


def render_provider_tabular(models_df: pd.DataFrame, providers: list[str], links_by_name: dict[str, str]) -> list[str]:
    lines = [
        r"\begin{tabular}{@{}p{0.16\linewidth}p{0.39\linewidth}p{0.07\linewidth}p{0.07\linewidth}ccp{0.11\linewidth}c@{}}",
        r"\toprule",
        r"\textbf{Provider} & \textbf{Model} & \textbf{B} & \textbf{Act.} & \textbf{R} & \textbf{O} & \textbf{Rel.} & \textbf{\faExternalLink*} \\",
        r"\midrule",
    ]
    first_group = True
    for provider in providers:
        group = models_df[models_df["provider"] == provider].sort_values(["release_date", "name"])
        if group.empty:
            continue
        if not first_group:
            lines.append(r"\cmidrule(lr){1-8}")
        first_group = False
        for i, (model_id, row) in enumerate(group.iterrows()):
            provider_cell = (
                rf"\multirow{{{len(group)}}}{{*}}{{\parbox[t]{{0.16\linewidth}}{{{latex_escape(provider)}}}}}"
                if i == 0 and len(group) > 1
                else (latex_escape(provider) if i == 0 else "")
            )
            name = str(row["name"])
            url = links_by_name.get(normalize_model_name(name), "")
            lines.append(
                f"{provider_cell} & {latex_escape(name)} & {fmt_params_millions(row['params_total_M'])} "
                f"& {fmt_params_millions(row['params_active_M'])} & {fmt_bool(row['is_reasoning'])} "
                f"& {fmt_bool(row['open_weights'])} & {fmt_release(row['release_date'])} & {href(url)} \\\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return lines


def render_model_table(models_df: pd.DataFrame) -> str:
    rows = models_df.copy()
    rows["provider"] = rows["provider"].astype(str)
    rows["name"] = rows["name"].astype(str)
    left, right = split_provider_groups(rows)
    existing_links = {
        normalize_model_name(name): url
        for name, url in parse_existing_model_links().items()
    }

    lines = [
        rf"% Source: {SOURCE}",
        r"% ── Model table (combined, side by side) ──────────────────────────",
        r"\begin{table*}[!htbp]",
        rf"\caption{{\textbf{{Model inventory.}} All {len(rows)} models from {rows['provider'].nunique()} providers. \emph{{R}} = reasoning (chain-of-thought). \emph{{O}} = open-weight. Parameter counts in billions; ``--- = undisclosed. Active parameters shown only for MoE models.}}",
        r"\label{tab:models}",
        r"\tiny",
        r"\setlength{\tabcolsep}{0.5pt}",
        r"\begin{minipage}[t]{0.495\textwidth}",
        r"\centering",
        *render_provider_tabular(rows, left, existing_links),
        r"\end{minipage}%",
        r"\hfill",
        r"\begin{minipage}[t]{0.495\textwidth}",
        r"\centering",
        *render_provider_tabular(rows, right, existing_links),
        r"\end{minipage}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def render_tables() -> str:
    _df, info, models_df, bench_df = load_score_matrix(return_info=True, return_metadata=True)
    assert info.n_models == len(models_df)
    assert info.n_benchmarks == len(bench_df)
    parts = [
        render_benchmark_table(bench_df),
        "",
        render_model_table(models_df),
    ]
    return "\n".join(parts)


def write_overleaf(tables: str) -> None:
    text = OVERLEAF_APPENDIX.read_text()
    start = text.index("% ── Benchmark table")
    source_line = f"% Source: {SOURCE}\n"
    while text[:start].endswith(source_line):
        start -= len(source_line)
    end_marker = r"\end{table*}"
    end = text.index(end_marker, start) + len(end_marker)
    updated = text[:start] + tables + text[end:]
    OVERLEAF_APPENDIX.write_text(updated)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-overleaf", action="store_true", help="replace Appendix A.1 inventory tables in overleaf/arxiv/appendix.tex")
    args = parser.parse_args()
    tables = render_tables()
    if args.write_overleaf:
        write_overleaf(tables)
        print(f"updated {OVERLEAF_APPENDIX}")
    else:
        print(tables)


if __name__ == "__main__":
    main()
