"""Custom score-matrix loading, validation, and exchange formats."""

from __future__ import annotations

import csv
import json
import os

import numpy as np

from benchpress.io_utils import open_text_auto


class _ScoreMatrixArray(np.ndarray):
    """Numpy matrix view carrying metric metadata for predictor calls."""

    def __new__(cls, values, model_ids, benchmark_ids, metric):
        obj = np.asarray(values, dtype=float).view(cls)
        obj.model_ids = list(model_ids)
        obj.benchmark_ids = list(benchmark_ids)
        obj.metric = metric
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.model_ids = getattr(obj, 'model_ids', None)
        self.benchmark_ids = getattr(obj, 'benchmark_ids', None)
        self.metric = getattr(obj, 'metric', None)


def _default_metric():
    return {'type': 'pct', 'range': [0.0, 100.0], 'higher_is_better': True}


def _as_bool(value, field_name):
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {'true', '1', 'yes'}:
            return True
        if lowered in {'false', '0', 'no'}:
            return False
    raise ValueError(f"metric field '{field_name}' must be a boolean")


def _as_float(value, field_name):
    if isinstance(value, str):
        value = value.strip()
        if value.endswith('%'):
            value = value[:-1]
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc


def _normalize_range(score_range, benchmark_id):
    if score_range is None:
        return None
    if not isinstance(score_range, (list, tuple)) or len(score_range) != 2:
        raise ValueError(
            f"metric for benchmark '{benchmark_id}' must declare range as [min, max]"
        )
    lo = _as_float(score_range[0], f"range[0] for benchmark '{benchmark_id}'")
    hi = _as_float(score_range[1], f"range[1] for benchmark '{benchmark_id}'")
    if not np.isfinite(lo) or not np.isfinite(hi) or lo > hi:
        raise ValueError(
            f"metric for benchmark '{benchmark_id}' has invalid range [{lo}, {hi}]"
        )
    return [lo, hi]


def _metric_type_from_name(name):
    lowered = str(name).strip().lower()
    if lowered in {'', 'none', 'null'}:
        return None
    if lowered in {'pct', 'percent', 'percentage'}:
        return 'pct'
    if lowered in {'accuracy', 'acc', 'exact_match', 'em'}:
        return 'pct'
    if 'pass' in lowered and '@' in lowered:
        return 'pct'
    if 'elo' in lowered:
        return 'elo'
    if 'rating' in lowered:
        return 'rating'
    if 'dollar' in lowered or lowered in {'$', 'usd'}:
        return 'dollars'
    return lowered.replace(' ', '_')


def _normalize_metric(spec, benchmark_id):
    if spec is None:
        return _default_metric()
    if not isinstance(spec, dict):
        raise ValueError(f"metric for benchmark '{benchmark_id}' must be an object")

    metric_type = spec.get('type', spec.get('metric_type'))
    metric_type = _metric_type_from_name(metric_type) if metric_type is not None else 'pct'
    score_range = spec.get('range')
    if score_range is None and metric_type == 'pct':
        score_range = [0, 100]
    return {
        'type': metric_type,
        'range': _normalize_range(score_range, benchmark_id),
        'higher_is_better': _as_bool(
            spec.get('higher_is_better', True),
            f"higher_is_better for benchmark '{benchmark_id}'",
        ),
    }


def _ensure_unique(ids, label):
    seen = set()
    for value in ids:
        if value in seen:
            raise ValueError(f"duplicate {label} id: '{value}'")
        seen.add(value)


def _clean_id(value, label, row_label):
    text = str(value).strip()
    if not text:
        raise ValueError(f"empty {label} id in {row_label}")
    return text


def _nested_value(record, path):
    value = record
    for part in path.split('.'):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _first_present(record, paths):
    for path in paths:
        value = _nested_value(record, path)
        if value is not None and value != '':
            return value
    return None


def _coerce_id(value):
    if isinstance(value, dict):
        value = _first_present(
            value,
            ['id', 'model_id', 'benchmark_id', 'name', 'slug', 'display_name'],
        )
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _extract_score(record):
    value = _first_present(
        record,
        ['score', 'value', 'metric_value', 'normalized_score', 'accuracy', 'acc'],
    )
    if value is None and isinstance(record.get('metrics'), dict):
        for candidate in ['score', 'accuracy', 'acc', 'exact_match']:
            if candidate in record['metrics']:
                value = record['metrics'][candidate]
                break
        if value is None:
            for candidate in record['metrics'].values():
                try:
                    return _as_float(candidate, 'score')
                except ValueError:
                    continue
    if value is None:
        return None
    return _as_float(value, 'score')


def _extract_metric(record):
    metric = {}
    raw_metric = record.get('metric')
    if isinstance(raw_metric, dict):
        metric.update(raw_metric)
    elif raw_metric is not None:
        metric['type'] = _metric_type_from_name(raw_metric)

    metric_type = _first_present(
        record,
        ['metric_type', 'score_type', 'type', 'metric.type', 'metric.metric_type'],
    )
    if metric_type is not None:
        metric['type'] = _metric_type_from_name(metric_type)

    score_range = _first_present(record, ['range', 'score_range', 'metric.range'])
    if score_range is None:
        lo = _first_present(record, ['min', 'score_min', 'metric.min'])
        hi = _first_present(record, ['max', 'score_max', 'metric.max'])
        if lo is not None or hi is not None:
            score_range = [lo, hi]
    if score_range is not None:
        metric['range'] = score_range

    higher_is_better = _first_present(
        record,
        ['higher_is_better', 'metric.higher_is_better'],
    )
    if higher_is_better is not None:
        metric['higher_is_better'] = higher_is_better
    return metric or None


def _load_records(path):
    lower = str(path).lower()
    if lower.endswith('.jsonl') or lower.endswith('.jsonl.gz'):
        records = []
        with open_text_auto(path, 'rt') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records
    if lower.endswith('.csv'):
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            return list(reader)
    with open_text_auto(path, 'rt') as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ['records', 'results', 'data']:
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"could not find EEE records in '{path}'")


class ScoreMatrix:
    """A model-by-benchmark score matrix with per-column metric metadata."""

    def __init__(self, model_ids, benchmark_ids, values, metric=None):
        self.model_ids = [_clean_id(value, 'model', 'model_ids') for value in model_ids]
        self.benchmark_ids = [
            _clean_id(value, 'benchmark', 'benchmark_ids') for value in benchmark_ids
        ]
        _ensure_unique(self.model_ids, 'model')
        _ensure_unique(self.benchmark_ids, 'benchmark')

        values = np.asarray(values, dtype=float)
        expected_shape = (len(self.model_ids), len(self.benchmark_ids))
        if values.shape != expected_shape:
            raise ValueError(
                f"values shape {values.shape} does not match "
                f"{len(self.model_ids)} models x {len(self.benchmark_ids)} benchmarks"
            )

        metric = metric or {}
        unknown = [key for key in metric if key not in self.benchmark_ids]
        if unknown:
            raise ValueError(
                "metadata includes unknown benchmark id(s): " + ", ".join(sorted(unknown))
            )
        self.metric = {
            benchmark_id: _normalize_metric(metric.get(benchmark_id), benchmark_id)
            for benchmark_id in self.benchmark_ids
        }
        self._values = _ScoreMatrixArray(
            values, self.model_ids, self.benchmark_ids, self.metric)

    @property
    def values(self):
        return self._values

    @classmethod
    def from_file(cls, path):
        lower = str(path).lower()
        if lower.endswith('.csv'):
            with open(path, newline='') as f:
                first_row = next(csv.reader(f), None)
            if first_row and first_row[0].strip().lstrip('\ufeff') == 'model':
                return cls.from_csv(path)
        return cls.from_eee(path)

    @classmethod
    def from_csv(cls, path):
        with open(path, newline='') as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise ValueError("CSV is empty; first header cell must be 'model'") from exc
            if not header or header[0].strip().lstrip('\ufeff') != 'model':
                raise ValueError("first CSV header cell must be 'model'")
            benchmark_ids = [
                _clean_id(value, 'benchmark', 'CSV header') for value in header[1:]
            ]
            _ensure_unique(benchmark_ids, 'benchmark')

            model_ids = []
            rows = []
            for line_number, row in enumerate(reader, start=2):
                if not any(cell.strip() for cell in row):
                    continue
                if len(row) > len(header):
                    raise ValueError(
                        f"CSV row {line_number} has {len(row)} cells; expected {len(header)}"
                    )
                row = row + [''] * (len(header) - len(row))
                model_id = _clean_id(row[0], 'model', f'CSV row {line_number}')
                model_ids.append(model_id)
                values = []
                for benchmark_id, cell in zip(benchmark_ids, row[1:]):
                    text = cell.strip()
                    if not text:
                        values.append(np.nan)
                    else:
                        values.append(_as_float(text, f"score for '{model_id}/{benchmark_id}'"))
                rows.append(values)
            _ensure_unique(model_ids, 'model')

        meta_path = os.path.splitext(path)[0] + '.meta.json'
        metric = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                metric = json.load(f)
            if not isinstance(metric, dict):
                raise ValueError(f"metadata file '{meta_path}' must contain a JSON object")
            for benchmark_id, spec in metric.items():
                if not isinstance(spec, dict):
                    raise ValueError(
                        f"metadata for benchmark '{benchmark_id}' must be an object"
                    )
                if 'type' not in spec and 'metric_type' not in spec:
                    raise ValueError(
                        f"metadata for benchmark '{benchmark_id}' must declare 'type'"
                    )
                if 'range' not in spec:
                    raise ValueError(
                        f"metadata for benchmark '{benchmark_id}' must declare 'range'"
                    )

        return cls(model_ids, benchmark_ids, np.asarray(rows, dtype=float), metric=metric)

    @classmethod
    def from_eee(cls, path):
        records = _load_records(path)
        model_ids = []
        benchmark_ids = []
        cells = {}
        metric = {}

        for index, record in enumerate(records, start=1):
            if not isinstance(record, dict):
                raise ValueError(f"EEE record {index} must be a JSON object")
            model_id = _coerce_id(_first_present(record, ['model_id', 'model', 'model.id']))
            benchmark_id = _coerce_id(
                _first_present(
                    record,
                    ['benchmark_id', 'benchmark', 'benchmark.id', 'eval_id', 'eval.id', 'task'],
                )
            )
            score = _extract_score(record)
            if model_id is None or benchmark_id is None or score is None or not np.isfinite(score):
                continue
            if model_id not in model_ids:
                model_ids.append(model_id)
            if benchmark_id not in benchmark_ids:
                benchmark_ids.append(benchmark_id)
            key = (model_id, benchmark_id)
            if key in cells and not np.isclose(cells[key], score, equal_nan=True):
                raise ValueError(
                    f"duplicate EEE score for model '{model_id}' and benchmark "
                    f"'{benchmark_id}'"
                )
            cells[key] = score
            record_metric = _extract_metric(record)
            if benchmark_id not in metric or metric[benchmark_id] is None:
                metric[benchmark_id] = record_metric

        if not cells:
            raise ValueError(
                f"EEE file '{path}' did not contain records with model_id, "
                "benchmark_id, and numeric score"
            )

        values = np.full((len(model_ids), len(benchmark_ids)), np.nan, dtype=float)
        model_index = {model_id: i for i, model_id in enumerate(model_ids)}
        benchmark_index = {benchmark_id: j for j, benchmark_id in enumerate(benchmark_ids)}
        for (model_id, benchmark_id), score in cells.items():
            values[model_index[model_id], benchmark_index[benchmark_id]] = score
        return cls(model_ids, benchmark_ids, values, metric=metric)

    def validate(self, rank=2):
        if self.values.shape != (len(self.model_ids), len(self.benchmark_ids)):
            raise ValueError("values shape does not match model_ids and benchmark_ids")

        observed = np.isfinite(self.values)
        if np.isinf(self.values).any():
            raise ValueError("score matrix contains infinite values")

        for benchmark_id in self.benchmark_ids:
            spec = self.metric.get(benchmark_id)
            if not spec or not spec.get('type'):
                raise ValueError(f"benchmark '{benchmark_id}' has no resolved metric type")
            score_range = spec.get('range')
            if score_range is None:
                raise ValueError(f"benchmark '{benchmark_id}' has no declared score range")
            lo, hi = _normalize_range(score_range, benchmark_id)
            j = self.benchmark_ids.index(benchmark_id)
            col = self.values[:, j]
            bad = np.isfinite(col) & ((col < lo) | (col > hi))
            if bad.any():
                bad_models = [self.model_ids[i] for i in np.where(bad)[0]]
                raise ValueError(
                    f"benchmark '{benchmark_id}' has score(s) outside declared "
                    f"range [{lo}, {hi}] for model(s): " + ", ".join(bad_models)
                )

        if len(self.model_ids) < rank + 1:
            raise ValueError(
                f"rank-{rank} + bias prediction needs at least {rank + 1} models; "
                f"got {len(self.model_ids)}"
            )
        if len(self.benchmark_ids) < rank + 1:
            raise ValueError(
                f"rank-{rank} + bias prediction needs at least {rank + 1} benchmarks; "
                f"got {len(self.benchmark_ids)}"
            )
        if observed.sum() < rank + 1:
            raise ValueError(
                f"rank-{rank} + bias prediction needs at least {rank + 1} observed "
                f"cells; got {int(observed.sum())}"
            )
        empty_rows = [self.model_ids[i] for i in np.where(observed.sum(axis=1) == 0)[0]]
        if empty_rows:
            raise ValueError("every model needs at least one observed score; empty: "
                             + ", ".join(empty_rows))
        empty_cols = [
            self.benchmark_ids[j] for j in np.where(observed.sum(axis=0) == 0)[0]
        ]
        if empty_cols:
            raise ValueError("every benchmark needs at least one observed score; empty: "
                             + ", ".join(empty_cols))
        return self


__all__ = ['ScoreMatrix']