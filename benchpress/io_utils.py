#!/usr/bin/env python3
"""Shared file I/O helpers for BenchPress scripts."""

import gzip
import json
import os
import tempfile

import numpy as np


def open_text_auto(path, mode="rt", encoding="utf-8"):
    """Open plain text or gzip-compressed text based on the file suffix."""
    opener = gzip.open if str(path).endswith(".gz") else open
    return opener(path, mode, encoding=encoding)


def load_json(path):
    """Load JSON from a plain `.json` or gzip-compressed `.json.gz` file."""
    with open_text_auto(path, "rt") as f:
        return json.load(f)


def write_json(
    path,
    payload,
    indent=None,
    sort_keys=False,
    trailing_newline=False,
    default=None,
):
    """Write JSON to a plain `.json` or gzip-compressed `.json.gz` file."""
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open_text_auto(path, "wt") as f:
        json.dump(payload, f, indent=indent, sort_keys=sort_keys, default=default)
        if trailing_newline:
            f.write("\n")


def write_json_next_to(source_file, payload, filename="results.json", **kwargs):
    """Write JSON next to `source_file` and return the output path."""
    path = os.path.join(os.path.dirname(os.path.abspath(source_file)), filename)
    write_json(path, payload, **kwargs)
    return path


def write_json_atomic(
    path,
    payload,
    indent=None,
    sort_keys=False,
    trailing_newline=False,
    default=None,
):
    """Atomically write JSON to a plain `.json` or gzip-compressed `.json.gz` file."""
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    tmp_path = f"{path}.tmp.gz" if str(path).endswith(".gz") else f"{path}.tmp"
    try:
        write_json(
            tmp_path,
            payload,
            indent=indent,
            sort_keys=sort_keys,
            trailing_newline=trailing_newline,
            default=default,
        )
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def load_jsonl(path, missing_ok=False):
    """Load newline-delimited JSON records, skipping blank lines."""
    if missing_ok and not os.path.exists(path):
        return []
    records = []
    with open_text_auto(path, "rt") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_jsonl_keyed(path, key_fn, missing_ok=False):
    """Load JSONL records into a dict keyed by `key_fn(record)`."""
    return {key_fn(record): record for record in load_jsonl(path, missing_ok=missing_ok)}


def append_jsonl(path, record, sort_keys=True):
    """Append one JSON record to a newline-delimited JSON file."""
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open_text_auto(path, "at") as f:
        f.write(json.dumps(record, sort_keys=sort_keys) + "\n")
        f.flush()


def write_npz_compressed_atomic(path, **arrays):
    """Atomically write a compressed `.npz` archive."""
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".tmp_", suffix=".npz", dir=dir_name or None)
    os.close(fd)
    try:
        np.savez_compressed(tmp_path, **arrays)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def safe_token(value):
    """Filesystem-safe token for cache names derived from user-visible strings."""
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(value))
