"""
Validation v3 — compare measured Tot_ict vs simulated fleet power
(+ editable non-GPU baseline) on a 1-day slice.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from validation.core import compute_metrics  # noqa: E402
from validation_v2.explorer_core import load_slice  # noqa: E402

_V3 = os.path.dirname(os.path.abspath(__file__))
_SLICE_CFG = os.path.join(_V3, "slice_config.json")
_CACHE_DIR = os.path.join(_V3, "slice_cache")
_CACHE_FILE = os.path.join(_CACHE_DIR, "day_slice.json")

MAPE_FLOOR_KW = 1.0
DEFAULT_BASELINE_KW = 255.0


def day_slice_config():
    with open(_SLICE_CFG) as f:
        return json.load(f)


def load_day_slice(force_rebuild=False):
    return load_slice(
        force_rebuild=force_rebuild,
        cfg=day_slice_config(),
        cache_path=_CACHE_FILE,
    )


def build_compare(baseline_kw: float | None = None, force_rebuild: bool = False) -> dict:
    slice_data = load_day_slice(force_rebuild=force_rebuild)
    measured = np.asarray(slice_data["measured"]["kw"], float)
    gpu = np.asarray(slice_data["modeled"]["fleet_kw"], float)
    base = DEFAULT_BASELINE_KW if baseline_kw is None else float(baseline_kw)
    calc = gpu + base

    t0 = slice_data["meta"]["window"]["t0"]
    t0_ms = int(pd.Timestamp(t0, tz="UTC").timestamp() * 1000)
    abs_epoch = [t0_ms + int(x) for x in slice_data["measured"]["epoch_ms"]]
    ts_utc = pd.to_datetime(abs_epoch, unit="ms", utc=True)

    metrics = compute_metrics(measured, calc, ts_utc, mape_floor_kw=MAPE_FLOOR_KW)
    by_stage = slice_data["modeled"].get("by_stage") or {}
    alloc = slice_data["modeled"].get("alloc_gpus") or []
    residual = [round(float(c - m), 3) for m, c in zip(measured, calc)]

    return {
        "labels": slice_data["measured"]["labels"],
        "epoch_ms": slice_data["measured"]["epoch_ms"],
        "measured": [round(float(x), 3) for x in measured],
        "model_gpu": [round(float(x), 3) for x in gpu],
        "calculated": [round(float(x), 3) for x in calc],
        "residual": residual,
        "by_stage": by_stage,
        "alloc_gpus": [int(x) for x in alloc],
        "metrics": metrics,
        "meta": {
            **slice_data["meta"],
            "baseline_kw": round(base, 1),
            "default_baseline_kw": DEFAULT_BASELINE_KW,
            "window_label": "1-day",
            "measured_source": "logics_pub Tot_ict",
            "model_source": "job_table → server.py (GPU fleet)",
        },
    }


def recompute_with_baseline(payload: dict, baseline_kw: float) -> dict:
    measured = np.asarray(payload["measured"], float)
    gpu = np.asarray(payload["model_gpu"], float)
    base = float(baseline_kw)
    calc = gpu + base

    t0 = payload["meta"]["window"]["t0"]
    t0_ms = int(pd.Timestamp(t0, tz="UTC").timestamp() * 1000)
    abs_epoch = [t0_ms + int(x) for x in payload["epoch_ms"]]
    ts_utc = pd.to_datetime(abs_epoch, unit="ms", utc=True)

    metrics = compute_metrics(measured, calc, ts_utc, mape_floor_kw=MAPE_FLOOR_KW)
    residual = [round(float(c - m), 3) for m, c in zip(measured, calc)]

    out = dict(payload)
    out["calculated"] = [round(float(x), 3) for x in calc]
    out["residual"] = residual
    out["metrics"] = metrics
    out["meta"] = {**payload["meta"], "baseline_kw": round(base, 1)}
    return out
