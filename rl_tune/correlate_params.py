"""
Rank config.json knobs by how strongly they line up with Ganglia GPU power.

Uses the cached validation-v2 day slice (measured source = ganglia_pub) when
raw ExaData is not present. Each physics parameter θ multiplies a time series
X_θ(t) in the mean-field fleet formula:

    P(t) = (P_max/1000) * [ n_train(t)*u_tr + n_ft(t)*u_ft
                            + n_inf(t)*u_inf + n_idle(t)*rho ]

corr(X_θ, P_ganglia) and the partial correlation (controlling the other stage
counts) say which knobs can track the measured shape. A 1-D MAPE sweep says
which knobs move error magnitude.
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Any

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

STAGES = ("training", "fine_tuning", "inference")
DEFAULT_SLICE = os.path.join(_ROOT, "validation_v2", "slice_cache", "day_slice.json")
DEFAULT_OUT = os.path.join(_ROOT, "rl_tune", "results")
CONFIG_PATH = os.path.join(_ROOT, "config.json")

# Mean-field util used when the inner loop does not resimulate stochastic_util.
DEFAULT_U = {
    "training.u_plateau": 0.60,
    "fine_tuning.u_plateau": 0.68,
    "inference.u_eff": 0.32,
    "rho": 0.117,
}
SWEEPS = {
    "training.u_plateau": np.linspace(0.45, 0.90, 19),
    "fine_tuning.u_plateau": np.linspace(0.45, 0.90, 19),
    "inference.u_eff": np.linspace(0.10, 0.65, 19),
    "rho": np.linspace(0.05, 0.25, 17),
}
# Shape correlation uses requested GPU counts (what tracks Ganglia).
# MAPE sweeps use inventory-capped n_eff (what the formula can actually emit).
PARAM_REGRESSOR = {
    "training.u_plateau": "n_req_training",
    "fine_tuning.u_plateau": "n_req_fine_tuning",
    "inference.u_eff": "n_req_inference",
    "rho": "n_eff_idle",
}
PARAM_SWEEP_REGRESSOR = {
    "training.u_plateau": "n_eff_training",
    "fine_tuning.u_plateau": "n_eff_fine_tuning",
    "inference.u_eff": "n_eff_inference",
    "rho": "n_eff_idle",
}


def pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.size < 3 or b.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.size < 3 or b.size < 3:
        return None
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return pearson(ra.astype(float), rb.astype(float))


def _lstsq(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def partial_r(y: np.ndarray, x: np.ndarray, Z: np.ndarray) -> float | None:
    """Pearson r of residuals after linearly removing columns of Z (with intercept)."""
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    Z = np.asarray(Z, float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    Z1 = np.column_stack([np.ones(len(y)), Z])
    ry = y - Z1 @ _lstsq(Z1, y)
    rx = x - Z1 @ _lstsq(Z1, x)
    return pearson(ry, rx)


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def load_slice(path: str = DEFAULT_SLICE) -> dict:
    with open(path) as f:
        return json.load(f)


def features_from_slice(data: dict, gpus_per_node: int = 4) -> dict[str, np.ndarray]:
    measured = np.asarray(data["measured"]["kw"], float)
    modeled = data["modeled"]
    meta = data.get("meta") or {}
    inventory = meta.get("inventory") or {}
    hw = meta.get("hw_type", "V100")
    n_k = float(inventory.get(hw, 3780))
    nodes = {
        s: np.asarray(modeled["active_nodes"][s], float) for s in STAGES
    }
    n_req = {s: nodes[s] * gpus_per_node for s in STAGES}
    total_req = sum(n_req.values())
    scale = np.minimum(1.0, n_k / np.maximum(total_req, 1.0))
    n_eff = {s: n_req[s] * scale for s in STAGES}
    n_eff_idle = np.clip(n_k - sum(n_eff.values()), 0.0, None)
    alloc = np.asarray(modeled.get("alloc_gpus") or total_req, float)
    frac = {s: n_req[s] / np.maximum(total_req, 1.0) for s in STAGES}
    stage_kw = {
        s: np.asarray((modeled.get("by_stage") or {}).get(s) or np.zeros_like(measured), float)
        for s in STAGES
    }
    out: dict[str, np.ndarray] = {
        "ganglia_kw": measured,
        "modeled_fleet_kw": np.asarray(modeled["fleet_kw"], float),
        "alloc_gpus": alloc,
        "total_req_gpus": total_req,
        "n_k": np.full_like(measured, n_k),
        "oversubscribed": (total_req > n_k).astype(float),
        "n_eff_idle": n_eff_idle,
        "eta": np.minimum(1.0, total_req / n_k),
    }
    for s in STAGES:
        out[f"n_req_{s}"] = n_req[s]
        out[f"n_eff_{s}"] = n_eff[s]
        out[f"frac_{s}"] = frac[s]
        out[f"model_kw_{s}"] = stage_kw[s]
        out[f"n_jobs_{s}"] = np.asarray(modeled["active_jobs"][s], float)
    return out


def mean_field_power(
    feats: dict[str, np.ndarray],
    u_train: float,
    u_ft: float,
    u_inf: float,
    rho: float,
    p_max_w: float = 300.0,
) -> np.ndarray:
    p_kw = p_max_w / 1000.0
    return p_kw * (
        feats["n_eff_training"] * u_train
        + feats["n_eff_fine_tuning"] * u_ft
        + feats["n_eff_inference"] * u_inf
        + feats["n_eff_idle"] * rho
    )


def mape(pred: np.ndarray, meas: np.ndarray, floor: float = 1.0) -> float:
    meas = np.asarray(meas, float)
    pred = np.asarray(pred, float)
    mask = np.isfinite(meas) & (meas > floor)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(pred[mask] - meas[mask]) / meas[mask]) * 100.0)


def _round(x: float | None, nd: int = 4) -> float | None:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return round(float(x), nd)


def analyze(data: dict | None = None, p_max_w: float = 300.0) -> dict[str, Any]:
    if data is None:
        data = load_slice()
    cfg = load_config()
    hw = (data.get("meta") or {}).get("hw_type", "V100")
    rho_cfg = cfg.get("hardware", {}).get(hw, {}).get("rho")
    if rho_cfg is None:
        rho_cfg = DEFAULT_U["rho"]
    phy = cfg.get("stage_physics") or {}
    defaults = {
        "training.u_plateau": float(phy.get("training", {}).get("u_plateau", DEFAULT_U["training.u_plateau"])),
        "fine_tuning.u_plateau": float(phy.get("fine_tuning", {}).get("u_plateau", DEFAULT_U["fine_tuning.u_plateau"])),
        "inference.u_eff": float(cfg.get("stage_defaults", {}).get("inference", {}).get("u_default", DEFAULT_U["inference.u_eff"])),
        "rho": float(rho_cfg),
    }

    feats = features_from_slice(data)
    g = feats["ganglia_kw"]
    meta = data.get("meta") or {}
    measured_src = (data.get("measured") or {}).get("source") or ""

    feature_rows = []
    feature_names = [
        ("modeled_fleet_kw", "Current modeled fleet (not a knob)"),
        ("alloc_gpus", "Allocated / requested GPUs (schedule, not a knob)"),
        ("n_req_training", "Requested training GPUs (training.u_plateau)"),
        ("n_req_fine_tuning", "Requested fine-tuning GPUs (fine_tuning.u_plateau)"),
        ("n_req_inference", "Requested inference GPUs (inference.u_eff)"),
        ("n_eff_training", "Capped training GPUs"),
        ("n_eff_fine_tuning", "Capped fine-tuning GPUs"),
        ("n_eff_inference", "Capped inference GPUs"),
        ("n_eff_idle", "V100.rho regressor (idle GPUs)"),
        ("frac_training", "Share of requested GPUs in training"),
        ("frac_fine_tuning", "Share of requested GPUs in fine-tuning"),
        ("frac_inference", "Share of requested GPUs in inference"),
        ("n_jobs_training", "Active training jobs"),
        ("n_jobs_fine_tuning", "Active fine-tuning jobs"),
        ("n_jobs_inference", "Active inference jobs"),
    ]
    for key, label in feature_names:
        series = feats[key]
        feature_rows.append({
            "feature": key,
            "label": label,
            "pearson_r": _round(pearson(series, g)),
            "spearman_r": _round(spearman(series, g)),
            "mean": _round(float(np.mean(series)), 3),
            "std": _round(float(np.std(series)), 3),
        })

    controls = {
        "n_req_training": np.column_stack([feats["n_req_fine_tuning"], feats["n_req_inference"]]),
        "n_req_fine_tuning": np.column_stack([feats["n_req_training"], feats["n_req_inference"]]),
        "n_req_inference": np.column_stack([feats["n_req_training"], feats["n_req_fine_tuning"]]),
        "n_eff_training": np.column_stack([feats["n_eff_fine_tuning"], feats["n_eff_inference"]]),
        "n_eff_fine_tuning": np.column_stack([feats["n_eff_training"], feats["n_eff_inference"]]),
        "n_eff_inference": np.column_stack([feats["n_eff_training"], feats["n_eff_fine_tuning"]]),
        "n_eff_idle": np.column_stack([
            feats["n_req_training"], feats["n_req_fine_tuning"], feats["n_req_inference"]
        ]),
        "alloc_gpus": np.column_stack([
            feats["n_req_training"], feats["n_req_fine_tuning"], feats["n_req_inference"]
        ]),
    }
    partial_rows = []
    for key, Z in controls.items():
        partial_rows.append({
            "feature": key,
            "partial_r_vs_ganglia": _round(partial_r(g, feats[key], Z)),
        })
    partial_map = {r["feature"]: r["partial_r_vs_ganglia"] for r in partial_rows}

    p_kw = p_max_w / 1000.0
    base = mean_field_power(
        feats,
        defaults["training.u_plateau"],
        defaults["fine_tuning.u_plateau"],
        defaults["inference.u_eff"],
        defaults["rho"],
        p_max_w=p_max_w,
    )
    resid = g - base

    param_rows = []
    for name, feat_key in PARAM_REGRESSOR.items():
        sweep_key = PARAM_SWEEP_REGRESSOR[name]
        dP = p_kw * feats[sweep_key]
        param_rows.append({
            "parameter": name,
            "regressor": feat_key,
            "pearson_r_vs_ganglia": _round(pearson(feats[feat_key], g)),
            "spearman_r_vs_ganglia": _round(spearman(feats[feat_key], g)),
            "partial_r_vs_ganglia": partial_map.get(feat_key),
            "pearson_r_capped": _round(pearson(feats[sweep_key], g)),
            "pearson_r_dP_vs_residual": _round(pearson(dP, resid)),
            "regressor_std": _round(float(np.std(feats[feat_key])), 3),
        })

    def pred_from(kwargs: dict[str, float]) -> np.ndarray:
        return mean_field_power(
            feats,
            kwargs["training.u_plateau"],
            kwargs["fine_tuning.u_plateau"],
            kwargs["inference.u_eff"],
            kwargs["rho"],
            p_max_w=p_max_w,
        )

    sweep_rows = []
    for name, xs in SWEEPS.items():
        points = []
        for x in xs:
            kw = dict(defaults)
            kw[name] = float(x)
            y = pred_from(kw)
            points.append({
                "value": _round(float(x), 4),
                "pearson_r": _round(pearson(y, g)),
                "mape_pct": _round(mape(y, g), 3),
                "bias_kw": _round(float(np.mean(y - g)), 3),
            })
        mapes = [p["mape_pct"] for p in points if p["mape_pct"] is not None]
        corrs = [p["pearson_r"] for p in points if p["pearson_r"] is not None]
        best = min(points, key=lambda p: p["mape_pct"] if p["mape_pct"] is not None else 1e9)
        sweep_rows.append({
            "parameter": name,
            "mape_range_pct": [_round(min(mapes), 3), _round(max(mapes), 3)] if mapes else None,
            "mape_span_pct": _round(max(mapes) - min(mapes), 3) if mapes else None,
            "r_range": [_round(min(corrs), 4), _round(max(corrs), 4)] if corrs else None,
            "best_on_this_slice": best,
            "curve": points,
        })
    sweep_map = {r["parameter"]: r for r in sweep_rows}

    ranked = []
    for row in param_rows:
        name = row["parameter"]
        pr = abs(row["partial_r_vs_ganglia"] or 0.0)
        span = float(sweep_map[name]["mape_span_pct"] or 0.0)
        ranked.append({
            **row,
            "mape_span_pct": sweep_map[name]["mape_span_pct"],
            "shape_score": _round(pr),
            "level_score": _round(span, 3),
            "rl_priority": _round(pr + min(span, 20.0) / 20.0),
        })
    ranked.sort(
        key=lambda r: abs(r["partial_r_vs_ganglia"] or 0.0),
        reverse=True,
    )

    X = np.column_stack([
        feats["n_eff_training"],
        feats["n_eff_fine_tuning"],
        feats["n_eff_inference"],
        feats["n_eff_idle"],
    ])
    beta = _lstsq(X, g)
    ols_pred = X @ beta
    ss_res = float(np.sum((g - ols_pred) ** 2))
    ss_tot = float(np.sum((g - g.mean()) ** 2))
    ols = {
        "r2": _round(1.0 - ss_res / ss_tot if ss_tot else None),
        "pearson_r": _round(pearson(ols_pred, g)),
        "kw_per_gpu": {
            "training": _round(float(beta[0]), 5),
            "fine_tuning": _round(float(beta[1]), 5),
            "inference": _round(float(beta[2]), 5),
            "idle": _round(float(beta[3]), 5),
        },
        "implied_util": {
            "training.u_plateau": _round(float(beta[0]) / p_kw),
            "fine_tuning.u_plateau": _round(float(beta[1]) / p_kw),
            "inference.u_eff": _round(float(beta[2]) / p_kw),
            "rho": _round(float(beta[3]) / p_kw),
        },
    }

    n_over = float(np.mean(feats["oversubscribed"]))
    return {
        "window": meta.get("window"),
        "year_month": meta.get("year_month"),
        "measured_source": measured_src,
        "n_points": int(len(g)),
        "ganglia_nodes": meta.get("ganglia_nodes"),
        "inventory": meta.get("inventory"),
        "ganglia_mean_kw": _round(float(np.mean(g)), 3),
        "ganglia_std_kw": _round(float(np.std(g)), 3),
        "modeled_mean_kw": _round(float(np.mean(feats["modeled_fleet_kw"])), 3),
        "r_current_model_vs_ganglia": _round(pearson(feats["modeled_fleet_kw"], g)),
        "oversubscribed_fraction": _round(n_over),
        "mean_requested_gpus": _round(float(np.mean(feats["total_req_gpus"])), 1),
        "defaults_used": {k: _round(v) for k, v in defaults.items()},
        "mean_field_default": {
            "pearson_r": _round(pearson(base, g)),
            "mape_pct": _round(mape(base, g), 3),
            "bias_kw": _round(float(np.mean(base - g)), 3),
        },
        "features_vs_ganglia": feature_rows,
        "partial_correlations": partial_rows,
        "parameters": ranked,
        "sweeps": [{k: v for k, v in r.items() if k != "curve"} for r in sweep_rows],
        "sweep_curves": {r["parameter"]: r["curve"] for r in sweep_rows},
        "ols_mean_field": ols,
        "rl_search_recommendation": _recommendation(ranked, n_over),
        "_feats": feats,
        "_sweep_curves": sweep_rows,
        "_g": g,
        "_base": base,
    }


def _recommendation(ranked: list[dict], oversubscribed_fraction: float) -> dict:
    high = []
    medium = []
    confounders = []
    low = []
    for r in ranked:
        name = r["parameter"]
        pr = abs(r["partial_r_vs_ganglia"] or 0.0)
        pear = abs(r["pearson_r_vs_ganglia"] or 0.0)
        span = float(r.get("mape_span_pct") or 0.0)
        if pr >= 0.4 or (pear >= 0.5 and pr >= 0.25) or span >= 20:
            high.append(name)
        elif pr >= 0.15:
            medium.append(name)
        elif pear >= 0.4 and pr < 0.15:
            confounders.append(name)
        else:
            low.append(name)
    notes = []
    if oversubscribed_fraction >= 0.8:
        notes.append(
            "Fleet is oversubscribed on this slice, so idle GPUs are rare and "
            "V100.rho barely changes MAPE. Re-check rho on a window with idle capacity."
        )
    notes.append(
        "training.u_plateau has a strong raw anti-correlation with Ganglia that "
        "collapses after controlling for fine-tuning/inference counts — it is a "
        "confounder of stage mix, not an independent driver. Stage-classification "
        "thresholds that move jobs between training and fine-tuning are high leverage."
    )
    return {
        "search_first": high,
        "search_next": medium,
        "confounders": confounders,
        "defer": low,
        "also_consider": [
            "replay.stage_thresholds.training_min_duration_h",
            "replay.stage_thresholds.training_min_nodes",
            "replay.stage_thresholds.finetuning_min_duration_h",
            "replay.stage_thresholds.inference_max_nodes",
        ],
        "notes": notes,
    }
    notes = []
    if oversubscribed_fraction >= 0.8:
        notes.append(
            "Fleet is oversubscribed on this slice, so idle GPUs are rare and "
            "V100.rho barely changes MAPE. Re-check rho on a window with idle capacity."
        )
    notes.append(
        "training.u_plateau has a strong raw anti-correlation with Ganglia that "
        "collapses after controlling for fine-tuning/inference counts — it is a "
        "confounder of stage mix, not an independent driver. Stage-classification "
        "thresholds that move jobs between training and fine-tuning are high leverage."
    )
    return {
        "search_first": high,
        "search_next": medium,
        "defer": low,
        "also_consider": [
            "replay.stage_thresholds.training_min_duration_h",
            "replay.stage_thresholds.training_min_nodes",
            "replay.stage_thresholds.finetuning_min_duration_h",
            "replay.stage_thresholds.inference_max_nodes",
        ],
        "notes": notes,
    }


def _jsonable(report: dict) -> dict:
    skip = {"_feats", "_sweep_curves", "_g", "_base"}
    return {k: v for k, v in report.items() if k not in skip}


def plot_report(report: dict, out_dir: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    feats: dict[str, np.ndarray] = report["_feats"]
    g = report["_g"]
    paths: list[str] = []

    # 1. Correlation bars
    labels = [
        "fine_tuning.u_plateau",
        "inference.u_eff",
        "training.u_plateau",
        "V100.rho",
    ]
    key = [
        "n_req_fine_tuning",
        "n_req_inference",
        "n_req_training",
        "n_eff_idle",
    ]
    partial_map = {r["feature"]: r["partial_r_vs_ganglia"] or 0.0 for r in report["partial_correlations"]}
    pearson_vals = [pearson(feats[k], g) or 0.0 for k in key]
    partial_vals = [partial_map.get(k, 0.0) for k in key]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    y = np.arange(len(labels))
    h = 0.36
    ax.barh(y + h / 2, pearson_vals, h, label="Pearson r of regressor vs Ganglia", color="#3b6ea5")
    ax.barh(y - h / 2, partial_vals, h, label="Partial r (other stages held fixed)", color="#e07a3d")
    ax.axvline(0, color="#333", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Correlation with Ganglia fleet GPU power")
    ax.set_xlim(-1.05, 1.05)
    ax.set_title("Which physics knobs line up with Ganglia (2022-03-20)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    fig.tight_layout()
    p = os.path.join(out_dir, "corr_bars.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # 2. Heatmap
    names = ["Ganglia", "FT GPUs", "Inf GPUs", "Train GPUs", "Idle GPUs", "Alloc GPUs", "Model kW"]
    series = [
        g,
        feats["n_req_fine_tuning"],
        feats["n_req_inference"],
        feats["n_req_training"],
        feats["n_eff_idle"],
        feats["alloc_gpus"],
        feats["modeled_fleet_kw"],
    ]
    C = np.corrcoef(np.vstack([np.asarray(s, float) for s in series]))
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(C[i, j]) > 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Feature correlation matrix vs Ganglia")
    fig.tight_layout()
    p = os.path.join(out_dir, "corr_heatmap.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # 3. Time series (z-scored shape comparison)
    def z(x):
        x = np.asarray(x, float)
        return (x - x.mean()) / (x.std() if x.std() else 1.0)

    t = np.arange(len(g)) / 12.0  # 5-min bins → hours
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.plot(t, z(g), color="#111", lw=1.8, label="Ganglia kW")
    ax.plot(t, z(feats["n_req_fine_tuning"]), color="#e07a3d", lw=1.2, label="Fine-tuning GPUs")
    ax.plot(t, z(feats["n_req_training"]), color="#3b6ea5", lw=1.2, alpha=0.9, label="Training GPUs")
    ax.plot(t, z(feats["n_req_inference"]), color="#6a9a6a", lw=1.0, alpha=0.75, label="Inference GPUs")
    ax.set_xlabel("Hours from 2022-03-20 00:00 UTC")
    ax.set_ylabel("z-score")
    ax.set_title("Shape match: fine-tuning occupancy tracks Ganglia; training does not")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    p = os.path.join(out_dir, "timeseries_zscore.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # 4. Parameter sweeps
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.4))
    axes = axes.ravel()
    for ax, row in zip(axes, report["_sweep_curves"]):
        xs = [p["value"] for p in row["curve"]]
        mapes = [p["mape_pct"] for p in row["curve"]]
        rs = [p["pearson_r"] for p in row["curve"]]
        ax.plot(xs, mapes, color="#3b6ea5", marker="o", ms=3, label="MAPE %")
        ax.set_xlabel(row["parameter"])
        ax.set_ylabel("MAPE %", color="#3b6ea5")
        ax2 = ax.twinx()
        ax2.plot(xs, rs, color="#e07a3d", marker="s", ms=3, label="r")
        ax2.set_ylabel("Pearson r", color="#e07a3d")
        ax.set_title(row["parameter"], fontsize=10)
        ax.grid(linestyle=":", alpha=0.5)
    fig.suptitle("One-at-a-time mean-field sweep (other knobs at literature defaults)")
    fig.tight_layout()
    p = os.path.join(out_dir, "param_sweeps.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # 5. Scatter FT vs ganglia
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    sc = ax.scatter(
        feats["n_req_fine_tuning"], g,
        c=np.arange(len(g)), cmap="viridis", s=14, alpha=0.85,
    )
    ax.set_xlabel("Requested fine-tuning GPUs")
    ax.set_ylabel("Ganglia fleet GPU power (kW)")
    r = pearson(feats["n_req_fine_tuning"], g)
    ax.set_title(f"Fine-tuning occupancy vs Ganglia  (r = {r:.2f})")
    fig.colorbar(sc, ax=ax, label="timestep (5 min)")
    ax.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    p = os.path.join(out_dir, "scatter_ft_vs_ganglia.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    return paths


def write_markdown(report: dict, path: str) -> None:
    rec = report["rl_search_recommendation"]
    lines = [
        "# Parameters correlated with Ganglia GPU power",
        "",
        f"Window: **{report['year_month']}** `{report['window']['t0']}` → `{report['window']['t1']}` "
        f"({report['n_points']} points at 5 min).",
        "",
        f"Measured source: `{report['measured_source'] or 'ganglia_pub (cached day slice)'}` "
        f"· mean {report['ganglia_mean_kw']} kW · current model r = "
        f"**{report['r_current_model_vs_ganglia']}**.",
        "",
        f"Requested GPUs exceed inventory on **{100 * report['oversubscribed_fraction']:.0f}%** "
        f"of bins (mean request {report['mean_requested_gpus']} vs "
        f"{list((report['inventory'] or {}).values())[0]} V100s). "
        "On this slice, occupancy is saturated; stage *mix* drives Ganglia variation.",
        "",
        "## Ranked physics knobs",
        "",
        "| Parameter | Pearson r (requested GPUs) | Partial r | MAPE span (1-D sweep) | RL priority |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in report["parameters"]:
        lines.append(
            f"| `{r['parameter']}` | {r['pearson_r_vs_ganglia']} | "
            f"{r['partial_r_vs_ganglia']} | {r['mape_span_pct']} | {r['rl_priority']} |"
        )
    lines += [
        "",
        "**Partial r** is the correlation of that parameter’s GPU-count regressor with "
        "Ganglia after linearly removing the other stages. That is the number to trust "
        "for “does this knob independently track measured power?”",
        "",
        "### Search first",
        "",
    ]
    for p in rec["search_first"]:
        lines.append(f"- `{p}`")
    lines += ["", "### Search next (level / mix)", ""]
    if rec["search_next"]:
        for p in rec["search_next"]:
            lines.append(f"- `{p}`")
    else:
        lines.append("- *(none on this slice)*")
    lines += ["", "### Confounders (raw r only — do not search yet)", ""]
    if rec.get("confounders"):
        for p in rec["confounders"]:
            lines.append(f"- `{p}`")
    else:
        lines.append("- *(none)*")
    lines += ["", "### Defer on this slice", ""]
    for p in rec["defer"]:
        lines.append(f"- `{p}`")
    lines += ["", "### Classification thresholds (high leverage, not a util knob)", ""]
    for p in rec["also_consider"]:
        lines.append(f"- `{p}`")
    lines += ["", "## Notes", ""]
    for n in rec["notes"]:
        lines.append(f"- {n}")
    ols = report["ols_mean_field"]
    lines += [
        "",
        "## OLS implied util (mean-field, no intercept)",
        "",
        f"Fit `P_ganglia ≈ Σ β_s n_eff,s` then `u = β / (P_max/1000)`. "
        f"R² = {ols['r2']}, r = {ols['pearson_r']}.",
        "",
        "| Term | kW / GPU | Implied util |",
        "|---|---:|---:|",
        f"| training | {ols['kw_per_gpu']['training']} | {ols['implied_util']['training.u_plateau']} |",
        f"| fine-tuning | {ols['kw_per_gpu']['fine_tuning']} | {ols['implied_util']['fine_tuning.u_plateau']} |",
        f"| inference | {ols['kw_per_gpu']['inference']} | {ols['implied_util']['inference.u_eff']} |",
        f"| idle (rho) | {ols['kw_per_gpu']['idle']} | {ols['implied_util']['rho']} |",
        "",
        "Training’s implied util is negative because its GPU count is a confounder "
        "(raw r vs Ganglia is negative; partial r ≈ 0). Fine-tuning implied util "
        "(~0.70) sits next to the literature plateau 0.68.",
        "",
        "## Plots",
        "",
        "![Correlation bars](../rl_tune/results/corr_bars.png)",
        "",
        "![Heatmap](../rl_tune/results/corr_heatmap.png)",
        "",
        "![Time series](../rl_tune/results/timeseries_zscore.png)",
        "",
        "![Sweeps](../rl_tune/results/param_sweeps.png)",
        "",
        "![FT scatter](../rl_tune/results/scatter_ft_vs_ganglia.png)",
        "",
        "## How to rerun",
        "",
        "```bash",
        "python -m rl_tune.correlate_params",
        "```",
        "",
        "Raw numbers: `rl_tune/results/ganglia_param_correlation.json`.",
        "",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    del argv
    report = analyze()
    out_dir = DEFAULT_OUT
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "ganglia_param_correlation.json")
    with open(json_path, "w") as f:
        json.dump(_jsonable(report), f, indent=2)
    plots = plot_report(report, out_dir)
    md_path = os.path.join(_ROOT, "docs", "ganglia_parameter_correlation.md")
    write_markdown(report, md_path)
    print(json.dumps({
        "json": json_path,
        "markdown": md_path,
        "plots": plots,
        "parameters": report["parameters"],
        "recommendation": report["rl_search_recommendation"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
