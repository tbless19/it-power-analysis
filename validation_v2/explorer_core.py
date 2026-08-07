"""Schedule Explorer — measured M100 power + job timeline + stage breakdown."""
import json
import math
import os
import random
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import server  # noqa: E402
from validation import exadata  # noqa: E402

_V2 = os.path.dirname(os.path.abspath(__file__))
_SLICE_CFG_PATH = os.path.join(_V2, "slice_config.json")
_CACHE_DIR = os.path.join(_V2, "slice_cache")
_CACHE_FILE = os.path.join(_CACHE_DIR, "day_slice.json")

STAGES = ("training", "fine_tuning", "inference")
STAGE_LABELS = {
    "training":    "Training",
    "fine_tuning": "Fine-tuning",
    "inference":   "Inference",
}


def slice_cache_path():
    return _CACHE_FILE


def default_slice_config():
    with open(_SLICE_CFG_PATH) as f:
        return json.load(f)


def _load_project_config():
    with open(os.path.join(_ROOT, "config.json")) as f:
        return json.load(f)


def downsample_series(timestamps, values, grid_seconds, max_points):
    ts = pd.to_datetime(pd.Series(timestamps), utc=True)
    series = pd.Series(np.asarray(values, float), index=ts).sort_index()
    series = series.groupby(level=0).median()
    if grid_seconds and grid_seconds > 0:
        series = series.resample(f"{int(grid_seconds)}s").median().dropna()
    if max_points and len(series) > max_points:
        step = max(1, math.ceil(len(series) / max_points))
        series = series.iloc[::step]
    return series.index, series.to_numpy(dtype=float)


GANTT_JOB_COLUMNS = [
    "metric", "start_time", "end_time", "num_nodes", "job_id",
    "partition", "qos", "job_state", "user_id", "num_cpus", "num_tasks", "submit_time",
]


def load_jobs_extended(root, ym, t0, t1):
    """Like exadata.load_jobs_df but includes partition/qos/etc. for the detail panel."""
    root = exadata._resolve_root(root or exadata.DATA_ROOT)
    path = exadata._part_file(root, ym, exadata.JOB_PLUGIN)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")

    t0p, t1p = exadata._to_utc(t0), exadata._to_utc(t1)
    df = exadata.read_columns(path, GANTT_JOB_COLUMNS)
    if "metric" in df.columns:
        df = df[df["metric"] == "job_info_marconi100"]
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True, errors="coerce")
    df["end_time"] = pd.to_datetime(df["end_time"], utc=True, errors="coerce")
    if "submit_time" in df.columns:
        df["submit_time"] = pd.to_datetime(df["submit_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["start_time", "end_time"])
    df = df[(df["end_time"] > df["start_time"]) & (df["start_time"] < t1p) & (df["end_time"] > t0p)]
    return df.reset_index(drop=True)


def _select_gantt_jobs(jobs_df, t0_ms, t1_ms, hw_type, min_nodes, max_jobs,
                       per_stage_max=None, inference_min_nodes=1):
    """
    Pick Gantt rows. When per_stage_max is set, take top jobs *per stage* so
    inference/fine-tuning are visible (not crowded out by large training jobs).
    """
    buckets = {s: [] for s in STAGES}
    for _, r in jobs_df.iterrows():
        s = int(r["start_time"].timestamp() * 1000)
        e = int(r["end_time"].timestamp() * 1000)
        if e <= t0_ms or s >= t1_ms:
            continue
        n_nodes = int(r.get("num_nodes") or 1)
        job_id = str(r.get("job_id", ""))
        row = {
            "job_id": job_id,
            "num_nodes": n_nodes,
            "num_gpus": n_nodes * 4,
            "start_ms": max(0, s - t0_ms),
            "end_ms": min(t1_ms - t0_ms, e - t0_ms),
            "start_iso": r["start_time"].isoformat(),
            "end_iso": r["end_time"].isoformat(),
            "duration_h": round((e - s) / 3_600_000, 2),
            "_sort": n_nodes * (e - s),
        }
        for col in ("partition", "qos", "job_state", "user_id"):
            if col in r.index and pd.notna(r[col]):
                row[col] = str(r[col])
        for col in ("num_cpus", "num_tasks"):
            if col in r.index and pd.notna(r[col]):
                row[col] = int(r[col])
        if "submit_time" in r.index and pd.notna(r["submit_time"]):
            row["submit_iso"] = pd.Timestamp(r["submit_time"]).isoformat()

        task = server.job_to_task({
            "job_id": job_id,
            "start_time": row["start_iso"],
            "end_time": row["end_iso"],
            "num_nodes": n_nodes,
            "num_gpus": row["num_gpus"],
        }, 0, hw_type, t0_ms)
        stage = task["stage"] if task else "inference"
        row["stage"] = stage
        row["stage_label"] = STAGE_LABELS.get(stage, stage)

        floor = inference_min_nodes if stage == "inference" else min_nodes
        if n_nodes < floor:
            continue
        buckets[stage].append(row)

    if per_stage_max:
        rows = []
        for stage in STAGES:
            picked = sorted(buckets[stage], key=lambda x: -x["_sort"])[:per_stage_max]
            rows.extend(picked)
        rows.sort(key=lambda x: (x["start_ms"], -x["_sort"]))
    else:
        rows = []
        for stage in STAGES:
            rows.extend(buckets[stage])
        rows.sort(key=lambda x: (-x["_sort"], x["start_ms"]))
        rows = rows[:max_jobs]

    if per_stage_max and max_jobs and len(rows) > max_jobs:
        rows = rows[:max_jobs]

    for row in rows:
        del row["_sort"]
    return rows


def _nodes_for_task(task, gpus_per_node=4):
    if task.get("num_nodes"):
        return int(task["num_nodes"])
    dev = int(task.get("num_devices") or 1)
    return max(1, dev // gpus_per_node)


def _compute_active_stats(ts_idx, t0_ms, tasks, hw_type):
    """Active job count and node sum per stage at each grid point (all jobs)."""
    jobs_series = {s: [] for s in STAGES}
    nodes_series = {s: [] for s in STAGES}
    active = []
    pending_i = 0

    for t_i in ts_idx:
        ts_ms = int(t_i.timestamp() * 1000) - t0_ms
        while pending_i < len(tasks) and tasks[pending_i]["start_ms"] <= ts_ms:
            active.append(tasks[pending_i])
            pending_i += 1
        if active:
            active = [t for t in active if t["end_ms"] > ts_ms]

        counts = {s: 0 for s in STAGES}
        nodes = {s: 0 for s in STAGES}
        for t in active:
            if t["hardware_type"] != hw_type:
                continue
            st = t["stage"]
            counts[st] += 1
            nodes[st] += _nodes_for_task(t)

        for s in STAGES:
            jobs_series[s].append(counts[s])
            nodes_series[s].append(nodes[s])

    return jobs_series, nodes_series


def _fleet_at(ts_ms, same_hw, inventory, hw_type, rngs, rho, p_max, Nk):
    """Instantaneous fleet + stage kW and allocated device count at ts_ms."""
    if not same_hw:
        idle = (Nk * p_max * rho) / 1000 if Nk else 0.0
        return idle, {s: 0.0 for s in STAGES}, 0

    n_alloc = sum(t.get("num_devices", 0) for t in same_hw)
    total_req = n_alloc
    scale = 1.0 if total_req <= Nk else Nk / total_req
    Nk_alloc = min(total_req, Nk)
    eta_k = Nk_alloc / Nk if Nk else 0
    Uk = 0.0
    stage_u_sum = {s: 0.0 for s in STAGES}

    for t in same_hw:
        n_eff = t.get("num_devices", 0) * scale
        alpha_m = n_eff / Nk_alloc if Nk_alloc > 0 else 0
        elapsed = ts_ms - t["start_ms"]
        u = server.stochastic_util(
            t["stage"], elapsed, t, None, 1, rngs[t["stage"]], rho)
        Uk += alpha_m * u
        stage_u_sum[t["stage"]] += n_eff * p_max * max(u, rho) / 1000

    Uk = max(Uk, rho)
    fleet = (Nk * p_max * (eta_k * Uk + (1 - eta_k) * rho)) / 1000
    return fleet, stage_u_sum, n_alloc


def _compute_series(ts_idx, t0_ms, tasks, inventory, hw_type, seed, draws,
                    grid_seconds=300, substep_seconds=30):
    """
    Fleet + stage power on the reporting grid.

    Point-sampling at grid_seconds aliases training checkpoints (period == 300 s)
    and FT evals (240 s). Instead evaluate at substep_seconds within each bin and
    average — same duty cycle every task should see, matching a 5-min median of a
    finer measured signal.
    """
    n = len(ts_idx)
    fleet_kw = np.zeros(n, dtype=float)
    stage_kw = {s: np.zeros(n, dtype=float) for s in STAGES}
    alloc_gpus = np.zeros(n, dtype=float)

    sub_ms = max(1, int(substep_seconds) * 1000)
    grid_ms = max(sub_ms, int(grid_seconds) * 1000)
    n_sub = max(1, grid_ms // sub_ms)

    # (bin_index, elapsed_ms) — chronological if ts_idx is sorted
    samples = []
    for i, t_i in enumerate(ts_idx):
        bin_start = int(t_i.timestamp() * 1000) - t0_ms
        for k in range(n_sub):
            samples.append((i, bin_start + k * sub_ms))

    n_draws = max(1, int(draws))
    for d in range(n_draws):
        rngs = {
            s: random.Random(seed + 1000 * d + j)
            for j, s in enumerate(STAGES)
        }
        active = []
        pending_i = 0
        hw = server.HW.get(hw_type, {})
        rho = hw.get("rho", 0.117)
        p_max = hw.get("p_max", 300)
        Nk = inventory.get(hw_type, 0)

        for bin_i, ts_ms in samples:
            while pending_i < len(tasks) and tasks[pending_i]["start_ms"] <= ts_ms:
                active.append(tasks[pending_i])
                pending_i += 1
            if active:
                active = [t for t in active if t["end_ms"] > ts_ms]

            same_hw = [t for t in active if t["hardware_type"] == hw_type]
            p, stage_u, n_alloc = _fleet_at(
                ts_ms, same_hw, inventory, hw_type, rngs, rho, p_max, Nk)
            fleet_kw[bin_i] += p
            alloc_gpus[bin_i] += n_alloc
            for s in STAGES:
                stage_kw[s][bin_i] += stage_u[s]

    denom = float(n_draws * n_sub)
    fleet_kw /= denom
    alloc_gpus = np.round(alloc_gpus / denom).astype(int)
    for s in STAGES:
        stage_kw[s] /= denom

    return fleet_kw, stage_kw, alloc_gpus


def build_slice(cfg, force=False, cache_path=None):
    """Load M100 window, compute series, write JSON cache."""
    cache_file = cache_path or _CACHE_FILE
    cache_dir = os.path.dirname(cache_file)
    os.makedirs(cache_dir, exist_ok=True)
    if os.path.exists(cache_file) and not force:
        with open(cache_file) as f:
            return json.load(f)

    root = exadata._resolve_root(cfg.get("root") or exadata.DATA_ROOT)
    ym = cfg["year_month"]
    t0 = cfg["t0"]
    t1 = cfg["t1"]
    grid_s = int(cfg.get("grid_seconds", 300))
    substep_s = int(cfg.get("substep_seconds", 30))
    max_pts = int(cfg.get("measured_max_points", 500))
    gantt_min = int(cfg.get("gantt_min_nodes", 16))
    gantt_max = int(cfg.get("gantt_max_jobs", 120))
    gantt_per_stage = cfg.get("gantt_per_stage_max", 40)
    gantt_inf_min = int(cfg.get("gantt_inference_min_nodes", 1))
    hw_type = cfg.get("hw_type", "V100")
    seed = int(cfg.get("seed", 12345))
    draws = max(1, int(cfg.get("draws", 1)))

    project_cfg = _load_project_config()
    inventory = project_cfg.get("m100_preset", {}).get("inventory", {"V100": 3920})
    gpus_per_node = project_cfg.get("m100_preset", {}).get("gpus_per_node", 4)

    ganglia_df = exadata.load_ganglia_gpu_power(root, ym, t0, t1)
    jobs_df = load_jobs_extended(root, ym, t0, t1)
    jobs = exadata.jobs_to_records(jobs_df, gpus_per_node=gpus_per_node)

    # Time grid from measured GPU power (ganglia)
    ts_idx, gang_kw = downsample_series(
        ganglia_df["timestamp"], ganglia_df["kw"], grid_s, max_pts)

    t0_ms = int(pd.Timestamp(t0, tz="UTC").timestamp() * 1000)
    t1_ms = int(pd.Timestamp(t1, tz="UTC").timestamp() * 1000)
    window_ms = t1_ms - t0_ms

    gantt_jobs = _select_gantt_jobs(
        jobs_df, t0_ms, t1_ms, hw_type, gantt_min, gantt_max,
        per_stage_max=gantt_per_stage, inference_min_nodes=gantt_inf_min)

    tasks = []
    for i, job in enumerate(jobs):
        task = server.job_to_task(job, i, hw_type, t0_ms)
        if task is not None:
            task["num_nodes"] = int(job.get("num_nodes") or max(1, task.get("num_devices", 1) // gpus_per_node))
            tasks.append(task)
    tasks.sort(key=lambda t: t["start_ms"])

    active_jobs, active_nodes = _compute_active_stats(ts_idx, t0_ms, tasks, hw_type)
    fleet_kw, stage_kw, alloc_gpus = _compute_series(
        ts_idx, t0_ms, tasks, inventory, hw_type, seed, draws,
        grid_seconds=grid_s, substep_seconds=substep_s)

    def _corr(a, b):
        aa, bb = np.asarray(a, float), np.asarray(b, float)
        mask = np.isfinite(aa) & np.isfinite(bb)
        if mask.sum() < 3 or np.std(aa[mask]) == 0 or np.std(bb[mask]) == 0:
            return None
        return round(float(np.corrcoef(aa[mask], bb[mask])[0, 1]), 4)

    stage_counts = {}
    for t in tasks:
        stage_counts[t["stage"]] = stage_counts.get(t["stage"], 0) + 1

    labels = [t.strftime("%m-%d %H:%M") for t in ts_idx]
    epoch_ms = [int(t.timestamp() * 1000) - t0_ms for t in ts_idx]

    result = {
        "measured": {
            "labels": labels,
            "epoch_ms": epoch_ms,
            "kw": [round(float(x), 3) for x in gang_kw],
            "source": "ganglia_pub Gpu*_power_usage sum",
            "n_nodes": int(ganglia_df.attrs.get("n_nodes", 0)),
            "n_rows": int(ganglia_df.attrs.get("n_rows", 0)),
        },
        "modeled": {
            "fleet_kw": [round(float(x), 3) for x in fleet_kw],
            "alloc_gpus": [int(x) for x in alloc_gpus],
            "by_stage": {
                s: [round(float(x), 3) for x in stage_kw[s].tolist()]
                for s in STAGES
            },
            "active_jobs": {s: active_jobs[s] for s in STAGES},
            "active_nodes": {s: active_nodes[s] for s in STAGES},
        },
        "gantt": {
            "jobs": gantt_jobs,
            "duration_ms": window_ms,
            "t0_ms": t0_ms,
            "min_nodes": gantt_min,
            "max_jobs": gantt_max,
            "per_stage_max": gantt_per_stage,
            "by_stage": {
                s: sum(1 for j in gantt_jobs if j.get("stage") == s)
                for s in STAGES
            },
            "total_jobs_in_window": len(jobs_df),
        },
        "meta": {
            "root": root,
            "year_month": ym,
            "window": {"t0": t0, "t1": t1},
            "grid_seconds": grid_s,
            "substep_seconds": substep_s,
            "measured_points": len(ts_idx),
            "gantt_jobs": len(gantt_jobs),
            "job_count": len(tasks),
            "stage_counts": stage_counts,
            "hw_type": hw_type,
            "inventory": inventory,
            "peak_ganglia_kw": round(float(np.max(gang_kw)), 2),
            "mean_ganglia_kw": round(float(np.mean(gang_kw)), 2),
            "peak_fleet_kw": round(float(np.max(fleet_kw)), 2),
            "mean_fleet_kw": round(float(np.mean(fleet_kw)), 2),
            "ganglia_nodes": int(ganglia_df.attrs.get("n_nodes", 0)),
            "r_fleet_vs_ganglia": _corr(fleet_kw, gang_kw),
        },
    }

    with open(cache_file, "w") as f:
        json.dump(result, f)

    return result


def _cache_is_stale(data):
    """Rebuild if cache still has Tot_ict or lacks ganglia-as-measured."""
    g = data.get("gantt", {})
    by = g.get("by_stage")
    if not by:
        return True
    per_stage = g.get("per_stage_max") or default_slice_config().get("gantt_per_stage_max")
    if per_stage and by.get("inference", 0) == 0:
        return True
    modeled = data.get("modeled", {})
    if "active_jobs" not in modeled or "active_nodes" not in modeled:
        return True
    measured = data.get("measured") or {}
    src = str(measured.get("source") or "")
    if "ganglia" not in src.lower():
        return True
    meta = data.get("meta") or {}
    if "peak_ganglia_kw" not in meta:
        return True
    if "substep_seconds" not in meta:
        return True
    return False


def load_slice(force_rebuild=False, cfg=None, cache_path=None):
    cache_file = cache_path or _CACHE_FILE
    config = cfg or default_slice_config()
    if force_rebuild or not os.path.exists(cache_file):
        return build_slice(config, force=True, cache_path=cache_file)
    with open(cache_file) as f:
        data = json.load(f)
    if _cache_is_stale(data):
        return build_slice(config, force=True, cache_path=cache_file)
    return data
