"""Validation logic: drive server.py on measured timestamps, compute metrics."""
import math
import random
import sys
import os

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import server  # noqa: E402


def downsample_grid(timestamps_utc, values_kw, grid_seconds, max_points=120):
    """Resample measured series to at most one point per grid_seconds bin."""
    ts = pd.to_datetime(pd.Series(timestamps_utc), utc=True)
    series = pd.Series(np.asarray(values_kw, float), index=ts).sort_index()
    if grid_seconds and grid_seconds > 0:
        series = series.groupby(level=0).median()
        series = series.resample(f"{int(grid_seconds)}s").median().dropna()
    if max_points and len(series) > max_points:
        step = max(1, math.ceil(len(series) / max_points))
        series = series.iloc[::step]
    return series.index, series.to_numpy(dtype=float)


def _hw_rho(hw_key, hw, rho_override):
    if rho_override is not None and hw_key == rho_override.get("hw_type"):
        return float(rho_override["rho"])
    return hw["rho"]


def fleet_idle_floor_kw(inventory, rho_override=None, hw_type="V100"):
    """Full-fleet idle power (kW) at rho for the primary hardware type."""
    hw = server.HW.get(hw_type, {})
    Nk = inventory.get(hw_type, 0)
    if Nk == 0 or not hw:
        return 0.0
    rho = _hw_rho(hw_type, hw, rho_override)
    return (Nk * hw["p_max"] * rho) / 1000


def _fleet_kw_at(ts_ms, active_tasks, inventory, schedules, rngs, rho_override=None):
    """Same formula as server.compute_macro_at, but only over pre-filtered active tasks."""
    total_kw = 0.0
    for hw_key, hw in server.HW.items():
        Nk = inventory.get(hw_key, 0)
        if Nk == 0:
            continue
        rho = _hw_rho(hw_key, hw, rho_override)
        scale_hw = Nk * hw["p_max"] / 1000
        active = [t for t in active_tasks if t["hardware_type"] == hw_key]
        if not active:
            total_kw += scale_hw * rho
            continue
        total_req = sum(t.get("num_devices", 0) for t in active)
        scale = 1.0 if total_req <= Nk else Nk / total_req
        Nk_alloc = min(total_req, Nk)
        eta_k = Nk_alloc / Nk
        Uk = 0.0
        for t in active:
            n_eff = t.get("num_devices", 0) * scale
            alpha_m = n_eff / Nk_alloc if Nk_alloc > 0 else 0
            elapsed = ts_ms - t["start_ms"]
            sched = schedules.get(t["id"])
            N2D = t.get("_N2D", 1)
            Uk += alpha_m * server.stochastic_util(
                t["stage"], elapsed, t, sched, N2D, rngs[t["stage"]], rho)
        Uk = max(Uk, rho)
        total_kw += scale_hw * (eta_k * Uk + (1 - eta_k) * rho)
    return total_kw


def simulate_on_grid(jobs, timestamps_utc, inventory, hw_type="V100", seed=0, draws=1,
                     rho_override=None):
    """
    Evaluate server.py model at each UTC timestamp using a time-sweep over tasks
    (avoids re-scanning all jobs at every grid point).
    """
    ts = pd.to_datetime(pd.Series(timestamps_utc), utc=True).reset_index(drop=True)
    if ts.empty:
        return np.array([]), []

    ref_ms = int(ts.iloc[0].timestamp() * 1000)
    tasks = []
    for i, job in enumerate(jobs):
        task = server.job_to_task(job, i, hw_type, ref_ms)
        if task is not None:
            tasks.append(task)

    if not tasks:
        return np.zeros(len(ts)), []

    tasks.sort(key=lambda t: t["start_ms"])
    acc = np.zeros(len(ts), dtype=float)
    n_draws = max(1, int(draws))

    for d in range(n_draws):
        rngs = {
            s: random.Random(seed + 1000 * d + j)
            for j, s in enumerate(("training", "fine_tuning", "inference"))
        }
        schedules = server.build_schedules(tasks, inventory, rngs["inference"])

        active = []
        pending_i = 0
        n_tasks = len(tasks)

        for i, t_i in enumerate(ts):
            ts_ms = int(t_i.timestamp() * 1000) - ref_ms
            while pending_i < n_tasks and tasks[pending_i]["start_ms"] <= ts_ms:
                active.append(tasks[pending_i])
                pending_i += 1
            if active:
                active = [t for t in active if t["end_ms"] > ts_ms]
            acc[i] += _fleet_kw_at(ts_ms, active, inventory, schedules, rngs, rho_override)

    return acc / n_draws, tasks


def compute_metrics(measured_kw, calculated_kw, timestamps_utc, mape_floor_kw=1.0):
    meas = np.asarray(measured_kw, float)
    calc = np.asarray(calculated_kw, float)
    mask = np.isfinite(meas) & (meas > mape_floor_kw)

    mape = float(np.mean(np.abs(calc[mask] - meas[mask]) / meas[mask]) * 100) if mask.any() else None
    bias = float(np.mean(calc[mask] - meas[mask])) if mask.any() else None

    t_h = pd.to_datetime(pd.Series(timestamps_utc), utc=True).astype("int64").to_numpy() / 1e9 / 3600.0
    trapz = getattr(np, "trapezoid", getattr(np, "trapz"))
    e_meas = float(trapz(meas, t_h))
    e_calc = float(trapz(calc, t_h))
    energy_err = abs(e_calc - e_meas) / e_meas * 100 if e_meas > 0 else None

    return {
        "mape_pct":        round(mape, 2) if mape is not None else None,
        "energy_err_pct":  round(energy_err, 2) if energy_err is not None else None,
        "bias_kw":         round(bias, 2) if bias is not None else None,
        "e_meas_kwh":      round(e_meas, 1),
        "e_calc_kwh":      round(e_calc, 1),
        "peak_meas":       round(float(meas.max()), 2),
        "peak_calc":       round(float(calc.max()), 2),
        "mean_meas":       round(float(meas.mean()), 2),
        "mean_calc":       round(float(calc.mean()), 2),
        "n_points":        int(len(meas)),
        "n_compared":      int(mask.sum()),
    }


def run_validation(measured_df, jobs, inventory, hw_type="V100", grid_seconds=300,
                   seed=0, draws=1, mape_floor_kw=1.0, max_grid_points=120,
                   baseline_kw=0.0):
    """
    Full pipeline: align measured grid, simulate, compute metrics.
    Returns dict ready for JSON serialization.
    """
    ts_idx, meas_kw = downsample_grid(
        measured_df["timestamp"], measured_df["kw"], grid_seconds, max_grid_points)

    if len(ts_idx) == 0:
        raise ValueError("No measured points after downsampling")

    hw = server.HW.get(hw_type, {})
    rho_used = float(hw.get("rho", 0.117))
    rho_override = {"hw_type": hw_type, "rho": rho_used}

    calc_kw, tasks = simulate_on_grid(
        jobs, ts_idx, inventory, hw_type=hw_type, seed=seed, draws=draws,
        rho_override=rho_override)

    baseline_kw = float(baseline_kw or 0.0)
    model_kw = calc_kw.copy()
    if baseline_kw:
        calc_kw = calc_kw + baseline_kw

    metrics = compute_metrics(meas_kw, calc_kw, ts_idx, mape_floor_kw=mape_floor_kw)
    epoch_ms = [int(t.timestamp() * 1000) for t in ts_idx]
    labels = [t.strftime("%m-%d %H:%M") for t in ts_idx]
    residual = [round(float(c - m), 3) for m, c in zip(meas_kw, calc_kw)]

    stage_counts = {}
    for t in tasks:
        stage_counts[t["stage"]] = stage_counts.get(t["stage"], 0) + 1

    idle_w = round(rho_used * hw.get("p_max", 0), 1) if hw else None

    return {
        "labels":     labels,
        "epoch_ms":   epoch_ms,
        "measured":   [round(float(x), 3) for x in meas_kw],
        "model":      [round(float(x), 3) for x in model_kw],
        "calculated": [round(float(x), 3) for x in calc_kw],
        "residual":   residual,
        "metrics":    metrics,
        "meta": {
            "task_count":      len(tasks),
            "job_count":       len(jobs),
            "stage_counts":    stage_counts,
            "grid_points":     len(ts_idx),
            "max_grid_points": max_grid_points,
            "window": {
                "t0": ts_idx[0].isoformat(),
                "t1": ts_idx[-1].isoformat(),
            },
            "grid_seconds": grid_seconds,
            "draws":        draws,
            "inventory":    inventory,
            "hw_type":      hw_type,
            "rho":          round(rho_used, 4) if rho_used is not None else None,
            "idle_w_per_gpu": idle_w,
            "fleet_idle_floor_kw": round(
                fleet_idle_floor_kw(inventory, rho_override, hw_type), 1),
            "baseline_kw": round(baseline_kw, 1),
        },
    }
