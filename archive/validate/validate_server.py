"""
IT Power Validation — overlay measured vs modeled power from the M100 dataset.

Loads real M100 job records, pushes their variables through the power model,
and overlays the modeled fleet power on the measured cluster power for the same
wall-clock window. Reports MAPE (power) and energy error against the targets.

This file is SELF-CONTAINED: it carries its own copy of the model engine
(mirrored from server.py — same threaded-rng pattern, same inference fixes,
same classify_stage) and loads config.json directly, so it does not depend on
the exact internals of server.py.

Run:  python validate_server.py   ->  http://localhost:5001
(config.json and exadata.py must sit in the same directory.)
"""
import json, math, os, random
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

import exadata

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__, static_folder=None)

with open(os.path.join(BASE_DIR, "config.json")) as f:
    CFG = json.load(f)

HW    = CFG["hardware"]
S_DEF = CFG["stage_defaults"]
S_PHY = CFG["stage_physics"]

# Validation config: prefer a "validate" block in config.json, else defaults.
VCFG = CFG.get("validate", {})
GPUS_PER_NODE  = VCFG.get("gpus_per_node", 4)
HW_TYPE        = VCFG.get("hardware_type", "V100")
FULL_INVENTORY = VCFG.get("inventory", {"V100": 3920})    # 980 nodes x 4 V100
DEFAULT_GRID_S = VCFG.get("grid_seconds", 60)
MAPE_FLOOR_KW  = VCFG.get("mape_floor_kw", 1.0)
DEFAULT_SOURCE = VCFG.get("default_source", "logics_ict")
RNG_SEED       = VCFG.get("rng_seed", 12345)


# ============================================================================
# MODEL ENGINE  (mirrors server.py; do not diverge without syncing server.py)
# Formula: P(t) = sum_k Nk*Pk_max [eta_k(t)*sum_m(um*alpha_m(t)) + (1-eta_k)*rho_k]
# ============================================================================
def clamp(x, lo, hi): return max(lo, min(hi, x))
def gauss(mean, std, rng): return rng.gauss(mean, std)

def poisson_draw(lam, rng):
    if lam <= 0: return 0
    if lam < 30:
        L, k, p = math.exp(-lam), 0, 1.0
        while True:
            k += 1; p *= rng.random()
            if p <= L: return k - 1
    return max(0, round(gauss(lam, math.sqrt(lam), rng)))

def generate_inference_schedule(task, N, rng):
    ph  = S_PHY["inference"]
    rph = task.get("requests_per_hour")
    lam = ph["default_lambda"] if rph is None else rph
    D   = max(1, (task["end_ms"] - task["start_ms"]) / 1000)
    tau = min(ph["service_s"], D)
    n_req = poisson_draw(lam * D / 3600, rng)
    sched = []
    for _ in range(n_req):
        dev   = rng.randint(0, max(0, N - 1))
        start = rng.uniform(0, max(0, D - tau))
        sched.append({"dev": dev, "start": start, "end": start + tau})
    return sched

def count_busy(schedule, N, t_s):
    busy = {req["dev"] for req in schedule if req["start"] <= t_s < req["end"]}
    return min(len(busy), N)

def stochastic_util(stage, elapsed_ms, task=None, schedule=None, N2D=1, rng=None, hw_rho=0.0):
    ph   = S_PHY[stage]
    def_ = S_DEF[stage]
    rng  = rng or random.Random()
    if stage == "training":
        since_ck = elapsed_ms % ph["checkpoint_ms"]
        if elapsed_ms > 0 and since_ck < ph["checkpoint_dur_ms"]:
            return clamp(gauss(ph["checkpoint_u"], ph["checkpoint_sigma"], rng),
                         ph["checkpoint_u_min"], ph["checkpoint_u_max"])
        return clamp(gauss(ph["u_plateau"], ph["sigma_plateau"], rng), def_["u_min"], def_["u_max"])
    elif stage == "fine_tuning":
        since_ev = elapsed_ms % ph["eval_period_ms"]
        if elapsed_ms > 0 and since_ev < ph["eval_dur_ms"]:
            return clamp(gauss(ph["eval_u"], ph["eval_sigma"], rng), ph["eval_u_min"], ph["eval_u_max"])
        return clamp(gauss(ph["u_plateau"], ph["sigma_plateau"], rng), def_["u_min"], def_["u_max"])
    else:
        u_idle_eff = max(ph["u_idle"], hw_rho)
        if schedule is not None:
            t_s    = elapsed_ms / 1000
            n_busy = count_busy(schedule, N2D, t_s)
            n_idle = N2D - n_busy
            uB = clamp(gauss(ph["u_burst"], ph["sigma_burst"], rng), 0, 1.0) if n_busy > 0 else 0
            uI = clamp(gauss(u_idle_eff,   ph["sigma_idle"],  rng), 0, ph["idle_u_max"]) if n_idle > 0 else 0
            return (n_busy * uB + n_idle * uI) / max(1, N2D)
        rph = task.get("requests_per_hour") if task else None
        lam = ph["default_lambda"] if rph is None else rph
        occ = min(1.0, lam * ph["service_s"] / 3600)
        if rng.random() < occ:
            return clamp(gauss(ph["u_burst"], ph["sigma_burst"], rng), 0, 1.0)
        return clamp(gauss(u_idle_eff, ph["sigma_idle"], rng), 0, ph["idle_u_max"])

def classify_stage(start_ms, end_ms, num_nodes, num_gpus, hw_type=""):
    rp      = CFG.get("replay", {})
    allowed = rp.get("stage_classify_for", ["V100"])
    if hw_type not in allowed:
        return "inference"
    dur_h    = (end_ms - start_ms) / 3_600_000.0
    thr      = rp.get("stage_thresholds", {})
    t_dur    = thr.get("training_min_duration_h",   1.0)
    t_node   = thr.get("training_min_nodes",        16)
    ft_min   = thr.get("finetuning_min_duration_h", 0.25)
    inf_node = thr.get("inference_max_nodes",       4)
    n        = num_nodes or 0
    if dur_h >= t_dur and n >= t_node:
        return "training"
    if dur_h < ft_min or n < inf_node:
        return "inference"
    return "fine_tuning"

def build_schedules(tasks, inventory, rng):
    schedules = {}
    for t in tasks:
        if t["stage"] != "inference": continue
        Nk = inventory.get(t["hardware_type"], 0)
        if Nk == 0: continue
        same  = [x for x in tasks if x["hardware_type"] == t["hardware_type"]]
        total = sum(x.get("num_devices", 0) for x in same)
        scale = 1.0 if total <= Nk else Nk / total
        N2D   = max(1, round(t.get("num_devices", 1) * scale))
        t["_N2D"] = N2D
        schedules[t["id"]] = generate_inference_schedule(t, N2D, rng)
    return schedules

def compute_macro_at(ts_ms, tasks, inventory, schedules, rngs):
    total_kw = 0.0
    for hw_key, hw in HW.items():
        Nk = inventory.get(hw_key, 0)
        if Nk == 0: continue
        active = [t for t in tasks if t["hardware_type"] == hw_key
                  and t["start_ms"] <= ts_ms < t["end_ms"]]
        if not active:
            total_kw += (Nk * hw["p_max"] * hw["rho"]) / 1000
            continue
        total_req = sum(t.get("num_devices", 0) for t in active)
        scale     = 1.0 if total_req <= Nk else Nk / total_req
        Nk_alloc  = min(total_req, Nk)
        eta_k     = Nk_alloc / Nk
        Uk = 0.0
        for t in active:
            n_eff   = t.get("num_devices", 0) * scale
            alpha_m = n_eff / Nk_alloc if Nk_alloc > 0 else 0
            elapsed = ts_ms - t["start_ms"]
            sched   = schedules.get(t["id"])
            N2D     = t.get("_N2D", 1)
            Uk     += alpha_m * stochastic_util(t["stage"], elapsed, t, sched, N2D,
                                                rngs[t["stage"]], hw["rho"])
        Uk = max(Uk, hw["rho"])
        total_kw += (Nk * hw["p_max"] * (eta_k * Uk + (1 - eta_k) * hw["rho"])) / 1000
    return total_kw


# -- jobs -> model tasks (absolute epoch-ms; no ref subtraction) -------------
def jobs_to_tasks(jobs_df):
    tasks, skipped = [], 0
    for i, row in jobs_df.iterrows():
        s = row.get("start_time"); e = row.get("end_time")
        if pd.isna(s) or pd.isna(e):
            skipped += 1; continue
        s_ms = int(pd.Timestamp(s).timestamp() * 1000)
        e_ms = int(pd.Timestamp(e).timestamp() * 1000)
        if e_ms <= s_ms:
            skipped += 1; continue
        nodes = int(row.get("num_nodes") or 0) or 1
        devs  = max(1, nodes * GPUS_PER_NODE)
        tasks.append({
            "id":               str(row.get("job_id", i)),
            "label":            str(row.get("job_id", f"job-{i}")),
            "hardware_type":    HW_TYPE,
            "stage":            classify_stage(s_ms, e_ms, nodes, devs, HW_TYPE),
            "start_ms":         s_ms,
            "end_ms":           e_ms,
            "num_devices":      devs,
            "requests_per_hour": None,
        })
    return tasks, skipped


# -- modeled fleet power on a wall-clock grid --------------------------------
def modeled_series(tasks, inventory, grid_ms, draws=1):
    acc = np.zeros(len(grid_ms))
    for d in range(max(1, draws)):
        rngs = {s: random.Random(RNG_SEED + 1000 * d + j)
                for j, s in enumerate(("training", "fine_tuning", "inference"))}
        schedules = build_schedules(tasks, inventory, rngs["inference"])
        acc += np.array([compute_macro_at(int(ts), tasks, inventory, schedules, rngs)
                         for ts in grid_ms])
    return acc / max(1, draws)


# -- measured power resampled onto the same grid -----------------------------
def resample_measured(meas_df, grid_ms, step_ms):
    if meas_df is None or meas_df.empty:
        return np.full(len(grid_ms), np.nan)
    ts = pd.DatetimeIndex(meas_df["timestamp"]).tz_convert("UTC").as_unit("ms").asi8
    kw = meas_df["kw"].to_numpy(dtype=float)
    edges = np.append(grid_ms, grid_ms[-1] + step_ms)
    idx = np.searchsorted(edges, ts, side="right") - 1
    out = np.full(len(grid_ms), np.nan)
    valid = (idx >= 0) & (idx < len(grid_ms))
    if valid.any():
        sums = np.bincount(idx[valid], weights=kw[valid], minlength=len(grid_ms))
        cnts = np.bincount(idx[valid], minlength=len(grid_ms))
        nz = cnts > 0
        out[nz] = sums[nz] / cnts[nz]
    good = ~np.isnan(out)
    if good.sum() >= 2:
        gi = np.where(good)[0]
        out = np.interp(np.arange(len(out)), gi, out[gi])
    return out


# -- error metrics -----------------------------------------------------------
def _trapz(y, x):
    fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(fn(y, x))

def error_metrics(measured, modeled, grid_ms):
    a = np.asarray(measured, float)
    m = np.asarray(modeled, float)
    mask = np.isfinite(a) & (a > MAPE_FLOOR_KW)
    mape = float(np.mean(np.abs(m[mask] - a[mask]) / a[mask]) * 100) if mask.any() else None
    hrs = (grid_ms - grid_ms[0]) / 3_600_000.0
    e_meas = _trapz(np.nan_to_num(a), hrs)
    e_mod  = _trapz(m, hrs)
    e_err  = abs(e_mod - e_meas) / e_meas * 100 if e_meas > 0 else None
    bias   = float(np.mean(m[mask] - a[mask])) if mask.any() else None
    return {
        "mape_pct":        round(mape, 2) if mape is not None else None,
        "energy_meas_kwh": round(e_meas, 1),
        "energy_mod_kwh":  round(e_mod, 1),
        "energy_err_pct":  round(e_err, 2) if e_err is not None else None,
        "bias_kw":         round(bias, 2) if bias is not None else None,
        "n_points":        int(len(grid_ms)),
        "n_compared":      int(mask.sum()),
    }


# -- routes ------------------------------------------------------------------
@app.route("/api/config")
def api_config():
    return jsonify({
        "hardware":      CFG["hardware"],
        "stage_physics": CFG["stage_physics"],
        "replay":        CFG.get("replay", {}),
        "validate": {
            "gpus_per_node":  GPUS_PER_NODE,
            "hardware_type":  HW_TYPE,
            "inventory":      FULL_INVENTORY,
            "grid_seconds":   DEFAULT_GRID_S,
            "default_source": DEFAULT_SOURCE,
            "sources": {k: {"label": v["label"], "plugin": v["plugin"]}
                        for k, v in exadata.MEASURED_SOURCES.items()},
        },
        "targets": {"power_mape_pct": 15, "energy_err_pct": 10},
    })


@app.route("/api/windows")
def api_windows():
    root = request.args.get("root") or exadata.DATA_ROOT
    return jsonify(exadata.list_windows(root))


@app.route("/api/metrics")
def api_metrics():
    root   = request.args.get("root") or exadata.DATA_ROOT
    ym     = request.args.get("ym")
    plugin = request.args.get("plugin")
    if not (ym and plugin):
        return jsonify({"error": "ym and plugin are required"}), 400
    try:
        return jsonify({"ym": ym, "plugin": plugin,
                        "metrics": exadata.list_metrics(root, ym, plugin)})
    except Exception as ex:
        return jsonify({"error": str(ex), "metrics": []}), 500


@app.route("/api/validate", methods=["POST"])
def api_validate():
    b = request.get_json(force=True)
    root   = b.get("root") or exadata.DATA_ROOT
    ym     = b.get("year_month")
    t0     = b.get("t0")
    t1     = b.get("t1")
    source = b.get("source", DEFAULT_SOURCE)
    grid_s = int(b.get("grid_seconds", DEFAULT_GRID_S))
    draws  = int(b.get("draws", 1))
    inv    = b.get("inventory", FULL_INVENTORY)
    if not (ym and t0 and t1):
        return jsonify({"error": "year_month, t0 and t1 are required"}), 400

    try:
        t0p = exadata._to_utc(t0); t1p = exadata._to_utc(t1)
    except Exception as ex:
        return jsonify({"error": f"bad timestamps: {ex}"}), 400
    if t1p <= t0p:
        return jsonify({"error": "t1 must be after t0"}), 400

    try:
        jobs_df = exadata.load_jobs(root, ym, t0p, t1p)
    except Exception as ex:
        return jsonify({"error": f"job load failed: {ex}"}), 500
    tasks, skipped = jobs_to_tasks(jobs_df)
    if not tasks:
        return jsonify({"error": "no overlapping jobs in window",
                        "job_rows": int(len(jobs_df))}), 400

    step_ms = grid_s * 1000
    g0 = int(t0p.timestamp() * 1000)
    g1 = int(t1p.timestamp() * 1000)
    grid_ms = np.arange(g0, g1 + 1, step_ms, dtype=np.int64)

    modeled = modeled_series(tasks, inv, grid_ms, draws=draws)
    metric_override = (b.get("metric") or "").strip() or None
    try:
        meas_df = exadata.load_measured(root, ym, t0p, t1p, source, metric=metric_override)
    except Exception as ex:
        return jsonify({"error": f"measured load failed: {ex}"}), 500
    measured = resample_measured(meas_df, grid_ms, step_ms)

    # If no measured rows matched, surface what metrics DO exist so the user can
    # pick the right one (the green line not showing usually means a name miss).
    available_metrics = []
    if len(meas_df) == 0:
        try:
            plugin = exadata.MEASURED_SOURCES[source]["plugin"]
            available_metrics = exadata.list_metrics(root, ym, plugin)
        except Exception:
            available_metrics = []

    metrics = error_metrics(measured, modeled, grid_ms)

    stage_counts = {}
    for t in tasks:
        stage_counts[t["stage"]] = stage_counts.get(t["stage"], 0) + 1

    labels = [pd.Timestamp(int(ms), unit="ms", tz="UTC").strftime("%m-%d %H:%M")
              for ms in grid_ms]

    return jsonify({
        "labels":   labels,
        "epoch_ms": grid_ms.tolist(),
        "measured": [None if not np.isfinite(x) else round(float(x), 3) for x in measured],
        "modeled":  [round(float(x), 3) for x in modeled],
        "residual": [None if not np.isfinite(a) else round(float(m - a), 3)
                     for a, m in zip(measured, modeled)],
        "metrics":  metrics,
        "meta": {
            "year_month":    ym,
            "source":        source,
            "source_label":  exadata.MEASURED_SOURCES.get(source, {}).get("label", source),
            "grid_seconds":  grid_s,
            "draws":         draws,
            "job_rows":      int(len(jobs_df)),
            "task_count":    len(tasks),
            "skipped":       skipped,
            "stage_counts":  stage_counts,
            "measured_samples": int(len(meas_df)),
            "metric":        metric_override or exadata.MEASURED_SOURCES.get(source, {}).get(
                                 "metric_match", ("", ""))[1],
            "available_metrics": available_metrics,
            "inventory":     inv,
        },
    })


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "validate.html")

@app.route("/<path:fn>")
def static_files(fn):
    return send_from_directory(BASE_DIR, fn)


if __name__ == "__main__":
    print("IT Power Validation -> http://localhost:5001")
    print(f"Dataset root: {exadata.DATA_ROOT}  (override with EXADATA_ROOT)")
    app.run(debug=True, port=5001)