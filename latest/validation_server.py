"""
IT Power Validation Server
Loads real M100/ExaData Parquet data, extracts IT power (Tot_ict) and
job/utilisation variables, runs the simulation model, and returns both
series for comparison charting.

Run: python validation_server.py  →  http://localhost:5050
"""
import json, math, random, os, io
from datetime import datetime, timezone
from flask import Flask, jsonify, request, send_from_directory

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

# ── Robust Parquet reader ─────────────────────────────────────────────────────
def robust_read_parquet(parquet_bytes):
    """
    Read a Parquet file from bytes using multiple fallback strategies.
    Handles Spark/Hive files with 'Repetition level histogram size mismatch'
    and other metadata quirks that strict PyArrow rejects.
    Returns a pandas DataFrame or raises the last exception.
    """
    buf = io.BytesIO(parquet_bytes)
    errors = []

    # Strategy 1: PyArrow with pre_buffer=False (avoids some prefetch issues)
    try:
        buf.seek(0)
        table = pq.read_table(buf, pre_buffer=False)
        return table.to_pandas()
    except Exception as e:
        errors.append(f"pyarrow/pre_buffer=False: {e}")

    # Strategy 2: PyArrow ParquetFile reading row-groups individually
    try:
        buf.seek(0)
        pf = pq.ParquetFile(buf, pre_buffer=False)
        chunks = []
        for i in range(pf.metadata.num_row_groups):
            try:
                chunks.append(pf.read_row_group(i).to_pandas())
            except Exception:
                pass  # skip bad row groups
        if chunks:
            return pd.concat(chunks, ignore_index=True)
        errors.append("pyarrow/row_groups: all row groups failed")
    except Exception as e:
        errors.append(f"pyarrow/row_groups: {e}")

    # Strategy 3: pandas with fastparquet engine
    try:
        buf.seek(0)
        return pd.read_parquet(buf, engine="fastparquet")
    except Exception as e:
        errors.append(f"fastparquet: {e}")

    # Strategy 4: pandas with pyarrow engine (uses different code path)
    try:
        buf.seek(0)
        return pd.read_parquet(buf, engine="pyarrow")
    except Exception as e:
        errors.append(f"pandas/pyarrow: {e}")

    raise ValueError("All Parquet read strategies failed:\n" + "\n".join(errors))

BASE_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__) if '__file__' in dir() else os.getcwd()))
app = Flask(__name__, static_folder=BASE_DIR)

# ── Config ────────────────────────────────────────────────────────────────────
with open(os.path.join(BASE_DIR, "config.json")) as f:
    CFG = json.load(f)

HW      = CFG["hardware"]
S_DEF   = CFG["stage_defaults"]
S_PHY   = CFG["stage_physics"]
SIM_CFG = CFG["simulation"]
CALC_STEP_MS  = SIM_CFG["calc_step_ms"]
MICRO_MAX_PTS = SIM_CFG["micro_max_pts"]
MACRO_MAX_PTS = SIM_CFG["macro_max_pts"]
MICRO_STEP_MS = {"training": 1000, "fine_tuning": 100, "inference": 10}

# ── Math helpers ──────────────────────────────────────────────────────────────
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

def fmt_elapsed(ms):
    if ms == 0:   return "0s"
    if ms < 1000: return f"{int(ms)}ms"
    s = ms / 1000
    if s < 10:    return f"{s:.1f}s"
    if s < 60:    return f"{s:.0f}s"
    if s < 3600:  return f"{s/60:.0f}min"
    return f"{s/3600:.1f}h"

def fmt_window(ms):
    s = ms / 1000
    if s < 60:   return f"{s:.0f}s"
    if s < 3600: return f"{s/60:.0f}min"
    return f"{s/3600:.1f}h"

# ── Simulation core (identical to server.py) ──────────────────────────────────
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
            uI = clamp(gauss(u_idle_eff, ph["sigma_idle"], rng), 0, ph["idle_u_max"]) if n_idle > 0 else 0
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
    dur_h = (end_ms - start_ms) / 3_600_000.0
    thr   = rp.get("stage_thresholds", {})
    t_dur  = thr.get("training_min_duration_h",   1.0)
    t_node = thr.get("training_min_nodes",        16)
    ft_min = thr.get("finetuning_min_duration_h", 0.25)
    inf_node = thr.get("inference_max_nodes",     4)
    n = num_nodes or 0
    if dur_h >= t_dur and n >= t_node:
        return "training"
    if dur_h < ft_min or n < inf_node:
        return "inference"
    return "fine_tuning"

def iso_to_epoch_ms(iso_str):
    if not iso_str:
        return None
    try:
        dt = datetime.fromisoformat(str(iso_str).strip().replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except (ValueError, TypeError):
        return None

def job_to_task(job, idx, hw_type, ref_ms):
    s_ms = iso_to_epoch_ms(job.get("start_time"))
    e_ms = iso_to_epoch_ms(job.get("end_time"))
    if s_ms is None or e_ms is None or e_ms <= s_ms:
        return None
    elapsed_s = s_ms - ref_ms
    elapsed_e = e_ms - ref_ms
    num_nodes = int(job.get("num_nodes") or 1)
    num_gpus  = int(job.get("num_gpus")  or 0)
    num_devs  = max(1, num_gpus if num_gpus > 0 else num_nodes)
    return {
        "id":                str(job.get("job_id", idx)),
        "label":             str(job.get("job_id", f"job-{idx}")),
        "hardware_type":     hw_type,
        "stage":             classify_stage(elapsed_s, elapsed_e, num_nodes, num_gpus, hw_type),
        "start_ms":          elapsed_s,
        "end_ms":            elapsed_e,
        "num_devices":       num_devs,
        "requests_per_hour": None,
    }

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

def compute_chart_data(tasks, inventory, infra_kw, start_ms, dur_ms, rngs):
    end_ms    = start_ms + dur_ms
    n_pts     = math.ceil(dur_ms / CALC_STEP_MS)
    skip      = max(1, math.ceil(n_pts / MACRO_MAX_PTS))
    step_ms   = CALC_STEP_MS * skip
    schedules = build_schedules(tasks, inventory, rngs["inference"])
    rows = []
    for ts in range(int(start_ms), int(end_ms) + 1, int(step_ms)):
        mkw = compute_macro_at(ts, tasks, inventory, schedules, rngs)
        rows.append({
            "label":    fmt_elapsed(ts - start_ms),
            "ts_ms":    ts,
            "base_kw":  round(infra_kw, 3),
            "tasks_kw": round(mkw, 3),
            "total_kw": round(mkw + infra_kw, 3),
        })
    totals = [r["total_kw"] for r in rows]
    fleet  = [r["tasks_kw"] for r in rows]
    stats  = {
        "peak_kw":       round(max(totals), 2) if totals else 0,
        "avg_kw":        round(sum(totals) / len(totals), 2) if totals else 0,
        "task_count":    len(tasks),
        "peak_fleet_kw": round(max(fleet), 2) if fleet else 0,
    }
    return rows, stats, schedules

# ── Parquet helpers ───────────────────────────────────────────────────────────
def load_logics_ict(parquet_bytes, t0_utc=None, t1_utc=None):
    """Return resampled Tot_ict time series as list of {ts_ms, kw}."""
    try:
        df = robust_read_parquet(parquet_bytes)
    except Exception as e:
        return None, str(e)
    # Filter to Tot_ict on panel=generals / device=pue (facility IT total)
    mask = df["metric"] == "Tot_ict"
    if "panel" in df.columns:
        mask &= df["panel"].astype(str).isin(["generals", "nan", ""])
    if "device" in df.columns:
        mask &= df["device"].astype(str).isin(["pue", "nan", ""])
    sub = df[mask].copy()
    # If no panel/device filter matched, try just metric
    if sub.empty:
        sub = df[df["metric"] == "Tot_ict"].copy()
    if sub.empty:
        return None, "No Tot_ict rows found in file"
    sub["timestamp"] = pd.to_datetime(sub["timestamp"], utc=True, errors="coerce")
    sub = sub.dropna(subset=["timestamp", "value_numeric"])
    sub = sub.sort_values("timestamp")
    if t0_utc:
        sub = sub[sub["timestamp"] >= pd.Timestamp(t0_utc, tz="UTC")]
    if t1_utc:
        sub = sub[sub["timestamp"] < pd.Timestamp(t1_utc, tz="UTC")]
    if sub.empty:
        return None, "No Tot_ict rows in the requested time window"
    # Resample to 1-minute bins (median to reduce noise)
    sub = sub.set_index("timestamp")
    resampled = sub["value_numeric"].resample("1min").median().dropna()
    ref_ts = resampled.index[0]
    result = []
    for ts, val in resampled.items():
        elapsed_ms = int((ts - ref_ts).total_seconds() * 1000)
        result.append({"ts_ms": elapsed_ms, "kw": round(float(val), 3),
                        "label": fmt_elapsed(elapsed_ms)})
    return result, None

def load_jobs(parquet_bytes, t0_utc=None, t1_utc=None):
    """Return job records from job_table parquet."""
    try:
        df = robust_read_parquet(parquet_bytes)
    except Exception as e:
        return [], str(e)
    if "metric" in df.columns:
        df = df[df["metric"] == "job_info_marconi100"]
    if t0_utc:
        t0 = pd.Timestamp(t0_utc, tz="UTC")
        if "start_time" in df.columns:
            df["start_time"] = pd.to_datetime(df["start_time"], utc=True, errors="coerce")
            df = df[df["start_time"] >= t0]
    if t1_utc:
        t1 = pd.Timestamp(t1_utc, tz="UTC")
        if "end_time" in df.columns:
            df["end_time"] = pd.to_datetime(df["end_time"], utc=True, errors="coerce")
            df = df[df["end_time"] < t1]
    records = []
    for _, row in df.iterrows():
        rec = {}
        for col in ["job_id", "start_time", "end_time", "num_nodes", "num_cpus",
                    "num_tasks", "partition", "qos", "user_id", "job_state"]:
            if col in df.columns:
                val = row[col]
                rec[col] = None if pd.isna(val) else str(val) if col in ("job_id", "partition", "qos", "user_id", "job_state") else val
        records.append(rec)
    return records, None

def load_slurm_util(parquet_bytes, t0_utc=None, t1_utc=None):
    """Return cluster_cpu_util and s21.cluster_gpu_util time series."""
    try:
        df = robust_read_parquet(parquet_bytes)
    except Exception as e:
        return {}, str(e)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "value_numeric"])
    if t0_utc:
        df = df[df["timestamp"] >= pd.Timestamp(t0_utc, tz="UTC")]
    if t1_utc:
        df = df[df["timestamp"] < pd.Timestamp(t1_utc, tz="UTC")]
    metrics = ["cluster_cpu_util", "s21.cluster_gpu_util", "s21.cluster_mem_util",
               "total_nodes_alloc", "s21.jobs.tot_gpus"]
    result = {}
    for m in metrics:
        sub = df[df["metric"] == m].copy()
        if sub.empty:
            continue
        sub = sub.set_index("timestamp").sort_index()
        resampled = sub["value_numeric"].resample("1min").median().dropna()
        if resampled.empty:
            continue
        ref_ts = resampled.index[0]
        result[m] = [{"ts_ms": int((ts - ref_ts).total_seconds() * 1000),
                       "val": round(float(v), 4)}
                     for ts, v in resampled.items()]
    return result, None

def load_ganglia_power(parquet_bytes, t0_utc=None, t1_utc=None):
    """Return aggregated GPU power from ganglia_pub (sum across nodes)."""
    try:
        df = robust_read_parquet(parquet_bytes)
    except Exception as e:
        return None, str(e)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "value_numeric"])
    # Keep GPU power metrics
    gpu_mask = df["metric"].str.contains("power_usage", case=False, na=False)
    sub = df[gpu_mask].copy()
    if sub.empty:
        return None, "No GPU power metrics found"
    if t0_utc:
        sub = sub[sub["timestamp"] >= pd.Timestamp(t0_utc, tz="UTC")]
    if t1_utc:
        sub = sub[sub["timestamp"] < pd.Timestamp(t1_utc, tz="UTC")]
    if sub.empty:
        return None, "No GPU power in time window"
    # Sum across nodes per minute, convert W→kW
    sub = sub.set_index("timestamp")
    resampled = (sub["value_numeric"].resample("1min").sum() / 1000).dropna()
    ref_ts = resampled.index[0]
    result = [{"ts_ms": int((ts - ref_ts).total_seconds() * 1000),
               "kw": round(float(v), 3)}
              for ts, v in resampled.items()]
    return result, None

# ── Validation endpoint ───────────────────────────────────────────────────────
@app.route("/api/validate", methods=["POST"])
def api_validate():
    """
    Accepts multipart/form-data with optional Parquet file uploads:
      - logics_file:  logics_pub parquet  (for real Tot_ict)
      - jobs_file:    job_table parquet   (for job-derived tasks)
      - slurm_file:   slurm_pub parquet   (for utilisation overlay)
      - ganglia_file: ganglia_pub parquet (for per-node GPU power)
    Plus JSON body fields:
      - inventory, infra, hw_type, t0, t1
    """
    hw_type   = request.form.get("hw_type", "V100")
    t0_utc    = request.form.get("t0") or None
    t1_utc    = request.form.get("t1") or None
    inventory_str = request.form.get("inventory", "{}")
    infra_str     = request.form.get("infra", "{}")
    try:
        inventory = json.loads(inventory_str)
        # If all zeros (nothing set by user), fall back to config defaults
        if not any(v > 0 for v in inventory.values()):
            inventory = dict(CFG["defaults"]["inventory"])
    except Exception:
        inventory = dict(CFG["defaults"]["inventory"])
    try:
        infra = json.loads(infra_str)
    except Exception:
        infra = {}

    result = {"real": None, "simulated": None, "slurm": {}, "ganglia": None,
              "stats": {}, "errors": [], "meta": {}}

    # ── 1. Real IT power (logics_pub → Tot_ict) ───────────────────────────────
    logics_file = request.files.get("logics_file")
    if logics_file:
        real_series, err = load_logics_ict(logics_file.read(), t0_utc, t1_utc)
        if err:
            result["errors"].append(f"logics: {err}")
        else:
            result["real"] = real_series

    # ── 2. SLURM utilisation overlay ─────────────────────────────────────────
    slurm_file = request.files.get("slurm_file")
    if slurm_file:
        slurm_data, err = load_slurm_util(slurm_file.read(), t0_utc, t1_utc)
        if err:
            result["errors"].append(f"slurm: {err}")
        else:
            result["slurm"] = slurm_data

    # ── 3. Ganglia GPU power overlay ─────────────────────────────────────────
    ganglia_file = request.files.get("ganglia_file")
    if ganglia_file:
        ganglia_data, err = load_ganglia_power(ganglia_file.read(), t0_utc, t1_utc)
        if err:
            result["errors"].append(f"ganglia: {err}")
        else:
            result["ganglia"] = ganglia_data

    # ── 4. Simulation driven by job_table ─────────────────────────────────────
    jobs_file = request.files.get("jobs_file")
    tasks = []
    if jobs_file:
        jobs, err = load_jobs(jobs_file.read(), t0_utc, t1_utc)
        if err:
            result["errors"].append(f"jobs: {err}")
        else:
            starts = [iso_to_epoch_ms(j.get("start_time")) for j in jobs]
            starts = [s for s in starts if s is not None]
            if starts:
                ref_ms = min(starts)
                for i, job in enumerate(jobs):
                    task = job_to_task(job, i, hw_type, ref_ms)
                    if task:
                        tasks.append(task)
                result["meta"]["job_count"]  = len(jobs)
                result["meta"]["task_count"] = len(tasks)
                stage_counts = {}
                for t in tasks:
                    stage_counts[t["stage"]] = stage_counts.get(t["stage"], 0) + 1
                result["meta"]["stage_counts"] = stage_counts

    # ── 5. Run simulation ─────────────────────────────────────────────────────
    if tasks:
        infra_kw = sum(infra.get(k, 0) for k in ("network_kw", "storage_kw", "misc_kw"))
        dur_ms   = max(t["end_ms"] for t in tasks) + 60_000
        rngs     = {s: random.Random(42) for s in ("training", "fine_tuning", "inference")}
        macro, stats, _ = compute_chart_data(tasks, inventory, infra_kw, 0, dur_ms, rngs)
        result["simulated"] = macro
        result["stats"]     = stats
    elif result["real"]:
        # No jobs file: run a synthetic simulation over the same duration as real data
        dur_ms   = result["real"][-1]["ts_ms"] + 60_000
        infra_kw = sum(infra.get(k, 0) for k in ("network_kw", "storage_kw", "misc_kw"))
        # Build synthetic tasks from SLURM node allocation if available
        if "total_nodes_alloc" in result["slurm"]:
            node_series = result["slurm"]["total_nodes_alloc"]
            # Create a single synthetic task spanning the window
            avg_nodes = sum(p["val"] for p in node_series) / len(node_series)
            tasks = [{
                "id": "synth-1",
                "label": "Synthetic (SLURM nodes)",
                "hardware_type": hw_type,
                "stage": "training",
                "num_devices": max(1, round(avg_nodes)),
                "start_ms": 0,
                "end_ms": dur_ms,
                "requests_per_hour": None,
            }]
        else:
            n_devs = inventory.get(hw_type, 0)
            if n_devs == 0:
                n_devs = CFG["defaults"]["inventory"].get(hw_type, 128)
            tasks = [{
                "id": "synth-1",
                "label": "Synthetic baseline",
                "hardware_type": hw_type,
                "stage": "training",
                "num_devices": n_devs,
                "start_ms": 0,
                "end_ms": dur_ms,
                "requests_per_hour": None,
            }]
        rngs = {s: random.Random(42) for s in ("training", "fine_tuning", "inference")}
        macro, stats, _ = compute_chart_data(tasks, inventory, infra_kw, 0, dur_ms, rngs)
        result["simulated"] = macro
        result["stats"]     = stats
        result["meta"]["synthetic"] = True

    return jsonify(result)

# ── Config endpoint ───────────────────────────────────────────────────────────
@app.route("/api/config")
def api_config():
    return jsonify(CFG)

# ── Serve static ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "validate.html")

if __name__ == "__main__":
    print("IT Power Validation → http://localhost:5050")
    app.run(debug=True, port=5050, host="0.0.0.0")
