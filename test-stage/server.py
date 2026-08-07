"""
IT Power Simulation — Python backend (Flask)
Formula: P(t) = Σk Nk·Pk_max [ηk(t)·Σm(um·αm(t)) + (1−ηk(t))·ρk]
Run: python server.py  →  http://localhost:5000
"""
import json, math, random, os, shutil, uuid
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import pyarrow.parquet as pq
from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__, static_folder=BASE_DIR)

with open(os.path.join(BASE_DIR, "config.json")) as f:
    CFG = json.load(f)

HW      = CFG["hardware"]
S_DEF   = CFG["stage_defaults"]
S_PHY   = CFG["stage_physics"]
SIM_CFG = CFG["simulation"]
CALC_STEP_MS  = SIM_CFG["calc_step_ms"]
MICRO_MAX_PTS = SIM_CFG["micro_max_pts"]
MACRO_MAX_PTS = SIM_CFG["macro_max_pts"]
MICRO_STEP_MS = {"training": 1000, "fine_tuning": 1000, "inference": 1000}

def clamp(x, lo, hi): return max(lo, min(hi, x))
def gauss(mean, std, rng):  return rng.gauss(mean, std)

def poisson_draw(lam, rng):
    if lam <= 0: return 0
    if lam < 30:
        L, k, p = math.exp(-lam), 0, 1.0
        while True:
            k += 1; p *= rng.random()
            if p <= L: return k - 1
    return max(0, round(gauss(lam, math.sqrt(lam), rng)))

def fmt_elapsed(ms):
    if ms == 0:     return "0s"
    if ms < 1000:   return f"{int(ms)}ms"
    s = ms / 1000
    if s < 10:      return f"{s:.1f}s"
    if s < 60:      return f"{s:.0f}s"
    if s < 3600:    return f"{s/60:.0f}min"
    return f"{s/3600:.1f}h"

def fmt_window(ms):
    s = ms / 1000
    if s < 60:   return f"{s:.0f}s"
    if s < 3600: return f"{s/60:.0f}min"
    return f"{s/3600:.1f}h"

def generate_inference_schedule(task, N, rng):
    ph    = S_PHY["inference"]
    rph = task.get("requests_per_hour")
    lam = ph["default_lambda"] if rph is None else rph
    D     = max(1, (task["end_ms"] - task["start_ms"]) / 1000)
    tau   = min(ph["service_s"], D)
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
    ph    = S_PHY[stage]
    def_  = S_DEF[stage]
    rng   = rng or random.Random()
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

from datetime import datetime, timezone   # add to the imports at the top

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

@app.route("/api/replay", methods=["POST"])
def api_replay():
    b    = request.get_json(force=True)
    jobs = b.get("jobs", [])
    if not jobs:
        return jsonify({"error": "No jobs provided"}), 400

    hw_type   = b.get("hardware_type", CFG.get("replay", {}).get("default_hardware_type", "V100"))
    inventory = b.get("inventory", CFG.get("m100_preset", {}).get("inventory", CFG["defaults"]["inventory"]))
    infra     = b.get("infra", {})

    if hw_type not in HW:
        return jsonify({"error": f"Unknown hardware_type '{hw_type}'"}), 400

    starts = [iso_to_epoch_ms(j.get("start_time")) for j in jobs]
    starts = [s for s in starts if s is not None]
    if not starts:
        return jsonify({"error": "No valid start_time in any job"}), 400
    ref_ms = min(starts)

    tasks, skipped = [], 0
    for i, job in enumerate(jobs):
        task = job_to_task(job, i, hw_type, ref_ms)
        if task is None:
            skipped += 1
        else:
            tasks.append(task)

    if not tasks:
        return jsonify({"error": "No usable job records"}), 400

    infra_kw = sum(infra.get(k, 0) for k in ("network_kw", "storage_kw", "misc_kw"))
    dur_ms   = b.get("sim", {}).get("duration_ms", max(t["end_ms"] for t in tasks))

    rngs = {s: random.Random() for s in ("training", "fine_tuning", "inference")}
    macro, stats, schedules = compute_chart_data(tasks, inventory, infra_kw, 0, dur_ms, rngs)
    micro = {s: generate_micro_trace(s, tasks, inventory, schedules, dur_ms, rngs[s])
             for s in ("training", "fine_tuning", "inference")}

    stage_counts = {}
    for t in tasks:
        stage_counts[t["stage"]] = stage_counts.get(t["stage"], 0) + 1

    return jsonify({
        "macro": macro, "micro": micro, "stats": stats,
        "replay_meta": {
            "job_count":     len(jobs),
            "task_count":    len(tasks),
            "skipped_count": skipped,
            "stage_counts":  stage_counts,
            "hw_type":       hw_type,
            "ref_time_ms":   ref_ms,
            "duration_ms":   dur_ms,
        }
    })

def build_schedules(tasks, inventory, rng):
    schedules = {}
    for t in tasks:
        if t["stage"] != "inference": continue
        Nk   = inventory.get(t["hardware_type"], 0)
        if Nk == 0: continue
        same = [x for x in tasks if x["hardware_type"] == t["hardware_type"]]
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
    rows      = []
    for ts in range(int(start_ms), int(end_ms) + 1, int(step_ms)):
        mkw = compute_macro_at(ts, tasks, inventory, schedules, rngs)
        rows.append({"label": fmt_elapsed(ts - start_ms),
                     "base_kw":  round(infra_kw, 3),
                     "tasks_kw": round(mkw, 3),
                     "total_kw": round(mkw + infra_kw, 3)})
    totals = [r["total_kw"] for r in rows]
    fleet  = [r["tasks_kw"] for r in rows]
    stats  = {"peak_kw":      round(max(totals), 2) if totals else 0,
              "avg_kw":       round(sum(totals)/len(totals), 2) if totals else 0,
              "task_count":   len(tasks),
              "peak_fleet_kw":round(max(fleet), 2) if fleet else 0}
    return rows, stats, schedules

def generate_micro_trace(stage, tasks, inventory, schedules, sim_dur_ms, rng):
    stage_tasks = [t for t in tasks if t["stage"] == stage]
    ref         = stage_tasks[0] if stage_tasks else None
    hw_key      = ref["hardware_type"] if ref else "H100"
    hw          = HW[hw_key]
    n_dev       = ref.get("num_devices", 64) if ref else 64
    if stage_tasks:
        window_ms = max(sim_dur_ms, MICRO_STEP_MS[stage] * 10)
    else:
        window_ms = max(sim_dur_ms, 120 * 60 * 1000)
    native_step = MICRO_STEP_MS[stage]
    n_native    = math.ceil(window_ms / native_step)
    skip        = max(1, math.ceil(n_native / MICRO_MAX_PTS))
    step_ms     = native_step * skip
    N           = max(1, math.ceil(window_ms / step_ms)) + 1
    ph          = S_PHY[stage]
    labels, data = [], []
    task_rngs   = {}
    for i in range(N):
        el_ms = min(i * step_ms, window_ms)
        u     = 0.0
        if stage_tasks:
            active = [t for t in stage_tasks if t["start_ms"] <= el_ms < t["end_ms"]]
            if active:
                t_ref         = active[0]
                sched         = schedules.get(t_ref["id"]) if stage == "inference" else schedules.get(ref["id"]) if ref else None
                N2D           = t_ref.get("_N2D", 1)
                local_elapsed = el_ms - t_ref["start_ms"]
                if t_ref["id"] not in task_rngs:
                    task_rngs[t_ref["id"]] = random.Random(hash(t_ref["id"]) & 0x7fffffff)
                task_rng = task_rngs[t_ref["id"]]
                u = stochastic_util(stage, local_elapsed, t_ref, sched, N2D, task_rng, hw["rho"])
        pw_kw = round((hw["p_max"] * max(u, hw["rho"]) * n_dev) / 1000, 3)
        labels.append(fmt_elapsed(el_ms))
        data.append(pw_kw)
    n    = len(stage_tasks)
    pill = f"{fmt_window(window_ms)} window · {fmt_elapsed(step_ms)}/pt · {n} task{'s' if n!=1 else ''}"
    if stage == "training":
        ck_str = fmt_window(ph["checkpoint_ms"])
        ck_vis = "checkpoint visible" if step_ms < ph["checkpoint_ms"] else "no checkpoint in window"
        axis   = f"Elapsed time — compute plateau (u≈{ph['u_plateau']}) · dips every {ck_str} · {ck_vis}"
    elif stage == "fine_tuning":
        axis   = f"Elapsed time — compute plateau (u≈{ph['u_plateau']}) · eval dips every {fmt_window(ph['eval_period_ms'])}"
    else:
        lam  = ref.get("requests_per_hour", ph["default_lambda"]) if ref else ph["default_lambda"]
        axis = f"Elapsed time — Poisson burst/idle · λ={lam} req/hr · τ={ph['service_s']}s service"
    return {"labels": labels, "data": data, "timescale": pill, "axis_label": axis}



# ─────────────────────────────────────────────────────────────────────────────
# Real M100 dataset validation endpoint
# Upload a folder containing Hive-style partitions, for example:
# data/dataset=main_datasets/year_month=22-03/plugin=logics_pub/part-000.parquet
# The endpoint extracts measured IT power from logics_pub and creates model tasks
# from job_table, then returns both series on the same time axis.
# ─────────────────────────────────────────────────────────────────────────────
UPLOAD_ROOT = os.path.join(BASE_DIR, "_uploaded_m100")
os.makedirs(UPLOAD_ROOT, exist_ok=True)

def _safe_rel_path(filename):
    parts = []
    for part in str(filename).replace("\\", "/").split("/"):
        part = part.strip()
        if not part or part in (".", ".."):
            continue
        safe = "".join(ch for ch in part if ch.isalnum() or ch in "._=-")
        if safe:
            parts.append(safe)
    return os.path.join(*parts) if parts else f"file-{uuid.uuid4().hex}"

def _find_part_file(root_dir, year_month, plugin):
    root = Path(root_dir)
    # Prefer exact Hive-style match anywhere under uploaded tree.
    matches = list(root.rglob(f"year_month={year_month}/plugin={plugin}/part-000.parquet"))
    if matches:
        return matches[0]
    # Fallback: any plugin folder with matching part file.
    matches = [p for p in root.rglob("part-000.parquet") if f"plugin={plugin}" in str(p) and f"year_month={year_month}" in str(p)]
    return matches[0] if matches else None



def _parquet_columns(path):
    """Return physical top-level column names without decoding full data pages."""
    pf = pq.ParquetFile(path)
    return pf.schema_arrow.names


def _read_parquet_safe(path, wanted_columns):
    """
    Read only required flat columns from one physical Parquet file.

    Important: do NOT use pandas.read_parquet() or pyarrow.read_table() here.
    When a file lives inside Hive-style folders such as year_month=22-03/plugin=logics_pub,
    those readers can infer partition columns and try to merge them with physical columns.
    Some M100 files also contain year_month/plugin columns inside the file, which can produce:
        Field year_month has incompatible types: large_string vs dictionary<...>

    ParquetFile.read() reads the single physical file only and avoids partition schema merging.
    It also avoids decoding unused nested columns that can trigger repetition-level errors.
    """
    pf = pq.ParquetFile(path)
    available = set(pf.schema_arrow.names)
    cols = [c for c in wanted_columns if c in available]
    missing_required = [c for c in wanted_columns[:3] if c not in available]
    if missing_required:
        raise ValueError(f"Missing required parquet columns: {missing_required}. Available: {sorted(available)}")
    table = pf.read(columns=cols, use_threads=False)
    return table.to_pandas()

def _read_logics_power(logics_path, metric):
    df = _read_parquet_safe(logics_path, ["timestamp", "metric", "value_numeric", "panel", "device"])
    if "timestamp" not in df.columns or "metric" not in df.columns or "value_numeric" not in df.columns:
        raise ValueError("logics_pub must contain timestamp, metric, and value_numeric columns")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])

    if metric == "auto":
        available = set(df["metric"].astype(str).unique())
        for candidate in ("Tot_ict", "pit", "pt"):
            if candidate in available:
                metric = candidate
                break
        else:
            raise ValueError("No supported measured IT power metric found. Expected Tot_ict, pit, or pt.")

    power = df[df["metric"].astype(str) == metric].copy()
    if power.empty:
        raise ValueError(f"Metric '{metric}' not found in logics_pub")

    # Facility totals often use panel=generals, device=pue.
    if "panel" in power.columns and "device" in power.columns:
        preferred = power[
            (power["panel"].astype(str).str.lower() == "generals") &
            (power["device"].astype(str).str.lower() == "pue")
        ].copy()
        if not preferred.empty:
            power = preferred

    ts = power.set_index("timestamp")["value_numeric"].sort_index()
    # Resample to a stable display cadence. Native data is irregular.
    ts = ts.resample("5min").mean().dropna()

    # Units guard. README says Tot_* usually kW, pit/pt may be W. Convert likely watts to kW.
    unit_note = "kW"
    if len(ts) and ts.median() > 10000:
        ts = ts / 1000.0
        unit_note = "converted W to kW"
    return ts, metric, unit_note

def _read_jobs_as_tasks(job_path, hw_type, ref_ms, limit_jobs=5000):
    # Do not read the whole job_table. Some parquet builds fail on nested/unused columns.
    jobs = _read_parquet_safe(job_path, ["job_id", "start_time", "end_time", "num_nodes", "num_cpus", "num_tasks", "partition", "qos", "job_state"])
    required = {"start_time", "end_time"}
    if not required.issubset(set(jobs.columns)):
        raise ValueError("job_table must contain start_time and end_time columns")

    jobs["start_time"] = pd.to_datetime(jobs["start_time"], utc=True, errors="coerce")
    jobs["end_time"] = pd.to_datetime(jobs["end_time"], utc=True, errors="coerce")
    jobs = jobs.dropna(subset=["start_time", "end_time"])
    jobs = jobs[jobs["end_time"] > jobs["start_time"]].copy()
    jobs = jobs.sort_values("start_time").head(limit_jobs)

    out = []
    for i, row in jobs.iterrows():
        s_ms = int(row["start_time"].timestamp() * 1000)
        e_ms = int(row["end_time"].timestamp() * 1000)
        elapsed_s = s_ms - ref_ms
        elapsed_e = e_ms - ref_ms
        if elapsed_e <= 0:
            continue
        num_nodes = int(row.get("num_nodes") or 1)
        # Cleaned README lists num_nodes, num_cpus, num_tasks. No guaranteed num_gpus.
        num_gpus = int(row.get("num_gpus") or 0) if "num_gpus" in jobs.columns else 0
        num_devs = max(1, num_gpus if num_gpus > 0 else num_nodes)
        job_id = row.get("job_id", f"job-{len(out)}")
        out.append({
            "id": str(job_id),
            "label": str(job_id),
            "name": str(job_id),
            "hardware_type": hw_type,
            "stage": classify_stage(elapsed_s, elapsed_e, num_nodes, num_gpus, hw_type),
            "start_ms": max(0, elapsed_s),
            "end_ms": max(1, elapsed_e),
            "num_devices": num_devs,
            "requests_per_hour": None,
        })
    return out, len(jobs)

def _model_series_for_measured_index(tasks, inventory, infra_kw, measured_index):
    if not measured_index.size:
        return []
    ref_ms = int(measured_index[0].timestamp() * 1000)
    rngs = {s: random.Random(1234 + i) for i, s in enumerate(("training", "fine_tuning", "inference"))}
    schedules = build_schedules(tasks, inventory, rngs["inference"])
    rows = []
    for ts in measured_index:
        elapsed = int(ts.timestamp() * 1000) - ref_ms
        model_kw = compute_macro_at(elapsed, tasks, inventory, schedules, rngs) + infra_kw
        rows.append(round(model_kw, 3))
    return rows

@app.route("/api/validate-upload", methods=["POST"])
def api_validate_upload():
    if "files" not in request.files:
        return jsonify({"error": "No files uploaded. Select the dataset folder."}), 400

    year_month = request.form.get("year_month", "22-03")
    metric = request.form.get("metric", "auto")
    hw_type = request.form.get("hardware_type", CFG.get("replay", {}).get("default_hardware_type", "V100"))
    if hw_type not in HW:
        return jsonify({"error": f"Unknown hardware_type '{hw_type}'"}), 400

    try:
        inventory = json.loads(request.form.get("inventory", "{}")) or CFG["defaults"].get("inventory", {})
        infra = json.loads(request.form.get("infra", "{}")) or {}
    except json.JSONDecodeError:
        return jsonify({"error": "inventory and infra must be valid JSON"}), 400

    infra_kw = sum(float(infra.get(k, 0) or 0) for k in ("network_kw", "storage_kw", "misc_kw"))

    session_dir = os.path.join(UPLOAD_ROOT, uuid.uuid4().hex)
    os.makedirs(session_dir, exist_ok=True)
    try:
        count = 0
        for f in request.files.getlist("files"):
            rel = _safe_rel_path(f.filename)
            dst = os.path.join(session_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            f.save(dst)
            count += 1

        logics_path = _find_part_file(session_dir, year_month, "logics_pub")
        job_path = _find_part_file(session_dir, year_month, "job_table")
        if logics_path is None:
            return jsonify({"error": f"Missing logics_pub for year_month={year_month}"}), 400
        if job_path is None:
            return jsonify({"error": f"Missing job_table for year_month={year_month}"}), 400

        measured_ts, used_metric, unit_note = _read_logics_power(logics_path, metric)
        if measured_ts.empty:
            return jsonify({"error": "Measured IT power series is empty after filtering"}), 400

        ref_ms = int(measured_ts.index[0].timestamp() * 1000)
        tasks_from_jobs, usable_jobs = _read_jobs_as_tasks(job_path, hw_type, ref_ms)
        if not tasks_from_jobs:
            return jsonify({"error": "No usable job records found in job_table"}), 400

        model_values = _model_series_for_measured_index(tasks_from_jobs, inventory, infra_kw, measured_ts.index)
        labels = [t.strftime("%Y-%m-%d %H:%M") for t in measured_ts.index]

        measured_vals = [round(float(v), 3) for v in measured_ts.values]
        residuals = [round(m - p, 3) for m, p in zip(measured_vals, model_values)]
        rmse = math.sqrt(sum(e * e for e in residuals) / max(1, len(residuals)))
        mae = sum(abs(e) for e in residuals) / max(1, len(residuals))

        stage_counts = {}
        for t in tasks_from_jobs:
            stage_counts[t["stage"]] = stage_counts.get(t["stage"], 0) + 1

        return jsonify({
            "labels": labels,
            "measured_kw": measured_vals,
            "model_kw": model_values,
            "residual_kw": residuals,
            "meta": {
                "uploaded_files": count,
                "year_month": year_month,
                "metric": used_metric,
                "unit_note": unit_note,
                "logics_path": str(logics_path.relative_to(session_dir)),
                "job_table_path": str(job_path.relative_to(session_dir)),
                "points": len(labels),
                "task_count": len(tasks_from_jobs),
                "usable_jobs": usable_jobs,
                "stage_counts": stage_counts,
                "hw_type": hw_type,
                "rmse_kw": round(rmse, 3),
                "mae_kw": round(mae, 3),
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Keep server clean after processing. Remove this line if you want to inspect uploads.
        shutil.rmtree(session_dir, ignore_errors=True)

@app.route("/api/config")
def api_config():
    return jsonify(CFG)

@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    b         = request.get_json(force=True)
    tasks_in  = b.get("tasks", [])
    inventory = b.get("inventory", {})
    infra     = b.get("infra", {})
    sim       = b.get("sim", {})
    infra_kw  = sum(infra.get(k, 0) for k in ("network_kw","storage_kw","misc_kw"))
    start_ms  = sim.get("start_ms", 0)
    dur_ms    = sim.get("duration_ms", 7200000)
    rngs = {s: random.Random() for s in ("training","fine_tuning","inference")}
    macro, stats, schedules = compute_chart_data(tasks_in, inventory, infra_kw, start_ms, dur_ms, rngs)
    micro = {s: generate_micro_trace(s, tasks_in, inventory, schedules, dur_ms, rngs[s])
             for s in ("training","fine_tuning","inference")}
    return jsonify({"macro": macro, "micro": micro, "stats": stats})

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")

if __name__ == "__main__":
    print("IT Power Simulation → http://localhost:5000")
    app.run(debug=True, port=5000)
