"""
validation_core.py — measured-vs-simulated IT power for the M100/ExaData cleaned bundle.

Measured IT power  : logics_pub · metric=Tot_ict · panel=generals · device=pue   (kW)
Simulated IT power : job_table (job_info_marconi100) -> server.py stage-aware model
Both are evaluated on the SAME timestamp grid (the measured cadence) so they align
without resampling, then compared with MAPE (power) and energy error.

This module is import-safe: it reuses the *actual* functions in server.py, so the
simulated curve is produced by your model, not a reimplementation.
"""
import math, random
import numpy as np
import pandas as pd
import server  # uses server.HW, server.job_to_task, server.build_schedules, server.compute_macro_at

# M100 hardware facts (from the project notes / cleaned-data guide)
GPUS_PER_NODE   = 4            # 4x NVIDIA V100 per node
N_NODES_M100    = 980
DEFAULT_INV     = {"V100": N_NODES_M100 * GPUS_PER_NODE}   # 3920 V100 GPUs
HW_TYPE         = "V100"

# ---------------------------------------------------------------------------
# 0. Path discovery — robust to dataset=main vs dataset=main_datasets, and to
#    whatever the part file is actually called (part-000.parquet, part-0.parquet…)
# ---------------------------------------------------------------------------
import glob, os

def find_part(parquet_root, year_month, plugin):
    """
    Locate the parquet file for (year_month, plugin) under any dataset=* partition.
    Works for layouts like:
        data/dataset=main_datasets/year_month=22-03/plugin=logics_pub/part-000.parquet
    """
    pat = os.path.join(parquet_root, "dataset=*",
                       f"year_month={year_month}", f"plugin={plugin}", "part-*.parquet")
    hits = sorted(glob.glob(pat))
    if not hits:
        raise FileNotFoundError(
            f"No parquet for plugin={plugin}, year_month={year_month} under {parquet_root!r}.\n"
            f"  searched: {pat}\n"
            f"  (check the root points at the 'data' folder and the month exists)")
    return hits[0]

def clean_month(s):
    """Accept '22-03', 'year_month=22-03', or '.../year_month=22-03/' -> '22-03'."""
    s = str(s).strip().strip("/").split("/")[-1]
    return s.split("=", 1)[1] if s.startswith("year_month=") else s

def clean_root(s):
    """If the path points inside the partition tree, climb up to the folder that
    contains dataset=* . Accepts the data/ folder, the dataset=* folder, or a
    year_month=* / plugin=* folder."""
    p = os.path.abspath(str(s).rstrip("/"))
    while p and os.path.basename(p).split("=", 1)[0] in ("dataset", "year_month", "plugin"):
        p = os.path.dirname(p)
    return p

def autodetect_root(start="."):
    """Find a folder containing dataset=*/year_month=*/ by checking common spots."""
    cands = [start, "./data", "../data", "../../data", "..",
             os.path.join(start, "data")]
    seen = set()
    for c in cands:
        c = os.path.abspath(c)
        if c in seen:
            continue
        seen.add(c)
        if glob.glob(os.path.join(c, "dataset=*", "year_month=*")):
            return c
    return None

def available_months(parquet_root):
    """Sorted unique year_month strings present under the root."""
    hits = glob.glob(os.path.join(parquet_root, "dataset=*", "year_month=*"))
    months = {os.path.basename(h).split("=", 1)[1] for h in hits}
    return sorted(months)

def plugins_for(parquet_root, year_month):
    """Sorted plugin names available for a given month."""
    hits = glob.glob(os.path.join(parquet_root, "dataset=*",
                                  f"year_month={year_month}", "plugin=*"))
    return sorted(os.path.basename(h).split("=", 1)[1] for h in hits)

# ---------------------------------------------------------------------------
# 1. Loaders — exactly the columns/filters described in the cleaned-data guide
# ---------------------------------------------------------------------------
def measured_it_power(parquet_root, year_month, t0=None, t1=None):
    """Return DataFrame[timestamp(UTC), kw] of measured total ICT power (Tot_ict)."""
    df = pd.read_parquet(find_part(parquet_root, year_month, "logics_pub"))
    m = (df["metric"] == "Tot_ict") \
        & (df["panel"].astype(str) == "generals") \
        & (df["device"].astype(str) == "pue")
    out = df.loc[m, ["timestamp", "value_numeric"]].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.rename(columns={"value_numeric": "kw"}).sort_values("timestamp")
    if t0 is not None: out = out[out["timestamp"] >= pd.Timestamp(t0, tz="UTC")]
    if t1 is not None: out = out[out["timestamp"] <  pd.Timestamp(t1, tz="UTC")]
    out = out[out["kw"] > 0].reset_index(drop=True)
    return out

def load_jobs(parquet_root, year_month, t0=None, t1=None):
    """Return list[job dict] in the shape server.job_to_task expects."""
    df = pd.read_parquet(find_part(parquet_root, year_month, "job_table"))
    if "metric" in df.columns:
        df = df[df["metric"] == "job_info_marconi100"]
    df = df.copy()
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True, errors="coerce")
    df["end_time"]   = pd.to_datetime(df["end_time"],   utc=True, errors="coerce")
    df = df.dropna(subset=["start_time", "end_time"])
    df = df[df["end_time"] > df["start_time"]]
    if t0 is not None and t1 is not None:   # keep jobs overlapping the window
        a, b = pd.Timestamp(t0, tz="UTC"), pd.Timestamp(t1, tz="UTC")
        df = df[(df["start_time"] < b) & (df["end_time"] > a)]
    return _jobs_dataframe_to_records(df)

def _jobs_dataframe_to_records(df):
    jobs = []
    for _, r in df.iterrows():
        n_nodes = int(r.get("num_nodes") or 1)
        jobs.append({
            "job_id":     str(r.get("job_id", "")),
            "start_time": r["start_time"].isoformat(),
            "end_time":   r["end_time"].isoformat(),
            "num_nodes":  n_nodes,
            "num_gpus":   n_nodes * GPUS_PER_NODE,   # job_table has no GPU count -> derive
        })
    return jobs

# ---------------------------------------------------------------------------
# 2. Drive the real server.py model on a given timestamp grid
# ---------------------------------------------------------------------------
def simulate_on_grid(jobs, timestamps_utc, inventory=None, hw_type=HW_TYPE, seed=0):
    """
    timestamps_utc : pd.DatetimeIndex / Series (UTC) — typically the measured cadence.
    Returns np.array of simulated IT power (kW) at each timestamp, from server.py.
    """
    inventory = inventory or DEFAULT_INV
    ts = pd.to_datetime(pd.Series(timestamps_utc), utc=True).reset_index(drop=True)
    ref_ms = int(ts.iloc[0].timestamp() * 1000)

    # jobs -> model tasks using server.py's own converter (handles stage classification)
    tasks = []
    for i, j in enumerate(jobs):
        t = server.job_to_task(j, i, hw_type, ref_ms)
        if t is not None:
            tasks.append(t)

    rngs = {s: random.Random(seed) for s in ("training", "fine_tuning", "inference")}
    schedules = server.build_schedules(tasks, inventory, rngs["inference"])

    sim = np.empty(len(ts), dtype=float)
    for i, t_i in enumerate(ts):
        ts_ms = int(t_i.timestamp() * 1000) - ref_ms
        sim[i] = server.compute_macro_at(ts_ms, tasks, inventory, schedules, rngs)
    return sim, tasks

# ---------------------------------------------------------------------------
# 3. Metrics — the project's validation targets
# ---------------------------------------------------------------------------
def metrics(meas_kw, sim_kw, timestamps_utc):
    meas = np.asarray(meas_kw, float)
    sim  = np.asarray(sim_kw, float)
    ok   = meas > 0
    mape = float(np.mean(np.abs(sim[ok] - meas[ok]) / meas[ok]) * 100)
    # energy via trapezoid over real time (hours)
    t_h  = (pd.to_datetime(pd.Series(timestamps_utc), utc=True).astype("int64").to_numpy()
            / 1e9 / 3600.0)
    trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))  # numpy>=2.0 vs older
    e_meas = float(trapz(meas, t_h))
    e_sim  = float(trapz(sim,  t_h))
    energy_err = abs(e_sim - e_meas) / e_meas * 100 if e_meas else float("nan")
    return {"mape_pct": mape, "energy_err_pct": energy_err,
            "e_meas_kwh": e_meas, "e_sim_kwh": e_sim,
            "peak_meas": float(meas.max()), "peak_sim": float(sim.max()),
            "mean_meas": float(meas.mean()), "mean_sim": float(sim.mean())}

# ---------------------------------------------------------------------------
# 4. Plot — measured vs simulated overlay + residual, with legend
# ---------------------------------------------------------------------------
def plot_overlay(timestamps_utc, meas_kw, sim_kw, m, outpath,
                 title="M100 IT Power — Measured vs Simulated", subtitle=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    ts   = pd.to_datetime(pd.Series(timestamps_utc), utc=True)
    meas = np.asarray(meas_kw, float)
    sim  = np.asarray(sim_kw, float)
    resid_pct = np.where(meas > 0, (sim - meas) / meas * 100, np.nan)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.edgecolor": "#c5c1b8", "axes.linewidth": 0.8})
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
    fig.patch.set_facecolor("#f4f3ef")
    for ax in (ax1, ax2): ax.set_facecolor("#ffffff")

    ax1.plot(ts, meas, color="#15803d", lw=1.8, label="Measured IT power — Tot_ict (kW)")
    ax1.plot(ts, sim,  color="#1d4ed8", lw=1.6, ls="--",
             label="Simulated IT power — stage-aware model (kW)")
    ax1.fill_between(ts, meas, sim, color="#94a3b8", alpha=0.18,
                     label="Difference (residual)")
    ax1.set_ylabel("IT power (kW)")
    ax1.set_ylim(bottom=0)
    ax1.grid(True, color="#ece9e3", lw=0.7)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.95,
               edgecolor="#dddad3", fontsize=9)

    box = (f"MAPE {m['mape_pct']:.1f}%   (target < 15%)\n"
           f"Energy err {m['energy_err_pct']:.1f}%   (target < 10%)\n"
           f"peak meas {m['peak_meas']:.0f} / sim {m['peak_sim']:.0f} kW")
    ax1.text(0.012, 0.97, box, transform=ax1.transAxes, va="top", ha="left",
             fontsize=8.5, family="monospace",
             bbox=dict(boxstyle="round,pad=0.5", fc="#f9f8f5", ec="#c5c1b8"))

    ax2.axhline(0, color="#908d85", lw=0.8)
    ax2.plot(ts, resid_pct, color="#b45309", lw=1.2)
    ax2.fill_between(ts, resid_pct, 0, color="#b45309", alpha=0.12)
    ax2.set_ylabel("resid %")
    ax2.set_xlabel("Time (UTC)")
    ax2.grid(True, color="#ece9e3", lw=0.7)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate(rotation=0, ha="center")

    sup = title + ("\n" + subtitle if subtitle else "")
    fig.suptitle(sup, x=0.012, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outpath, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return outpath