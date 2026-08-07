"""
generate_parquet.py
-------------------
Generates synthetic ExaData-style Parquet data for testing the IT Power Simulation.
Run this locally (needs pyarrow):

    pip install pandas pyarrow numpy
    python generate_parquet.py

Output
------
cleaned_parquet/
  dataset=main/
    year_month=22-03/
      plugin=job_table/
        part-000.parquet      3 000 jobs
      plugin=logics_pub/
        part-000.parquet      Tot_ict time-series (facility IT power, kW)
"""

import pandas as pd
import numpy as np
import os, random
from datetime import datetime, timezone, timedelta

rng = np.random.default_rng(42)
random.seed(42)

# ── helpers ───────────────────────────────────────────────────────────────────

BASE     = datetime(2022, 3, 1, tzinfo=timezone.utc)
MONTH_S  = 31 * 24 * 3600

def make_dir(plugin):
    path = f"cleaned_parquet/dataset=main/year_month=22-03/plugin={plugin}"
    os.makedirs(path, exist_ok=True)
    return f"{path}/part-000.parquet"


# ── 1. job_table — 3 000 jobs ─────────────────────────────────────────────────

PARTITIONS = ["m100_all_serial", "m100_usr_prod", "m100_usr_dbg", "m100_qos_bprod"]
QOS        = ["normal", "high", "low", "special"]

jobs = []
for i in range(3000):
    r = random.random()
    if r < 0.10:                          # training: long, many nodes
        dur_h     = random.uniform(1.0, 72.0)
        num_nodes = random.randint(16, 256)
    elif r < 0.30:                        # fine-tuning: medium
        dur_h     = random.uniform(0.25, 1.0)
        num_nodes = random.randint(4, 64)
    else:                                 # inference: short, few nodes
        dur_h     = random.uniform(0.005, 0.249)
        num_nodes = random.randint(1, 8)

    dur_s    = dur_h * 3600
    offset   = random.uniform(0, max(1, MONTH_S - dur_s))
    t_start  = BASE + timedelta(seconds=offset)
    t_end    = t_start + timedelta(seconds=dur_s)
    t_submit = t_start - timedelta(seconds=random.uniform(5, 1800))

    jobs.append({
        "job_id":      100000 + i,
        "submit_time": t_submit,
        "start_time":  t_start,
        "end_time":    t_end,
        "job_state":   "COMPLETED",
        "partition":   random.choice(PARTITIONS),
        "qos":         random.choice(QOS),
        "user_id":     random.randint(1000, 9999),
        "num_nodes":   num_nodes,
        "num_cpus":    num_nodes * 42,   # Power9: 42 CPUs per node
        "num_gpus":    num_nodes * 4,    # 4 × V100 per node
        "nodes":       f"r{random.randint(1,50):02d}n{random.randint(1,20):02d}",
        "metric":      "job_info_marconi100",
    })

df_jobs = pd.DataFrame(jobs)
df_jobs["submit_time"] = pd.to_datetime(df_jobs["submit_time"], utc=True)
df_jobs["start_time"]  = pd.to_datetime(df_jobs["start_time"],  utc=True)
df_jobs["end_time"]    = pd.to_datetime(df_jobs["end_time"],    utc=True)
df_jobs.to_parquet(make_dir("job_table"), index=False)

# Stage breakdown (using classify_stage rules)
def stage(row):
    dur_h = (row.end_time - row.start_time).total_seconds() / 3600
    n     = row.num_nodes
    if dur_h >= 1.0 and n >= 16: return "training"
    if dur_h < 0.25 or n < 4:   return "inference"
    return "fine_tuning"

df_jobs["_stage"] = df_jobs.apply(stage, axis=1)
counts = df_jobs["_stage"].value_counts()
print(f"job_table   : {len(df_jobs):,} rows")
print(f"  training  : {counts.get('training',   0):,}")
print(f"  fine-tune : {counts.get('fine_tuning',0):,}")
print(f"  inference : {counts.get('inference',  0):,}")


# ── 2. logics_pub — Tot_ict facility IT power (kW) ───────────────────────────
# One reading every 5 minutes across March 2022.
# Power is modelled as a base load + workload-driven bump.

n_pts      = MONTH_S // 300          # one reading per 5 min
timestamps = [BASE + timedelta(seconds=i * 300) for i in range(n_pts)]

# Base load (M100 idle ≈ 980 nodes × 4 GPUs × 300 W × 0.117 idle ≈ 138 kW)
base_kw = 138.0
# Add a daily cycle and workload noise
t_arr    = np.arange(n_pts)
daily    = 30 * np.sin(2 * np.pi * t_arr / (24 * 12))          # daily swing ±30 kW
noise    = rng.normal(0, 8, n_pts)                              # measurement noise
# Workload bumps aligned with training jobs (simplistic)
workload = np.zeros(n_pts)
for _, row in df_jobs[df_jobs["_stage"] == "training"].iterrows():
    s = int((row.start_time.timestamp() - BASE.timestamp()) / 300)
    e = int((row.end_time.timestamp()   - BASE.timestamp()) / 300)
    s, e = max(0, s), min(n_pts, e)
    if s < e:
        workload[s:e] += row.num_nodes * 4 * 300 * 0.78 / 1000  # active power kW

tot_ict = np.clip(base_kw + daily + noise + np.clip(workload, 0, 800), 50, 1200)

logics_rows = []
for i, ts in enumerate(timestamps):
    logics_rows.append({
        "timestamp":      ts,
        "metric":         "Tot_ict",
        "value_numeric":  round(float(tot_ict[i]), 2),
        "metric_group":   "computing",
        "plugin":         "logics_pub",
        "panel":          "generals",
        "device":         "pue",
        "year_month":     "22-03",
        "source_dataset": "main",
    })

df_logics = pd.DataFrame(logics_rows)
df_logics["timestamp"] = pd.to_datetime(df_logics["timestamp"], utc=True)
df_logics.to_parquet(make_dir("logics_pub"), index=False)
print(f"\nlogics_pub  : {len(df_logics):,} rows  (Tot_ict, 5-min cadence)")
print(f"  min kW    : {df_logics['value_numeric'].min():.1f}")
print(f"  max kW    : {df_logics['value_numeric'].max():.1f}")
print(f"  mean kW   : {df_logics['value_numeric'].mean():.1f}")

print("\nDone — files written to ./cleaned_parquet/")