"""Filesystem access to the M100 ExaData Hive-partitioned parquet bundle."""
import glob
import io
import os

import pandas as pd
import pyarrow.parquet as pq

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
JOB_PLUGIN = "job_table"
LOGICS_PLUGIN = "logics_pub"
GANGLIA_PLUGIN = "ganglia_pub"
LOGICS_COLUMNS = ["timestamp", "metric", "value_numeric"]
JOB_COLUMNS = ["metric", "start_time", "end_time", "num_nodes", "job_id"]
GANGLIA_POWER_METRICS = (
    "Gpu0_power_usage",
    "Gpu1_power_usage",
    "Gpu2_power_usage",
    "Gpu3_power_usage",
)


def _resolve_root(root):
    if not root:
        return root
    root = os.path.abspath(root)
    if glob.glob(os.path.join(root, "year_month=*")):
        return root
    for ds in sorted(glob.glob(os.path.join(root, "dataset=*"))):
        if glob.glob(os.path.join(ds, "year_month=*")):
            return ds
    return root


def default_root():
    env = os.environ.get("EXADATA_ROOT")
    if env:
        r = _resolve_root(env)
        if glob.glob(os.path.join(r, "year_month=*")):
            return r
    for c in [
        os.path.join(_ROOT, "data", "dataset=main_datasets"),
        os.path.join(_ROOT, "data"),
        os.path.join(os.getcwd(), "data", "dataset=main_datasets"),
        "data/dataset=main_datasets",
    ]:
        r = _resolve_root(c)
        if glob.glob(os.path.join(r, "year_month=*")):
            return r
    return os.path.join(_ROOT, "data", "dataset=main_datasets")


DATA_ROOT = default_root()


def _part_file(root, ym, plugin):
    """Prefer part-000.parquet; fall back to part-000-*.parquet (e.g. ganglia)."""
    d = os.path.join(root, f"year_month={ym}", f"plugin={plugin}")
    preferred = os.path.join(d, "part-000.parquet")
    if os.path.exists(preferred):
        return preferred
    alts = sorted(glob.glob(os.path.join(d, "part-000*.parquet")))
    if alts:
        return alts[0]
    return preferred


def _schema_columns(path):
    return pq.ParquetFile(path, pre_buffer=False).schema.names


def _pick_columns(path, wanted):
    have = set(_schema_columns(path))
    return [c for c in wanted if c in have]


def read_columns(path, wanted):
    """Read flat columns from a parquet file — fastparquet first for compatibility."""
    cols = _pick_columns(path, wanted)
    if not cols:
        return pd.DataFrame()
    try:
        return pd.read_parquet(path, engine="fastparquet", columns=cols)
    except Exception:
        pass
    try:
        return pq.read_table(path, columns=cols, pre_buffer=False).to_pandas()
    except Exception:
        pass
    pf = pq.ParquetFile(path, pre_buffer=False)
    chunks = []
    for i in range(pf.metadata.num_row_groups):
        try:
            chunks.append(pf.read_row_group(i, columns=cols).to_pandas())
        except Exception:
            continue
    if chunks:
        return pd.concat(chunks, ignore_index=True)
    raise ValueError(f"Could not read {path}")


def _to_utc(ts):
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def list_windows(root=None):
    root = _resolve_root(root or DATA_ROOT)
    if not os.path.isdir(root):
        return {"root": root, "exists": False, "months": []}
    months = []
    for d in sorted(glob.glob(os.path.join(root, "year_month=*"))):
        ym = os.path.basename(d).split("=", 1)[1]
        plugins = [
            os.path.basename(p).split("=", 1)[1]
            for p in sorted(glob.glob(os.path.join(d, "plugin=*")))
        ]
        months.append({"year_month": ym, "plugins": plugins})
    return {"root": root, "exists": True, "months": months}


def load_measured_ict(root, ym, t0, t1):
    """Return DataFrame[timestamp, kw] for Tot_ict in the window."""
    root = _resolve_root(root or DATA_ROOT)
    path = _part_file(root, ym, LOGICS_PLUGIN)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")

    t0p, t1p = _to_utc(t0), _to_utc(t1)
    df = read_columns(path, LOGICS_COLUMNS)
    df = df[df["metric"] == "Tot_ict"].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["kw"] = pd.to_numeric(df["value_numeric"], errors="coerce")
    df = df.dropna(subset=["timestamp", "kw"])
    df = df[(df["timestamp"] >= t0p) & (df["timestamp"] < t1p) & (df["kw"] > 0)]
    return df[["timestamp", "kw"]].sort_values("timestamp").reset_index(drop=True)


def load_ganglia_gpu_power(root, ym, t0, t1):
    """
    Sum per-GPU power_usage across nodes → fleet GPU power (kW).

    Aggregation: floor to 1 min → mean per (minute, node, GpuN) → sum → /1000.
    Do not sum raw samples in a bin (that multi-counts ~20 s cadence).
    Returns DataFrame[timestamp, kw] plus attrs: n_nodes, n_rows.
    """
    import pyarrow.dataset as ds

    root = _resolve_root(root or DATA_ROOT)
    path = _part_file(root, ym, GANGLIA_PLUGIN)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")

    t0p, t1p = _to_utc(t0), _to_utc(t1)
    dataset = ds.dataset(path, format="parquet")
    metric_filt = ds.field("metric").isin(list(GANGLIA_POWER_METRICS))
    time_filt = (ds.field("timestamp") >= t0p) & (ds.field("timestamp") < t1p)
    table = dataset.to_table(
        filter=metric_filt & time_filt,
        columns=["timestamp", "metric", "value_numeric", "node"],
    )
    if table.num_rows == 0:
        raise ValueError(f"No Gpu*_power_usage rows in {path} for [{t0p}, {t1p})")

    df = table.to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["value_numeric"] = pd.to_numeric(df["value_numeric"], errors="coerce")
    df = df.dropna(subset=["timestamp", "value_numeric", "node", "metric"])
    n_nodes = int(df["node"].nunique())
    n_rows = len(df)

    df["tmin"] = df["timestamp"].dt.floor("1min")
    per = df.groupby(["tmin", "node", "metric"], as_index=False)["value_numeric"].mean()
    fleet_w = per.groupby("tmin")["value_numeric"].sum()
    out = pd.DataFrame({
        "timestamp": fleet_w.index,
        "kw": (fleet_w / 1000.0).to_numpy(dtype=float),
    }).reset_index(drop=True)
    out.attrs["n_nodes"] = n_nodes
    out.attrs["n_rows"] = n_rows
    out.attrs["source_path"] = path
    return out


def load_jobs_df(root, ym, t0, t1):
    """Return overlapping job_table rows as a DataFrame."""
    root = _resolve_root(root or DATA_ROOT)
    path = _part_file(root, ym, JOB_PLUGIN)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")

    t0p, t1p = _to_utc(t0), _to_utc(t1)
    df = read_columns(path, JOB_COLUMNS)
    if "metric" in df.columns:
        df = df[df["metric"] == "job_info_marconi100"]
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True, errors="coerce")
    df["end_time"] = pd.to_datetime(df["end_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["start_time", "end_time"])
    df = df[(df["end_time"] > df["start_time"]) & (df["start_time"] < t1p) & (df["end_time"] > t0p)]
    return df.reset_index(drop=True)


def jobs_to_records(df, gpus_per_node=4):
    jobs = []
    for _, r in df.iterrows():
        n_nodes = int(r.get("num_nodes") or 1)
        jobs.append({
            "job_id":     str(r.get("job_id", "")),
            "start_time": r["start_time"].isoformat(),
            "end_time":   r["end_time"].isoformat(),
            "num_nodes":  n_nodes,
            "num_gpus":   n_nodes * gpus_per_node,
        })
    return jobs
