"""
ExaData (M100 cleaned Parquet) data access layer.

Reads the Hive-partitioned cleaned bundle:
    <DATA_ROOT>/year_month=YY-MM/plugin=<name>/part-000.parquet

DATA_ROOT points at the dataset partition directory itself
(e.g. /data/dataset=main_dataset). The 'dataset=' level is consumed as the
root; year_month and plugin are discovered as Hive partitions below it.

This module returns plain pandas DataFrames. It does NOT depend on Flask or on
the model — so the validation maths can be tested against synthetic frames by
monkeypatching the three load_* functions.
"""
import os, glob, calendar
from datetime import datetime, timezone

import pandas as pd

# pyarrow is required for partition-filtered reads of the real bundle.
# It is optional at import time so the module can be imported (and the loaders
# monkeypatched) in environments without it.
try:
    import pyarrow.dataset as pads
    import pyarrow.compute as pc
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAVE_PYARROW = True
except Exception:                                    # pragma: no cover
    _HAVE_PYARROW = False

_DEFAULT_ENV = os.environ.get("EXADATA_ROOT", "")  # used by _default_root()

# Measured-power source definitions. Each describes how to turn a plugin's long
# rows into a single cluster-total kW series.
#   plugin       : Hive plugin partition to read
#   metric_match : ('exact', name) or ('contains', substr) on the `metric` column
#   scale        : multiply value_numeric by this to reach kW (W->kW = 0.001)
#   how          : 'sum'  -> sum all matching rows per timestamp (fleet total)
#                  'mean' -> average matching rows per timestamp (already-a-total)
MEASURED_SOURCES = {
    "ganglia_gpu": {
        "label":       "Aggregate GPU power (ganglia Gpu*_power_usage)",
        "plugin":      "ganglia_pub",
        "metric_match": ("contains", "power_usage"),
        "metric_prefix": "Gpu",
        "scale":       0.001,          # W -> kW
        "how":         "sum",
        "unit_src":    "W",
    },
    "logics_ict": {
        "label":       "Facility IT total (logics Tot_ict)",
        "plugin":      "logics_pub",
        "metric_match": ("exact", "Tot_ict"),
        "metric_prefix": None,
        "scale":       1.0,            # already kW
        "how":         "mean",         # a single facility total; mean dedups panels
        "unit_src":    "kW",
    },
    "ipmi_total": {
        "label":       "Aggregate node power (ipmi total_power)",
        "plugin":      "ipmi_pub",
        "metric_match": ("exact", "total_power"),
        "metric_prefix": None,
        "scale":       0.001,          # W -> kW
        "how":         "sum",
        "unit_src":    "W",
    },
}

JOB_PLUGIN = "job_table"


# ── helpers ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))


def _has_months(path):
    return bool(path and glob.glob(os.path.join(path, "year_month=*")))


def _default_root():
    env = os.environ.get("EXADATA_ROOT")
    if env:
        return env
    # Ordered candidates; the data folder next to this file wins so that an
    # in-project validate/data is preferred over an unrelated /data or ../data.
    candidates = [
        os.path.join(_HERE, "data"),            # e.g. validate/data
        "/data/dataset=main_datasets",
        "/data/dataset=main_dataset",
        "/data",
        os.path.join(_HERE, "..", "data"),
        os.path.join(os.getcwd(), "data"),
        "data",
    ]
    for c in candidates:
        r = _resolve_root(c)
        if _has_months(r):
            return r
    return os.path.join(_HERE, "data")


def _resolve_root(root):
    """Return the directory that directly contains year_month=* partitions.

    Accepts the dataset dir itself, or a parent 'data' dir holding a
    dataset=<name> level (any name, e.g. main_dataset / main_datasets).
    """
    if not root:
        return root
    if glob.glob(os.path.join(root, "year_month=*")):
        return root
    for ds in sorted(glob.glob(os.path.join(root, "dataset=*"))):
        if glob.glob(os.path.join(ds, "year_month=*")):
            return ds
    return root


DATA_ROOT = _default_root()
def _require_pyarrow():
    if not _HAVE_PYARROW:
        raise RuntimeError(
            "pyarrow is not installed in this environment. "
            "Install it (conda install pyarrow) to read the Parquet bundle."
        )


def _to_utc(ts):
    """Coerce anything timestamp-ish to a tz-aware UTC pandas Timestamp."""
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _month_bounds(ym):
    """Calendar bounds (UTC) for a 'YY-MM' partition string."""
    yy, mm = ym.split("-")
    year = 2000 + int(yy)
    month = int(mm)
    last = calendar.monthrange(year, month)[1]
    t0 = pd.Timestamp(year, month, 1, tz="UTC")
    t1 = pd.Timestamp(year, month, last, 23, 59, 59, tz="UTC")
    return t0, t1


def _partition_dir(root, ym, plugin):
    return os.path.join(root, f"year_month={ym}", f"plugin={plugin}")


# ── discovery ──────────────────────────────────────────────────────────────
def list_windows(root=None):
    """Return available months and (cheaply) the real job time-extent per month.

    Uses Parquet row-group statistics on the job_table timestamp/start_time
    column to report the actual data span without a full scan. Falls back to
    calendar month bounds if stats are unavailable.
    """
    root = root or DATA_ROOT
    root = _resolve_root(root)
    months = []
    if not os.path.isdir(root):
        return {"root": root, "exists": False, "months": []}

    for d in sorted(glob.glob(os.path.join(root, "year_month=*"))):
        ym = os.path.basename(d).split("=", 1)[1]
        cal0, cal1 = _month_bounds(ym)
        start_iso, end_iso = cal0.isoformat(), cal1.isoformat()
        # Try to tighten bounds from job_table stats.
        try:
            jf = os.path.join(_partition_dir(root, ym, JOB_PLUGIN), "part-000.parquet")
            if _HAVE_PYARROW and os.path.exists(jf):
                lo, hi = _column_minmax(jf, ("start_time", "submit_time", "timestamp"))
                if lo is not None:
                    start_iso = _to_utc(lo).isoformat()
                if hi is not None:
                    end_iso = _to_utc(hi).isoformat()
        except Exception:
            pass
        plugins = [os.path.basename(p).split("=", 1)[1]
                   for p in sorted(glob.glob(os.path.join(d, "plugin=*")))]
        months.append({
            "year_month": ym,
            "start": start_iso,
            "end": end_iso,
            "plugins": plugins,
        })
    return {"root": root, "exists": True, "months": months}


def _column_minmax(parquet_path, candidate_cols):
    """Min/max of the first present column using row-group statistics only."""
    pf = pq.ParquetFile(parquet_path)
    schema_names = set(pf.schema_arrow.names)
    col = next((c for c in candidate_cols if c in schema_names), None)
    if col is None:
        return None, None
    cidx = pf.schema_arrow.names.index(col)
    lo = hi = None
    md = pf.metadata
    for rg in range(md.num_row_groups):
        st = md.row_group(rg).column(cidx).statistics
        if st is None or not st.has_min_max:
            continue
        if lo is None or st.min < lo:
            lo = st.min
        if hi is None or st.max > hi:
            hi = st.max
    return lo, hi


# ── jobs ───────────────────────────────────────────────────────────────────
def load_jobs(root, ym, t0, t1):
    """Job records overlapping [t0, t1).

    Overlap rule: end_time > t0 AND start_time < t1. Jobs straddling the window
    edges are kept (the model clamps them via start_ms/end_ms comparisons).
    Returns a DataFrame with at least: job_id, start_time, end_time, num_nodes.
    """
    _require_pyarrow()
    root = _resolve_root(root)
    t0, t1 = _to_utc(t0), _to_utc(t1)
    pdir = _partition_dir(root, ym, JOB_PLUGIN)
    if not os.path.isdir(pdir):
        return pd.DataFrame()

    dset = pads.dataset(pdir, format="parquet")
    have = set(dset.schema.names)
    want = [c for c in ("job_id", "submit_time", "start_time", "end_time",
                        "job_state", "partition", "qos", "user_id",
                        "num_nodes", "num_cpus", "num_tasks", "nodes")
            if c in have]

    # Push the cheap half of the overlap predicate to the scanner when the
    # columns are timestamp-typed; refine in pandas afterwards.
    flt = None
    try:
        if "end_time" in have:
            flt = (pads.field("end_time") > pa.scalar(t0.to_pydatetime()))
        if "start_time" in have:
            s = (pads.field("start_time") < pa.scalar(t1.to_pydatetime()))
            flt = s if flt is None else (flt & s)
    except Exception:
        flt = None

    df = dset.to_table(columns=want, filter=flt).to_pandas()
    if df.empty:
        return df

    for c in ("start_time", "end_time", "submit_time"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")

    # Final exact overlap filter in pandas (covers non-pushable cases).
    if "start_time" in df.columns and "end_time" in df.columns:
        m = (df["end_time"] > t0) & (df["start_time"] < t1)
        df = df[m].copy()
    for c in ("num_nodes", "num_cpus", "num_tasks"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)


# ── measured power ─────────────────────────────────────────────────────────
def _aggregate_measured(df, spec, metric=None):
    """Turn long rows -> DataFrame[timestamp, kw]. Pure (no IO); tolerant of
    case/whitespace in the metric name. `metric` overrides the source's match."""
    cols = ["timestamp", "kw"]
    if df is None or df.empty or "metric" not in df.columns:
        return pd.DataFrame(columns=cols)
    df = df.copy()
    mser = df["metric"].astype(str).str.strip()
    if metric:                                   # explicit override (case-insensitive)
        sel = mser.str.casefold() == str(metric).strip().casefold()
    else:
        kind, val = spec["metric_match"]
        if kind == "exact":
            sel = mser.str.casefold() == str(val).casefold()
        else:                                    # contains
            sel = mser.str.contains(val, case=False, na=False)
        if spec.get("metric_prefix"):
            sel = sel & mser.str.casefold().str.startswith(spec["metric_prefix"].casefold())
    df = df[sel].dropna(subset=["timestamp", "value_numeric"])
    if df.empty:
        return pd.DataFrame(columns=cols)
    df["kw"] = pd.to_numeric(df["value_numeric"], errors="coerce") * spec["scale"]
    how = "sum" if spec["how"] == "sum" else "mean"
    out = df.groupby("timestamp", as_index=False)["kw"].agg(how)
    return out.sort_values("timestamp").reset_index(drop=True)


def load_measured(root, ym, t0, t1, source, metric=None):
    """Cluster-total measured power (kW) on its native cadence for [t0, t1).

    Returns DataFrame[timestamp(UTC), kw] sorted by timestamp. Metric matching
    is done in pandas (tolerant of case/whitespace); only the timestamp window
    is pushed to the scanner, so a slightly-off metric name still resolves.
    `metric` (optional) overrides the source's built-in metric match.
    """
    _require_pyarrow()
    if source not in MEASURED_SOURCES:
        raise ValueError(f"Unknown measured source '{source}'")
    spec = MEASURED_SOURCES[source]
    root = _resolve_root(root)
    t0, t1 = _to_utc(t0), _to_utc(t1)
    pdir = _partition_dir(root, ym, spec["plugin"])
    if not os.path.isdir(pdir):
        return pd.DataFrame(columns=["timestamp", "kw"])

    dset = pads.dataset(pdir, format="parquet")
    have = set(dset.schema.names)
    cols = [c for c in ("timestamp", "metric", "value_numeric") if c in have]

    flt = None
    try:
        if "timestamp" in have:
            flt = ((pads.field("timestamp") >= pa.scalar(t0.to_pydatetime())) &
                   (pads.field("timestamp") <  pa.scalar(t1.to_pydatetime())))
    except Exception:
        flt = None

    df = dset.to_table(columns=cols, filter=flt).to_pandas()
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "kw"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df[(df["timestamp"] >= t0) & (df["timestamp"] < t1)]
    return _aggregate_measured(df, spec, metric)


def list_metrics(root, ym, plugin, limit=50):
    """Distinct metric names (+counts) in a plugin for a month — reads only the
    `metric` column. Used by the UI to show what's actually available."""
    _require_pyarrow()
    root = _resolve_root(root)
    pdir = _partition_dir(root, ym, plugin)
    if not os.path.isdir(pdir):
        return []
    dset = pads.dataset(pdir, format="parquet")
    if "metric" not in set(dset.schema.names):
        return []
    s = dset.to_table(columns=["metric"]).column("metric").to_pandas().astype(str).str.strip()
    vc = s.value_counts().head(limit)
    return [{"metric": k, "count": int(v)} for k, v in vc.items()]