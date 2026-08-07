"""Parse uploaded M100 ExaData parquet files from in-memory bytes."""
import io

import pandas as pd
import pyarrow.parquet as pq

# Minimal flat columns — skip panel/device at read time (can trigger pyarrow bugs on
# some Spark/Hive exports). Tot_ict is identified by metric name alone.
LOGICS_COLUMNS = ["timestamp", "metric", "value_numeric"]
LOGICS_COLUMNS_EXTENDED = ["timestamp", "metric", "value_numeric", "panel", "device"]
JOB_COLUMNS = ["metric", "start_time", "end_time", "num_nodes", "job_id"]



def _schema_columns(parquet_bytes):
    buf = io.BytesIO(parquet_bytes)
    return pq.ParquetFile(buf, pre_buffer=False).schema.names


def _pick_columns(parquet_bytes, wanted):
    available = set(_schema_columns(parquet_bytes))
    return [c for c in wanted if c in available]


def _read_fastparquet(parquet_bytes, columns=None):
    buf = io.BytesIO(parquet_bytes)
    cols = _pick_columns(parquet_bytes, columns) if columns else None
    return pd.read_parquet(buf, engine="fastparquet", columns=cols or None)


def _read_pyarrow(parquet_bytes, columns=None, filters=None):
    read_cols = _pick_columns(parquet_bytes, columns) if columns else None
    buf = io.BytesIO(parquet_bytes)
    return pq.read_table(
        buf, columns=read_cols, filters=filters, pre_buffer=False
    ).to_pandas()


def _read_pyarrow_row_groups(parquet_bytes, columns=None):
    read_cols = _pick_columns(parquet_bytes, columns) if columns else None
    buf = io.BytesIO(parquet_bytes)
    pf = pq.ParquetFile(buf, pre_buffer=False)
    chunks = []
    for i in range(pf.metadata.num_row_groups):
        try:
            chunks.append(pf.read_row_group(i, columns=read_cols).to_pandas())
        except Exception:
            continue
    if not chunks:
        raise ValueError("all row groups failed")
    return pd.concat(chunks, ignore_index=True)


def robust_read_parquet(parquet_bytes, columns=None, filters=None):
    """
    Read uploaded parquet bytes with fallbacks for Spark/Hive metadata quirks.

    fastparquet is tried first because some pyarrow builds (e.g. conda pyarrow 19
    on Python 3.13) abort on these files with 'Repetition level histogram size
    mismatch'. pyarrow with predicate pushdown is used when available (faster).
    """
    errors = []

    # 1. fastparquet — most compatible with ExaData/Spark exports
    try:
        df = _read_fastparquet(parquet_bytes, columns)
        return _apply_filters(df, filters)
    except Exception as e:
        errors.append(f"fastparquet: {e}")

    # 2. pyarrow with filters — fast path when pyarrow can read the file
    if filters:
        try:
            return _read_pyarrow(parquet_bytes, columns, filters)
        except Exception as e:
            errors.append(f"pyarrow/filtered: {e}")

    # 3. pyarrow columns only
    try:
        df = _read_pyarrow(parquet_bytes, columns, filters=None)
        return _apply_filters(df, filters)
    except Exception as e:
        errors.append(f"pyarrow/columns: {e}")

    # 4. pyarrow row groups + pandas filter
    try:
        df = _read_pyarrow_row_groups(parquet_bytes, columns)
        return _apply_filters(df, filters)
    except Exception as e:
        errors.append(f"pyarrow/row_groups: {e}")

    # 5. fastparquet full file (no column projection)
    try:
        df = _read_fastparquet(parquet_bytes, columns=None)
        return _apply_filters(df, filters)
    except Exception as e:
        errors.append(f"fastparquet/full: {e}")

    raise ValueError("All Parquet read strategies failed:\n" + "\n".join(errors))


def _apply_filters(df, filters):
    """Apply pyarrow-style filters in pandas when pushdown is unavailable."""
    if not filters or df is None or df.empty:
        return df
    out = df
    for col, op, val in filters:
        if col not in out.columns:
            continue
        series = out[col]
        if op == "==":
            out = out[series == val]
        elif op == ">=":
            out = out[series >= val]
        elif op == "<":
            out = out[series < val]
        elif op == ">":
            out = out[series > val]
    return out


def _to_utc(ts):
    if ts is None or str(ts).strip() == "":
        return None
    return pd.Timestamp(ts, tz="UTC")


def _logics_filters(t0=None, t1=None):
    filters = [("metric", "==", "Tot_ict")]
    t0p, t1p = _to_utc(t0), _to_utc(t1)
    if t0p is not None:
        filters.append(("timestamp", ">=", t0p))
    if t1p is not None:
        filters.append(("timestamp", "<", t1p))
    return filters


def _job_filters(t0=None, t1=None):
    filters = [("metric", "==", "job_info_marconi100")]
    t0p, t1p = _to_utc(t0), _to_utc(t1)
    if t0p is not None and t1p is not None:
        filters.append(("start_time", "<", t1p))
        filters.append(("end_time", ">", t0p))
    return filters


def parse_logics_ict(parquet_bytes, t0=None, t1=None):
    """
    Extract measured facility IT power (Tot_ict, kW) from logics_pub upload.
    Returns (DataFrame[timestamp, kw], error_or_none).
    """
    filters = _logics_filters(t0, t1)
    df = None
    last_err = None

    for cols in (LOGICS_COLUMNS, LOGICS_COLUMNS_EXTENDED):
        try:
            df = robust_read_parquet(parquet_bytes, columns=cols, filters=filters)
            break
        except Exception as e:
            last_err = e
            # Retry without pushdown filters (fastparquet path filters in pandas)
            try:
                df = robust_read_parquet(parquet_bytes, columns=cols, filters=None)
                df = _apply_filters(df, filters)
                break
            except Exception as e2:
                last_err = e2

    if df is None:
        return None, str(last_err or "failed to read logics parquet")

    if "metric" not in df.columns or "value_numeric" not in df.columns:
        return None, "logics_pub file missing required columns (metric, value_numeric)"

    mask = df["metric"] == "Tot_ict"
    if "panel" in df.columns:
        panel_ok = df["panel"].astype(str).isin(["generals", "nan", ""])
        if panel_ok.any():
            mask &= panel_ok
    if "device" in df.columns:
        device_ok = df["device"].astype(str).isin(["pue", "nan", ""])
        if device_ok.any():
            mask &= device_ok

    out = df.loc[mask, ["timestamp", "value_numeric"]].copy()
    if out.empty:
        out = df.loc[df["metric"] == "Tot_ict", ["timestamp", "value_numeric"]].copy()
    if out.empty:
        return None, "No Tot_ict rows found in logics_pub file"

    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp", "value_numeric"])
    out = out.rename(columns={"value_numeric": "kw"}).sort_values("timestamp")

    t0p, t1p = _to_utc(t0), _to_utc(t1)
    if t0p is not None:
        out = out[out["timestamp"] >= t0p]
    if t1p is not None:
        out = out[out["timestamp"] < t1p]
    out = out[out["kw"] > 0].reset_index(drop=True)

    if out.empty:
        return None, "No Tot_ict rows in the requested time window"
    return out, None


def parse_jobs(parquet_bytes, t0=None, t1=None, gpus_per_node=4):
    """
    Extract SLURM jobs from job_table upload for server.job_to_task.
    Returns (list[job dict], error_or_none).
    """
    filters = _job_filters(t0, t1)
    df = None
    last_err = None

    try:
        df = robust_read_parquet(parquet_bytes, columns=JOB_COLUMNS, filters=filters)
    except Exception as e:
        last_err = e
        try:
            df = robust_read_parquet(parquet_bytes, columns=JOB_COLUMNS, filters=None)
            df = _apply_filters(df, filters)
        except Exception as e2:
            return None, str(e2)

    if df is None:
        return None, str(last_err or "failed to read job_table parquet")

    if "start_time" not in df.columns or "end_time" not in df.columns:
        return None, "job_table file missing required columns (start_time, end_time)"

    if "metric" in df.columns:
        df = df[df["metric"] == "job_info_marconi100"]
    df = df.copy()
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True, errors="coerce")
    df["end_time"] = pd.to_datetime(df["end_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["start_time", "end_time"])
    df = df[df["end_time"] > df["start_time"]]

    t0p, t1p = _to_utc(t0), _to_utc(t1)
    if t0p is not None and t1p is not None:
        df = df[(df["start_time"] < t1p) & (df["end_time"] > t0p)]

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
    return jobs, None
