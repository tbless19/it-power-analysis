#!/usr/bin/env python3
"""
Run on YOUR machine (this cloud VM cannot see your Downloads folder).

From the it-power-analysis repo:

    pip install -r validation/requirements.txt -r rl_tune/requirements.txt
    python -m rl_tune.run_local_correlation \\
        --root "/path/to/dataset=main_datasets"

That folder should contain year_month=22-03, 22-04, … with plugin=ganglia_pub
and plugin=job_table (the layout in your screenshot).

Default: one calendar day per month (the 15th) so Ganglia stays tractable.
Writes CSV + markdown + plots under --out (default: rl_tune/results_local/).
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
from calendar import monthrange
from datetime import datetime, timedelta, timezone

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rl_tune.correlate_params import (  # noqa: E402
    STAGES,
    analyze,
    plot_report,
    write_config_variable_table,
)


def _ym_to_year_month(ym: str) -> tuple[int, int]:
    yy, mm = ym.split("-")
    return 2000 + int(yy), int(mm)


def discover_months(root: str) -> list[str]:
    from validation import exadata
    info = exadata.list_windows(root)
    months = []
    for m in info.get("months") or []:
        plugins = set(m.get("plugins") or [])
        if "ganglia_pub" in plugins and "job_table" in plugins:
            months.append(m["year_month"])
    return months


def concat_slices(parts: list[dict]) -> dict:
    """Stack 1-day explorer slices into one payload for analyze()."""
    if not parts:
        raise ValueError("no slices to concatenate")
    out = copy.deepcopy(parts[0])
    keys_1d = ("kw", "fleet_kw", "alloc_gpus")
    out["measured"]["kw"] = _cat([p["measured"]["kw"] for p in parts])
    out["measured"]["labels"] = sum((p["measured"].get("labels") or [] for p in parts), [])
    out["modeled"]["fleet_kw"] = _cat([p["modeled"]["fleet_kw"] for p in parts])
    out["modeled"]["alloc_gpus"] = _cat([p["modeled"]["alloc_gpus"] for p in parts])
    for s in STAGES:
        out["modeled"]["active_nodes"][s] = _cat(
            [p["modeled"]["active_nodes"][s] for p in parts]
        )
        out["modeled"]["active_jobs"][s] = _cat(
            [p["modeled"]["active_jobs"][s] for p in parts]
        )
        by = [p["modeled"].get("by_stage", {}).get(s, []) for p in parts]
        if any(by):
            out["modeled"].setdefault("by_stage", {})[s] = _cat(by)
    meta = out.setdefault("meta", {})
    meta["year_month"] = "pooled"
    meta["windows"] = [p.get("meta", {}).get("window") for p in parts]
    meta["n_days"] = len(parts)
    t0s = [w["t0"] for w in meta["windows"] if w]
    t1s = [w["t1"] for w in meta["windows"] if w]
    if t0s:
        meta["window"] = {"t0": t0s[0], "t1": t1s[-1]}
    del keys_1d
    return out


def _cat(seqs):
    out = []
    for s in seqs:
        out.extend(list(s))
    return out


def _pick_day_window(ym: str, day: int) -> tuple[str, str]:
    year, month = _ym_to_year_month(ym)
    last = monthrange(year, month)[1]
    d = min(max(1, day), last)
    t0 = datetime(year, month, d, tzinfo=timezone.utc)
    t1 = t0 + timedelta(days=1)
    return t0.strftime("%Y-%m-%dT00:00:00Z"), t1.strftime("%Y-%m-%dT00:00:00Z")


def build_one_day(root: str, ym: str, day: int, out_dir: str, force: bool) -> dict:
    from validation_v2.explorer_core import build_slice, default_slice_config

    cfg = default_slice_config()
    cfg["root"] = root
    cfg["year_month"] = ym
    last_err = None
    year, month = _ym_to_year_month(ym)
    last = monthrange(year, month)[1]
    for d in range(day, min(day + 7, last + 1)):
        t0, t1 = _pick_day_window(ym, d)
        cfg["t0"], cfg["t1"] = t0, t1
        cache = os.path.join(out_dir, "slices", f"{ym}_{t0[:10]}.json")
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        print(f"  loading {ym} {t0[:10]} …", flush=True)
        try:
            return build_slice(cfg, force=force, cache_path=cache)
        except Exception as exc:
            last_err = exc
            print(f"    skip {t0[:10]}: {exc}", flush=True)
    raise RuntimeError(f"no usable day in {ym} starting {day}: {last_err}")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Correlate config.json knobs with Ganglia GPU power on local ExaData."
    )
    p.add_argument(
        "--root",
        default=os.environ.get("EXADATA_ROOT", ""),
        help="Path to dataset=main_datasets (folder that contains year_month=*). "
             "Or set EXADATA_ROOT.",
    )
    p.add_argument("--out", default=os.path.join(_ROOT, "rl_tune", "results_local"))
    p.add_argument("--day", type=int, default=15, help="Calendar day to take from each month (default 15).")
    p.add_argument("--months", nargs="*", default=None, help="Subset like 22-03 22-04. Default: all with ganglia+jobs.")
    p.add_argument("--force", action="store_true", help="Rebuild day caches even if JSON exists.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = os.path.abspath(os.path.expanduser(args.root)) if args.root else ""
    if not root or not os.path.isdir(root):
        print(
            "Pass your local ExaData root, for example:\n\n"
            "  python -m rl_tune.run_local_correlation \\\n"
            "      --root \"$HOME/Downloads/.../dataset=main_datasets\"\n",
            file=sys.stderr,
        )
        return 2

    print(f"ExaData root: {root}", flush=True)
    months = args.months or discover_months(root)
    if not months:
        print(f"No year_month=*/plugin=ganglia_pub + job_table under {root}", file=sys.stderr)
        return 1
    print(f"Months: {', '.join(months)}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    parts = []
    failed = []
    for ym in months:
        try:
            parts.append(build_one_day(root, ym, args.day, args.out, args.force))
        except Exception as exc:
            failed.append((ym, str(exc)))
            print(f"FAILED {ym}: {exc}", flush=True)
    if not parts:
        print("No days loaded.", file=sys.stderr)
        return 1

    pooled = concat_slices(parts)
    report = analyze(pooled)
    report["meta_local"] = {
        "root": root,
        "months": months,
        "days_loaded": len(parts),
        "failed": failed,
    }
    plots = plot_report(report, args.out)
    md_path = os.path.join(args.out, "config_variables_vs_ganglia.md")
    csv_path = os.path.join(args.out, "config_variables_vs_ganglia.csv")
    write_config_variable_table(report["config_variables"], md_path, csv_path)

    print("\n=== config.json vs Ganglia (pooled days) ===")
    for row in report["config_variables"]:
        r = row["pearson_r_vs_ganglia"]
        pr = row["partial_r_vs_ganglia"]
        if r is None:
            continue
        print(f"  {row['config_variable']:55s}  r={r:+.3f}  partial={'' if pr is None else f'{pr:+.3f}'}")
    print(f"\nWrote:\n  {csv_path}\n  {md_path}")
    for p in plots:
        print(f"  {p}")
    if failed:
        print("\nSkipped months:", failed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
