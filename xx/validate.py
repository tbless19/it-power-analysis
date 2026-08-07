"""
validate.py — measured-vs-simulated IT power overlay on the REAL cleaned Parquet bundle.

Just run it from inside the validate/ folder:
    python validate.py                      # auto-detects ../data and the first month
    python validate.py --list               # show available months + plugins, then exit
    python validate.py --month 22-03        # pick a month
    python validate.py --month 22-03 --t0 "2022-03-15 00:00" --t1 "2022-03-16 00:00"

--root is optional; if omitted it looks for a folder containing dataset=*/year_month=*/
(checks ./data, ../data, ../../data, ..). You can pass the data/ folder, the
dataset=main_datasets folder, or even a year_month folder — it climbs to the right level.
--month accepts '22-03' or 'year_month=22-03'.

Requires: pandas, numpy, matplotlib, pyarrow (all in your Anaconda env).
Needs server.py + config.json importable (same folder).
"""
import argparse, sys
import validation_core as vc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",  default=None, help="data/ folder (auto-detected if omitted)")
    ap.add_argument("--month", default=None, help="e.g. 22-03 (defaults to first available)")
    ap.add_argument("--t0", default=None, help='window start, e.g. "2022-03-15 00:00"')
    ap.add_argument("--t1", default=None, help='window end,   e.g. "2022-03-16 00:00"')
    ap.add_argument("--out", default=None, help="output PNG (default validation_<month>.png)")
    ap.add_argument("--list", action="store_true", help="list months + plugins and exit")
    args = ap.parse_args()

    # ---- resolve root -------------------------------------------------------
    root = vc.clean_root(args.root) if args.root else vc.autodetect_root()
    if not root or not vc.available_months(root):
        sys.exit("Could not find the data tree. Pass --root pointing at the 'data' folder "
                 "(the one containing dataset=main_datasets/).")
    months = vc.available_months(root)
     print(f"data root: {root}")
    print(f"months available: {', '.join(months)}")

    if args.list:
        for mth in months:
            print(f"  {mth}: {', '.join(vc.plugins_for(root, mth))}")
        return

    # ---- resolve month ------------------------------------------------------
    month = vc.clean_month(args.month) if args.month else months[0]
    if month not in months:
        sys.exit(f"Month {month!r} not found. Available: {', '.join(months)}")
    plugins = vc.plugins_for(root, month)
    for need in ("logics_pub", "job_table"):
        if need not in plugins:
            sys.exit(f"Month {month} is missing plugin={need} (has: {', '.join(plugins)}).")
    out = args.out or f"validation_{month}.png"
    print(f"month: {month}  (plugins: {', '.join(plugins)})")

    # ---- run ----------------------------------------------------------------
    print(f"[1/4] Loading measured Tot_ict …")
    meas = vc.measured_it_power(root, month, args.t0, args.t1)
    if meas.empty:
        sys.exit("No measured Tot_ict rows in that window — try without --t0/--t1, "
                 "or widen the window.")
    print(f"      {len(meas)} points  ({meas['timestamp'].iloc[0]} → {meas['timestamp'].iloc[-1]})")

    print(f"[2/4] Loading job_table → model jobs …")
    jobs = vc.load_jobs(root, month, args.t0, args.t1)
    print(f"      {len(jobs)} jobs overlap the window")

    print(f"[3/4] Simulating on the measured grid (server.py model) …")
    sim, tasks = vc.simulate_on_grid(jobs, meas["timestamp"])
    stage_counts = {}
    for t in tasks:
        stage_counts[t["stage"]] = stage_counts.get(t["stage"], 0) + 1
    print(f"      stage mix: {stage_counts}")

    m = vc.metrics(meas["kw"].to_numpy(), sim, meas["timestamp"])
    print(f"[4/4] MAPE {m['mape_pct']:.2f}%  |  energy error {m['energy_err_pct']:.2f}%")
    print(f"      energy: measured {m['e_meas_kwh']:.0f} kWh, simulated {m['e_sim_kwh']:.0f} kWh")

    vc.plot_overlay(
        meas["timestamp"], meas["kw"].to_numpy(), sim, m, out,
        title="M100 IT Power — Measured vs Simulated",
        subtitle=f"year_month={month}  ·  {len(jobs)} jobs  ·  "
                 f"Tot_ict (logics_pub) vs stage-aware model")
    print(f"Saved {out}")

if __name__ == "__main__":
    main()