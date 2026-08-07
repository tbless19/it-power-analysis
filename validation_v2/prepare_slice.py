"""Build or refresh the cached 1-day M100 slice for Schedule Explorer."""
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from validation_v2.explorer_core import build_slice, default_slice_config, slice_cache_path  # noqa: E402


def main():
    cfg_path = os.path.join(os.path.dirname(__file__), "slice_config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg.setdefault("root", None)

    print("Building slice cache …")
    result = build_slice(cfg, force=True)
    m = result["meta"]
    print(f"  cache: {slice_cache_path()}")
    print(f"  window: {m['window']['t0']} → {m['window']['t1']}")
    print(f"  measured points: {m['measured_points']}")
    print(f"  ganglia GPU mean/peak: {m.get('mean_ganglia_kw')} / {m.get('peak_ganglia_kw')} kW "
          f"({m.get('ganglia_nodes')} nodes)")
    print(f"  r(fleet, ganglia): {m.get('r_fleet_vs_ganglia')}")
    print(f"  gantt jobs: {m['gantt_jobs']}")
    print(f"  modeled jobs: {m['job_count']}")
    print("Done.")


if __name__ == "__main__":
    main()
