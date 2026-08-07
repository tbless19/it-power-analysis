"""
M100 IT Power Validation — upload-based test interface.

Run from project root:
    python validation/app.py
Open: http://localhost:5050

Upload logics_pub + job_table parquet files to overlay measured Tot_ict
against calculated power from server.py.
"""
import json
import os
import sys

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from validation.loaders import parse_jobs, parse_logics_ict  # noqa: E402
from validation.core import run_validation  # noqa: E402
from validation import exadata  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(_ROOT, "config.json")

with open(CFG_PATH) as f:
    CFG = json.load(f)

VCFG = CFG.get("validate", {})
M100 = CFG.get("m100_preset", {})
DEFAULT_INV = M100.get("inventory", {"V100": 3920})
DEFAULT_HW = VCFG.get("hardware_type", "V100")
DEFAULT_BASELINE_KW = VCFG.get("baseline_kw", 0.0)
DEFAULT_GRID = VCFG.get("grid_seconds", 300)
DEFAULT_DRAWS = VCFG.get("draws", 1)
DEFAULT_SEED = VCFG.get("rng_seed", 12345)
MAPE_FLOOR = VCFG.get("mape_floor_kw", 1.0)
MAX_GRID_POINTS = VCFG.get("max_grid_points", 120)
GPUS_PER_NODE = M100.get("gpus_per_node", 4)
TARGETS = VCFG.get("targets", {"power_mape_pct": 15, "energy_err_pct": 10})

_LOCAL_DATA_CACHE = {}
_LOCAL_DATA_CACHE_MAX = 6


def _local_data_cache_key(root, ym, t0, t1):
    return (exadata._resolve_root(root), ym, t0, t1)


def _load_local_window(root, ym, t0, t1):
    key = _local_data_cache_key(root, ym, t0, t1)
    if key in _LOCAL_DATA_CACHE:
        return _LOCAL_DATA_CACHE[key]

    measured_df = exadata.load_measured_ict(root, ym, t0, t1)
    jobs_df = exadata.load_jobs_df(root, ym, t0, t1)
    jobs = exadata.jobs_to_records(jobs_df, gpus_per_node=GPUS_PER_NODE)
    _LOCAL_DATA_CACHE[key] = (measured_df, jobs)
    if len(_LOCAL_DATA_CACHE) > _LOCAL_DATA_CACHE_MAX:
        _LOCAL_DATA_CACHE.pop(next(iter(_LOCAL_DATA_CACHE)))

    return measured_df, jobs

app = Flask(__name__, static_folder=BASE_DIR)


@app.route("/api/config")
def api_config():
    return jsonify({
        "hardware":   CFG["hardware"],
        "m100_preset": M100,
        "validate": {
            "hardware_type": DEFAULT_HW,
            "grid_seconds":  DEFAULT_GRID,
            "draws":         DEFAULT_DRAWS,
            "inventory":     DEFAULT_INV,
            "max_grid_points": MAX_GRID_POINTS,
            "targets":       TARGETS,
            "baseline_kw":   DEFAULT_BASELINE_KW,
        },
    })


@app.route("/api/validate", methods=["POST"])
def api_validate():
    logics_file = request.files.get("logics_file")
    jobs_file = request.files.get("jobs_file")

    if not logics_file or not jobs_file:
        return jsonify({"error": "Both logics_file and jobs_file are required"}), 400

    t0 = request.form.get("t0") or None
    t1 = request.form.get("t1") or None
    grid_s = int(request.form.get("grid_seconds", DEFAULT_GRID))
    draws = int(request.form.get("draws", DEFAULT_DRAWS))
    hw_type = request.form.get("hw_type", DEFAULT_HW)
    baseline_kw = float(request.form.get("baseline_kw", DEFAULT_BASELINE_KW) or 0)

    try:
        inventory = json.loads(request.form.get("inventory", "{}"))
        if not inventory:
            inventory = dict(DEFAULT_INV)
    except Exception:
        inventory = dict(DEFAULT_INV)

    errors = []
    measured_df, err = parse_logics_ict(logics_file.read(), t0, t1)
    if err:
        return jsonify({"error": f"logics: {err}"}), 400

    if t0 is None:
        t0 = measured_df["timestamp"].iloc[0].isoformat()
    if t1 is None:
        t1 = measured_df["timestamp"].iloc[-1].isoformat()

    jobs, err = parse_jobs(jobs_file.read(), t0, t1, gpus_per_node=GPUS_PER_NODE)
    if err:
        return jsonify({"error": f"jobs: {err}"}), 400
    if not jobs:
        return jsonify({"error": "No overlapping jobs in the selected window"}), 400

    try:
        result = run_validation(
            measured_df, jobs, inventory,
            hw_type=hw_type,
            grid_seconds=grid_s,
            seed=DEFAULT_SEED,
            draws=draws,
            mape_floor_kw=MAPE_FLOOR,
            max_grid_points=MAX_GRID_POINTS,
            baseline_kw=baseline_kw,
        )
    except Exception as ex:
        return jsonify({"error": str(ex)}), 500

    result["errors"] = errors
    result["meta"]["targets"] = TARGETS
    return jsonify(result)


@app.route("/api/windows")
def api_windows():
    root = request.args.get("root") or None
    return jsonify(exadata.list_windows(root))


@app.route("/api/validate-local", methods=["POST"])
def api_validate_local():
    b = request.get_json(force=True)
    root = b.get("root") or exadata.DATA_ROOT
    ym = b.get("year_month")
    t0 = b.get("t0")
    t1 = b.get("t1")
    if not (ym and t0 and t1):
        return jsonify({"error": "year_month, t0 and t1 are required"}), 400

    grid_s = int(b.get("grid_seconds", DEFAULT_GRID))
    draws = int(b.get("draws", DEFAULT_DRAWS))
    hw_type = b.get("hw_type", DEFAULT_HW)
    inventory = b.get("inventory") or dict(DEFAULT_INV)
    baseline_kw = float(b.get("baseline_kw", DEFAULT_BASELINE_KW) or 0)

    try:
        measured_df, jobs = _load_local_window(root, ym, t0, t1)
        if measured_df.empty:
            return jsonify({"error": "No Tot_ict rows in window"}), 400
        if not jobs:
            return jsonify({"error": "No overlapping jobs in window"}), 400

        result = run_validation(
            measured_df, jobs, inventory,
            hw_type=hw_type,
            grid_seconds=grid_s,
            seed=DEFAULT_SEED,
            draws=draws,
            mape_floor_kw=MAPE_FLOOR,
            max_grid_points=MAX_GRID_POINTS,
            baseline_kw=baseline_kw,
        )
    except FileNotFoundError as ex:
        return jsonify({"error": str(ex)}), 404
    except Exception as ex:
        return jsonify({"error": str(ex)}), 500

    result["meta"]["targets"] = TARGETS
    result["meta"]["root"] = exadata._resolve_root(root)
    result["meta"]["year_month"] = ym
    return jsonify(result)


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "validate.html")


@app.route("/local")
def index_local():
    return send_from_directory(BASE_DIR, "validate_local.html")


if __name__ == "__main__":
    print("M100 IT Power Validation")
    print("  Upload:  http://localhost:5050/")
    print("  Local:   http://localhost:5050/local")
    print(f"  Dataset: {exadata.DATA_ROOT}")
    print(f"  Project: {_ROOT}")
    app.run(debug=True, port=5050, host="0.0.0.0", use_reloader=False)
