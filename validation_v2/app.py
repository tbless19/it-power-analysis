"""
M100 Schedule Explorer (validation v2).

Measured IT power + job schedule Gantt + stage power — 1-day slice.

Run:
    python validation_v2/prepare_slice.py   # first time (builds cache)
    python validation_v2/app.py
Open: http://localhost:5051/
"""
import json
import os
import sys

from flask import Flask, jsonify, request, send_from_directory

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from validation_v2.explorer_core import (  # noqa: E402
    build_slice,
    default_slice_config,
    load_slice,
    slice_cache_path,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(_ROOT, "config.json")

with open(CFG_PATH) as f:
    CFG = json.load(f)

app = Flask(__name__, static_folder=BASE_DIR)


@app.route("/api/config")
def api_config():
    sc = default_slice_config()
    return jsonify({
        "slice": sc,
        "m100_preset": CFG.get("m100_preset", {}),
        "hardware": CFG.get("hardware", {}),
        "cache_path": slice_cache_path(),
        "cache_exists": os.path.exists(slice_cache_path()),
    })


@app.route("/api/slice")
def api_slice():
    rebuild = request.args.get("rebuild", "").lower() in ("1", "true", "yes")
    try:
        data = load_slice(force_rebuild=rebuild)
    except Exception as ex:
        return jsonify({"error": str(ex)}), 500
    return jsonify(data)


@app.route("/api/slice/rebuild", methods=["POST"])
def api_slice_rebuild():
    try:
        cfg = default_slice_config()
        body = request.get_json(silent=True) or {}
        cfg.update({k: body[k] for k in body if k in cfg})
        data = build_slice(cfg, force=True)
    except Exception as ex:
        return jsonify({"error": str(ex)}), 500
    return jsonify(data)


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "explorer.html")


if __name__ == "__main__":
    cache = slice_cache_path()
    if not os.path.exists(cache):
        print("No slice cache — building default 1-day slice (first run may take ~1 min) …")
        build_slice(default_slice_config(), force=False)
        print(f"  wrote {cache}")
    print("M100 Schedule Explorer → http://localhost:5051/")
    app.run(debug=True, port=5051, host="0.0.0.0", use_reloader=False)
