"""
Validation v3 — 1-day measured vs simulated compare.

Run:
    python validation_v3/app.py
Open: http://localhost:5052/
"""
import json
import os
import sys

from flask import Flask, jsonify, request, send_from_directory

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from validation_v3.improve_core import (  # noqa: E402
    DEFAULT_BASELINE_KW,
    build_compare,
    recompute_with_baseline,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(_ROOT, "config.json")

with open(CFG_PATH) as f:
    CFG = json.load(f)

app = Flask(__name__, static_folder=BASE_DIR)


@app.route("/api/config")
def api_config():
    return jsonify({
        "default_baseline_kw": DEFAULT_BASELINE_KW,
        "hardware": CFG.get("hardware", {}),
        "m100_preset": CFG.get("m100_preset", {}),
    })


@app.route("/api/compare")
def api_compare():
    baseline = request.args.get("baseline_kw")
    rebuild = request.args.get("rebuild", "").lower() in ("1", "true", "yes")
    try:
        kw = float(baseline) if baseline not in (None, "") else DEFAULT_BASELINE_KW
        data = build_compare(baseline_kw=kw, force_rebuild=rebuild)
    except Exception as ex:
        return jsonify({"error": str(ex)}), 500
    return jsonify(data)


@app.route("/api/compare/recompute", methods=["POST"])
def api_recompute():
    body = request.get_json(force=True) or {}
    baseline = float(body.get("baseline_kw", DEFAULT_BASELINE_KW) or 0)
    payload = body.get("payload")
    if not payload:
        return jsonify({"error": "payload required"}), 400
    try:
        data = recompute_with_baseline(payload, baseline)
    except Exception as ex:
        return jsonify({"error": str(ex)}), 500
    return jsonify(data)


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "improve.html")


if __name__ == "__main__":
    print("Validation v3 → http://localhost:5052/")
    print(f"  1-day slice · non-GPU baseline default {DEFAULT_BASELINE_KW} kW")
    app.run(debug=True, port=5052, host="0.0.0.0", use_reloader=False)
