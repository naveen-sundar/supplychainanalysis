"""
Flask backend server for Route Optimization UI.
Bridges the React frontend with Python optimization scripts.

Endpoints:
  GET  /api/routes  — Load initial route data from CSV
  POST /api/run     — Run optimization (spawns Python subprocess)
  GET  /api/events  — SSE stream for live updates
"""

import json
import os
import queue
import subprocess
import sys
import threading
import pandas as pd

from flask import Flask, jsonify, request, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# In production, Tauri sets HATSUN_SCRIPT_DIR to the directory containing server.py.
# In dev, we derive it from __file__ as usual.
SCRIPT_DIR = os.environ.get("HATSUN_SCRIPT_DIR") or os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable  # Use the same Python that runs this server

# SSE event queue — one per client (simplified: single client)
event_queues = []


@app.route("/api/health", methods=["GET"])
def health():
    """Simple liveness check used by the frontend to wait for server startup."""
    return jsonify({"status": "ok"})


@app.route("/api/has-data", methods=["GET"])
def has_data():
    """Tell the frontend whether any uploaded route data is available.

    Returns {"has_uploaded": true} only when an explicit upload has been
    made.  The bundled static CSVs intentionally do NOT count — we want the
    map to start blank until the user provides their own data.
    """
    # Check the in-process override first (set after a /api/upload call)
    if route_optimizer.UPLOADED_CSV_PATH and os.path.exists(route_optimizer.UPLOADED_CSV_PATH):
        return jsonify({"has_uploaded": True})

    # Check HATSUN_DATA_DIR — the production upload destination
    data_dir = os.environ.get("HATSUN_DATA_DIR")
    if data_dir:
        if os.path.exists(os.path.join(data_dir, "Uploaded_Unified_Route_Data.csv")):
            return jsonify({"has_uploaded": True})

    # Check the dev upload destination (reuse SCRIPT_DIR which is already set at module level)
    dev_path = os.path.join(SCRIPT_DIR, "..", "csv_files", "Uploaded_Unified_Route_Data.csv")
    if os.path.exists(dev_path):
        return jsonify({"has_uploaded": True})

    return jsonify({"has_uploaded": False})


def send_event(data: dict):
    """Push an event to all connected SSE clients."""
    msg = json.dumps(data)
    for q in event_queues:
        q.put(msg)


# ─── Load initial routes (reuse route_optimizer's loader) ────────────────────

sys.path.insert(0, SCRIPT_DIR)
import route_optimizer


@app.route("/api/routes", methods=["GET"])
def get_routes():
    """Return initial route data for map display."""
    try:
        routes = route_optimizer.load_route_data()
        cc_lat, cc_lon = route_optimizer.CC_LAT, route_optimizer.CC_LON
        routes_json = []
        for r in routes:
            routes_json.append({
                "route_code": r.code,
                "route_name": r.name,
                "total_km": r.total_km,
                "capacity": r.capacity,
                "milk_qty": r.current_milk_qty,
                "uti_percent": r.uti_percent,
                "hmbs": [
                    {"name": h.name, "lat": h.lat, "lng": h.lon,
                     "sap_code": h.sap_code, "sequence": h.sequence}
                    for h in r.hmbs
                ],
            })
        return jsonify({"routes": routes_json, "cc": {"lat": cc_lat, "lng": cc_lon}})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload", methods=["POST"])
def upload_csv():
    """Upload a new Unified CSV file and reload the backend's memory."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400
        if file:
            # In production, HATSUN_DATA_DIR is set by Tauri to AppData\Local\HatsunVRP
            # (a writable location). In dev, fall back to the relative csv_files dir.
            data_dir = os.environ.get("HATSUN_DATA_DIR")
            if data_dir:
                csv_dir = data_dir
            else:
                csv_dir = os.path.join(SCRIPT_DIR, "..", "csv_files")
            os.makedirs(csv_dir, exist_ok=True)
            filepath = os.path.join(csv_dir, "Uploaded_Unified_Route_Data.csv")

            if file.filename.endswith(".xlsx") or file.filename.endswith(".xls"):
                df = pd.read_excel(file)
                df.to_csv(filepath, index=False)
            else:
                file.save(filepath)

            # Clear any stale cached path so a fresh load always happens.
            # This prevents a second upload from reusing the first file's data.
            route_optimizer.UPLOADED_CSV_PATH = None
            # Now point to the newly saved file and reload.
            route_optimizer.UPLOADED_CSV_PATH = filepath
            # Re-parse: updates CC_LAT/CC_LON globals and validates the file.
            route_optimizer.load_route_data()
            return jsonify({"status": "success", "saved_to": filepath})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── SSE endpoint ───────────────────────────────────────────────────────────

@app.route("/api/events", methods=["GET"])
def sse_events():
    """Server-Sent Events stream."""
    q = queue.Queue()
    event_queues.append(q)

    def stream():
        try:
            while True:
                try:
                    msg = q.get(timeout=30)
                    yield f"data: {msg}\n\n"
                except queue.Empty:
                    yield f"data: {json.dumps({'type': 'ping'})}\n\n"
        except GeneratorExit:
            pass
        finally:
            if q in event_queues:
                event_queues.remove(q)

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ─── Run optimization ───────────────────────────────────────────────────────

def _run_script(cmd, run_type):
    """Run a Python script and parse its JSON output."""
    try:
        send_event({"type": "phase", "phase": run_type, "status": "starting"})
        send_event({"type": "log", "message": f"Starting {run_type}..."})
        send_event({"type": "log", "message": f"Command: {' '.join(cmd)}"})

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, cwd=SCRIPT_DIR
        )

        if result.returncode != 0:
            err = result.stderr.strip() or "Unknown error"
            send_event({"type": "log", "message": f"ERROR: {err}"})
            send_event({"type": "error", "message": err})
            return None

        # Parse JSON from stdout
        stdout = result.stdout.strip()
        if not stdout:
            send_event({"type": "error", "message": "No output from script"})
            return None

        data = json.loads(stdout)
        send_event({"type": "log", "message": f"{run_type} completed successfully"})
        return data

    except subprocess.TimeoutExpired:
        send_event({"type": "error", "message": f"{run_type} timed out (5 min limit)"})
        return None
    except json.JSONDecodeError as e:
        send_event({"type": "error", "message": f"Failed to parse output: {e}"})
        return None
    except Exception as e:
        send_event({"type": "error", "message": str(e)})
        return None


def _run_optimization(params):
    """Background thread: run the requested optimization."""
    run_type = params.get("type", "insertion")
    lat = params.get("lat")
    lon = params.get("lon")
    milk_qty = params.get("milk_qty", 0)
    mode = params.get("mode", "osrm")

    send_event({"type": "status", "status": "running"})

    if run_type == "insertion":
        # Insertion optimization via route_optimizer.py
        if lat is None or lon is None:
            send_event({"type": "error", "message": "Lat/Lon required for insertion mode"})
            send_event({"type": "status", "status": "error"})
            return

        cmd = [
            PYTHON, os.path.join(SCRIPT_DIR, "route_optimizer.py"),
            "--lat", str(lat), "--lon", str(lon),
            "--milk-qty", str(milk_qty or 0),
            "--mode", mode, "--json"
        ]
        data = _run_script(cmd, "Insertion Optimization")
        if data:
            send_event({
                "type": "insertion_results", 
                "data": data.get("all_results", []),
                "new_hmb": data.get("new_hmb"),
                "cc": data.get("cc")
            })

    elif run_type == "full":
        # Full GA re-optimization via route_GA.py
        cmd = [
            PYTHON, os.path.join(SCRIPT_DIR, "route_GA.py"),
            "--mode", mode, "--json"
        ]
        if lat is not None and lon is not None and str(lat).strip() and str(lon).strip():
            cmd += ["--lat", str(lat), "--lon", str(lon),
                    "--milk-qty", str(milk_qty or 0)]

        data = _run_script(cmd, "Full Route Optimization (GA)")
        if data:
            send_event({"type": "full_results", "data": data})

    send_event({"type": "status", "status": "complete"})


@app.route("/api/run", methods=["POST"])
def run_optimization():
    """Kick off an optimization run in a background thread."""
    params = request.get_json(force=True)
    thread = threading.Thread(target=_run_optimization, args=(params,), daemon=True)
    thread.start()
    return jsonify({"status": "started"})


if __name__ == "__main__":
    print("Route Optimization Server starting on http://localhost:5051")
    print(f"Script dir: {SCRIPT_DIR}")
    print(f"Python: {PYTHON}")
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
