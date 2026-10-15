"""
Anchored-release firing telemetry visualizer (2026-09-22 session).

Fires single shots using the anchored-release scheme (pong_common.py's
compute_anchored_launch_params + fire_shot_anchored - deceleration always
begins at the same "virtual hard stop" arm angle regardless of speed,
vel_limit controls exit velocity independently), samples LAUNCH telemetry
during the actual launch move, and shows predicted-vs-actual graphs for
position, velocity, and acceleration right after the shot.

Predicted traces are the closed-form piecewise-constant-acceleration
kinematics compute_anchored_launch_params() is built on: accelerate at
accel_limit up to vel_limit, then immediately decelerate at decel_limit to
zero (no slack/cruise distance, by construction of start_pos/final_target).
Actual acceleration isn't a real ODrive telemetry value - it's derived by
numerically differentiating the sampled vel_estimate trace, so expect it to
look noticeably noisier than the position/velocity traces.

Locks PAN first (same as real operation, held for the whole session) and
fires real shots on LAUNCH.

Run with: odrive_env/bin/python anchored_fire_visualizer.py
Then open http://<this machine's LAN IP>:5003/ from another device.
"""
import json
import os
import threading
import time

from flask import Flask, jsonify, request
from odrive.enums import AxisState

from pong_common import (
    WATCHDOG_FEED_INTERVAL, WatchdogFeeder,
    compute_anchored_launch_params, connect_launch, connect_pan,
    fire_shot_anchored, lock_pan, prepare_launch,
)

HOST = "0.0.0.0"
PORT = 5003
SAMPLE_POST_BUFFER_S = 0.3  # keep sampling briefly after the launch move settles
SHOTS_DIR = "data/anchored_shots"  # one JSON file per fired shot, kept for later review


class Sampler:
    def __init__(self, axis, t0):
        self._axis = axis
        self._t0 = t0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.rows = []  # (t, pos, vel)

    def _run(self):
        while not self._stop.is_set():
            t = time.time() - self._t0
            pos = self._axis.pos_estimate
            vel = self._axis.vel_estimate
            self.rows.append((t, pos, vel))

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)


def numerical_accel(rows):
    """rows: list of (t, pos, vel). Central-difference on vel at each
    interior sample - ODrive has no direct acceleration telemetry."""
    n = len(rows)
    accels = []
    for i in range(n):
        if i == 0 or i == n - 1:
            accels.append(0.0)
            continue
        t0, _, v0 = rows[i - 1]
        t1, _, v1 = rows[i + 1]
        dt = t1 - t0
        accels.append((v1 - v0) / dt if dt > 0 else 0.0)
    return accels


def predicted_trace(params, sample_times):
    """Closed-form pos/vel/accel at each time in sample_times, for the
    accelerate-to-vel_limit-then-immediately-decelerate move
    compute_anchored_launch_params() sizes start_pos/final_target for."""
    v_limit = params["vel_limit"]
    accel = params["accel_limit"]
    decel = params["decel_limit"]
    start_pos = params["start_pos"]
    anchor = params["release_anchor_pos"]
    final_target = params["final_target"]
    t1 = v_limit / accel
    t2 = v_limit / decel

    out = []
    for t in sample_times:
        if t <= 0:
            pos, vel, acc = start_pos, 0.0, 0.0
        elif t <= t1:
            pos = start_pos + 0.5 * accel * t * t
            vel = accel * t
            acc = accel
        elif t <= t1 + t2:
            td = t - t1
            pos = anchor + v_limit * td - 0.5 * decel * td * td
            vel = v_limit - decel * td
            acc = -decel
        else:
            pos, vel, acc = final_target, 0.0, 0.0
        out.append((t, pos, vel, acc))
    return out


def main():
    print("=" * 55)
    print("ANCHORED-RELEASE FIRING VISUALIZER")
    print("=" * 55)

    os.makedirs(SHOTS_DIR, exist_ok=True)

    odrv_pan, axis_pan = connect_pan()
    pan_lock_target = lock_pan(odrv_pan, axis_pan)
    if pan_lock_target is None:
        print("ERROR: could not lock PAN, aborting.")
        raise SystemExit(1)
    pan_feeder = WatchdogFeeder(axis_pan, WATCHDOG_FEED_INTERVAL)
    pan_feeder.start()

    odrv_launch, axis_launch = connect_launch()
    ready_pos = prepare_launch(odrv_launch, axis_launch)
    if ready_pos is None:
        print("ERROR: could not prepare LAUNCH, aborting.")
        pan_feeder.stop()
        axis_pan.requested_state = AxisState.IDLE
        raise SystemExit(1)
    launch_feeder = WatchdogFeeder(axis_launch, WATCHDOG_FEED_INTERVAL)
    launch_feeder.start()

    fire_lock = threading.Lock()

    app = Flask(__name__)

    @app.route("/")
    def index():
        return INDEX_HTML

    @app.route("/api/fire", methods=["POST"])
    def api_fire():
        if not fire_lock.acquire(blocking=False):
            return jsonify({"ok": False, "message": "A shot is already in progress."}), 409

        try:
            data = request.get_json(force=True)
            vel_limit = float(data.get("vel_limit"))
        except (TypeError, ValueError):
            fire_lock.release()
            return jsonify({"ok": False, "message": "vel_limit must be a number."}), 400

        try:
            params = compute_anchored_launch_params(odrv_launch, axis_launch, vel_limit)
            if params is None:
                return jsonify({
                    "ok": False,
                    "message": "Rejected by the safety check - too close to a hard "
                               "stop at this vel_limit (see server console for the "
                               "exact bound). Try a different vel_limit.",
                }), 400

            sampler_holder = {}

            def on_windup_done():
                sampler_holder["sampler"] = Sampler(axis_launch, time.time())
                sampler_holder["sampler"].start()

            result = fire_shot_anchored(axis_launch, ready_pos, params,
                                         on_windup_done=on_windup_done)

            sampler = sampler_holder.get("sampler")
            if sampler is not None:
                time.sleep(SAMPLE_POST_BUFFER_S)
                sampler.stop()

            if result != "done":
                return jsonify({"ok": False,
                                 "message": f"fire_shot_anchored returned {result!r} - "
                                            "check the arm before firing again."}), 500

            rows = sampler.rows if sampler is not None else []
            accels = numerical_accel(rows)
            actual = [{"t": t, "pos": p, "vel": v, "acc": a}
                      for (t, p, v), a in zip(rows, accels)]

            sample_times = [r[0] for r in rows] if rows else [i * 0.005 for i in range(200)]
            predicted = [{"t": t, "pos": p, "vel": v, "acc": a}
                         for t, p, v, a in predicted_trace(params, sample_times)]

            shot_id = time.strftime("%Y%m%d_%H%M%S_") + f"{int(time.time() * 1000) % 1000:03d}"
            payload = {"ok": True, "id": shot_id, "params": params, "actual": actual,
                       "predicted": predicted}
            with open(os.path.join(SHOTS_DIR, f"{shot_id}.json"), "w") as f:
                json.dump(payload, f)

            return jsonify(payload)
        finally:
            fire_lock.release()

    @app.route("/api/shots")
    def api_shots_list():
        names = sorted(
            (f[:-5] for f in os.listdir(SHOTS_DIR) if f.endswith(".json")),
            reverse=True,
        )
        shots = []
        for name in names:
            try:
                with open(os.path.join(SHOTS_DIR, f"{name}.json")) as f:
                    data = json.load(f)
                shots.append({"id": name, "vel_limit": data["params"]["vel_limit"]})
            except (OSError, json.JSONDecodeError, KeyError):
                continue
        return jsonify({"shots": shots})

    @app.route("/api/shots/<shot_id>")
    def api_shots_get(shot_id):
        path = os.path.join(SHOTS_DIR, f"{shot_id}.json")
        if "/" in shot_id or not os.path.isfile(path):
            return jsonify({"ok": False, "message": "No such shot."}), 404
        with open(path) as f:
            return jsonify(json.load(f))

    print(f"\nOpen http://<this machine's LAN IP>:{PORT}/ from another device.\n")

    try:
        app.run(host=HOST, port=PORT, threaded=True, use_reloader=False)
    finally:
        launch_feeder.stop()
        pan_feeder.stop()
        axis_launch.requested_state = AxisState.IDLE
        axis_pan.requested_state = AxisState.IDLE
        print("\nBoth axes disabled.")


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Anchored Launch Visualizer</title>
<script src="/static/chart.umd.min.js"></script>
<style>
  :root { color-scheme: light; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #f5f5f7; color: #1d1d1f; margin: 0; padding: 24px;
  }
  .wrap { max-width: 900px; margin: 0 auto; }
  h1 { font-size: 20px; font-weight: 600; margin: 0 0 20px; }
  .card {
    background: #fff; border-radius: 12px; padding: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 16px;
  }
  .controls { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
  label { font-size: 14px; color: #6e6e73; }
  input[type=number] {
    font-size: 16px; padding: 8px 10px; border: 1px solid #d2d2d7;
    border-radius: 8px; width: 120px;
  }
  select {
    font-size: 15px; padding: 8px 10px; border: 1px solid #d2d2d7;
    border-radius: 8px; background: #fff; color: #1d1d1f;
  }
  button {
    font-size: 15px; font-weight: 500; padding: 9px 18px; border: none;
    border-radius: 8px; background: #0071e3; color: #fff; cursor: pointer;
  }
  button:disabled { background: #a9c9ec; cursor: default; }
  #message { font-size: 14px; margin-top: 10px; color: #6e6e73; min-height: 18px; }
  #message.error { color: #d70015; }
  .chart-title { font-size: 13px; color: #6e6e73; margin: 0 0 8px; font-weight: 500; }
  canvas { max-height: 240px; }
  .legend-note { font-size: 12px; color: #86868b; margin-top: 6px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Anchored-Release Launch Visualizer</h1>

  <div class="card">
    <div class="controls">
      <label for="vel">vel_limit (turns/s)</label>
      <input type="number" id="vel" step="0.01" value="4.41">
      <button id="fireBtn" onclick="fire()">Fire</button>
    </div>
    <div class="controls" style="margin-top: 12px;">
      <label for="shotSelect">Review shot</label>
      <select id="shotSelect" onchange="loadSelectedShot()">
        <option value="">(none saved yet)</option>
      </select>
    </div>
    <div id="message"></div>
  </div>

  <div class="card">
    <p class="chart-title">Angular position (turns) - solid = actual, dashed = predicted</p>
    <canvas id="posChart"></canvas>
  </div>
  <div class="card">
    <p class="chart-title">Angular velocity (turns/s) - solid = actual, dashed = predicted</p>
    <canvas id="velChart"></canvas>
  </div>
  <div class="card">
    <p class="chart-title">Angular acceleration (turns/s&sup2;) - solid = actual, dashed = predicted</p>
    <canvas id="accChart"></canvas>
    <p class="legend-note">Actual acceleration is numerically differentiated from measured
    velocity, not a direct sensor reading - expect more noise here than in position/velocity.</p>
  </div>
</div>

<script>
let charts = {};

function makeChart(canvasId, yLabel) {
  const ctx = document.getElementById(canvasId).getContext('2d');
  return new Chart(ctx, {
    type: 'line',
    data: { datasets: [
      { label: 'actual', borderColor: '#0071e3', backgroundColor: 'transparent',
        borderWidth: 2, pointRadius: 0, data: [] },
      { label: 'predicted', borderColor: '#ff9f0a', backgroundColor: 'transparent',
        borderWidth: 2, borderDash: [6, 4], pointRadius: 0, data: [] },
    ]},
    options: {
      animation: false,
      parsing: false,
      scales: {
        x: { type: 'linear', title: { display: true, text: 'time (s)' } },
        y: { title: { display: true, text: yLabel } },
      },
      plugins: { legend: { display: true, labels: { boxWidth: 20 } } },
    },
  });
}

window.addEventListener('DOMContentLoaded', () => {
  charts.pos = makeChart('posChart', 'turns');
  charts.vel = makeChart('velChart', 'turns/s');
  charts.acc = makeChart('accChart', 'turns/s^2');
  refreshShotList();
});

function toPoints(rows, key) {
  return rows.map(r => ({ x: r.t, y: r[key] }));
}

function updateChart(chart, actualRows, predictedRows, key) {
  chart.data.datasets[0].data = toPoints(actualRows, key);
  chart.data.datasets[1].data = toPoints(predictedRows, key);
  chart.update();
}

function renderShot(data, statusPrefix) {
  updateChart(charts.pos, data.actual, data.predicted, 'pos');
  updateChart(charts.vel, data.actual, data.predicted, 'vel');
  updateChart(charts.acc, data.actual, data.predicted, 'acc');
  const msg = document.getElementById('message');
  msg.textContent = `${statusPrefix} vel_limit=${data.params.vel_limit.toFixed(3)} turns/s - `
    + `start_pos=${data.params.start_pos.toFixed(4)}, `
    + `final_target=${data.params.final_target.toFixed(4)} turns.`;
  msg.className = '';
}

async function refreshShotList(selectId) {
  const sel = document.getElementById('shotSelect');
  try {
    const resp = await fetch('/api/shots');
    const data = await resp.json();
    sel.innerHTML = '';
    if (data.shots.length === 0) {
      sel.innerHTML = '<option value="">(none saved yet)</option>';
      return;
    }
    for (const s of data.shots) {
      const opt = document.createElement('option');
      opt.value = s.id;
      opt.textContent = `${s.id}  (vel_limit=${s.vel_limit.toFixed(3)})`;
      sel.appendChild(opt);
    }
    if (selectId) sel.value = selectId;
  } catch (e) {
    // shot list is a convenience, not worth surfacing a hard error for
  }
}

async function loadSelectedShot() {
  const sel = document.getElementById('shotSelect');
  if (!sel.value) return;
  const resp = await fetch('/api/shots/' + encodeURIComponent(sel.value));
  const data = await resp.json();
  if (data.ok) renderShot(data, 'Reviewing');
}

async function fire() {
  const btn = document.getElementById('fireBtn');
  const msg = document.getElementById('message');
  const vel = parseFloat(document.getElementById('vel').value);
  if (!isFinite(vel) || vel <= 0) {
    msg.textContent = 'Enter a positive vel_limit first.';
    msg.className = 'error';
    return;
  }

  btn.disabled = true;
  msg.textContent = 'Firing...';
  msg.className = '';

  try {
    const resp = await fetch('/api/fire', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ vel_limit: vel }),
    });
    const data = await resp.json();
    if (!data.ok) {
      msg.textContent = data.message;
      msg.className = 'error';
      return;
    }
    renderShot(data, 'Fired at');
    await refreshShotList(data.id);
  } catch (e) {
    msg.textContent = 'Request failed: ' + e;
    msg.className = 'error';
  } finally {
    btn.disabled = false;
  }
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
