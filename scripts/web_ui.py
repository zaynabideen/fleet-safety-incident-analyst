#!/usr/bin/env python3
"""
Local browser UI for testing the Fleet Safety Agent 1 -> Agent 2 pipeline,
so you don't have to edit JSON files and re-run a script to see a result.

Two tabs:
  - Run Pipeline: paste/edit incident JSON, run it, see the trace + result.
  - Dashboard: every run you've done (this session and previous ones -- it's
    persisted to data/pipeline_runs.json) as KPI tiles, a risk-score trend
    chart, and a clickable history table.

This is a thin presentation layer over the existing orchestrator -- it
constructs the same agents and calls build_fleet_safety_pipeline() /
orchestrator.run() exactly as scripts/run_pipeline.py does, then records
each result. No agent, schema, or orchestrator code is touched or
duplicated by this file.

Uses only Python's standard library (http.server, json) -- no Flask/
FastAPI, no new dependency to install, no database. History is one JSON
file; a local test tool doesn't need more than that.

Usage:
    python scripts/web_ui.py                # mock backends (default), opens a browser tab
    python scripts/web_ui.py --live          # real Anthropic API (needs ANTHROPIC_API_KEY)
    python scripts/web_ui.py --port 9000     # use a different port
    python scripts/web_ui.py --no-browser    # don't auto-open a tab (just print the URL)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import uuid
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fleet_safety.agents.driver_risk_analyst import DriverRiskAnalyst
from fleet_safety.agents.incident_analyst import FleetSafetyIncidentAnalyst
from fleet_safety.orchestration.fleet_pipeline import FleetPipelineInput, build_fleet_safety_pipeline
from fleet_safety.orchestration.orchestrator import Orchestrator
from fleet_safety.orchestration.types import AgentStatus, PipelineResult

DEMO_INCIDENTS = [
    {
        "incident_id": "INC-D1", "event_type": "harsh_braking", "vehicle_speed": 35,
        "following_distance": 2.2, "road_condition": "dry", "location_type": "residential street",
        "description": "Minor harsh braking event.", "timestamp": "2026-08-01T08:00:00",
    },
    {
        "incident_id": "INC-D2", "event_type": "harsh_braking", "vehicle_speed": 42,
        "following_distance": 1.1, "road_condition": "wet", "location_type": "highway",
        "description": "Harsh braking during highway driving, vehicle ahead stopped suddenly.",
        "timestamp": "2026-08-14T08:00:00",
    },
    {
        "incident_id": "INC-D3", "event_type": "harsh_braking", "vehicle_speed": 50,
        "following_distance": 0.7, "road_condition": "dry", "location_type": "highway",
        "description": "Harsh braking during highway driving.", "timestamp": "2026-08-25T08:00:00",
    },
    {
        "incident_id": "INC-D4", "event_type": "harsh_braking", "vehicle_speed": 45,
        "following_distance": 0.6, "road_condition": "dry", "location_type": "highway",
        "description": "Harsh braking during highway driving.", "timestamp": "2026-08-29T08:00:00",
    },
]
DEMO_DRIVER_ID = "DEMO-102"
DEMO_TIME_WINDOW_DAYS = 30

HISTORY_PATH = Path(__file__).resolve().parent.parent / "data" / "pipeline_runs.json"
_history_lock = threading.Lock()


# ---------------------------------------------------------------------------
# History persistence -- one JSON file, guarded by a lock (ThreadingHTTPServer
# handles requests concurrently). Deliberately not a database: this is a
# local test tool, and a run log rarely exceeds a few hundred entries.
# ---------------------------------------------------------------------------

def _load_history() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    try:
        return json.loads(HISTORY_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def _save_history(history: list[dict]) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(history, indent=2))


def _record_run(pipeline_input: FleetPipelineInput, response: dict) -> dict:
    record = {
        "id": f"run-{uuid.uuid4().hex[:8]}",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "driver_id": pipeline_input.driver_id,
        "time_window_days": pipeline_input.time_window_days,
        "incident_count": len(pipeline_input.raw_incidents),
        "response": response,
    }
    with _history_lock:
        history = _load_history()
        history.append(record)
        _save_history(history)
    return record


def _history_summary(record: dict) -> dict:
    response = record["response"]
    result = response.get("result")
    return {
        "id": record["id"],
        "timestamp": record["timestamp"],
        "driver_id": record["driver_id"],
        "incident_count": record["incident_count"],
        "status": response["status"],
        "failed_at": response.get("failed_at"),
        "risk_score": result["risk_score"] if result else None,
        "risk_level": result["risk_level"] if result else None,
        "trend": result["trend"] if result else None,
        "requires_immediate_attention": result["requires_immediate_attention"] if result else None,
    }


def _compute_stats(history: list[dict]) -> dict:
    total_runs = len(history)
    driver_ids = {r["driver_id"] for r in history}
    successful = [r for r in history if r["response"]["status"] == "SUCCESS"]
    failed_count = total_runs - len(successful)
    scores = [r["response"]["result"]["risk_score"] for r in successful]
    avg_risk_score = round(sum(scores) / len(scores), 1) if scores else None
    high_critical = sum(
        1 for r in successful if r["response"]["result"]["risk_level"] in ("HIGH", "CRITICAL")
    )
    level_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    for r in successful:
        level_counts[r["response"]["result"]["risk_level"]] += 1

    return {
        "total_runs": total_runs,
        "unique_drivers": len(driver_ids),
        "avg_risk_score": avg_risk_score,
        "high_critical_count": high_critical,
        "failed_count": failed_count,
        "level_counts": level_counts,
    }


# ---------------------------------------------------------------------------
# Serialization: PipelineResult -> plain JSON for the browser to render.
# Presentation-only -- reads the orchestrator's output, never changes it.
# ---------------------------------------------------------------------------

def _summarize_output(agent_name: str, output) -> str:
    if agent_name == "incident_analyst":
        return (
            f"{output.event_type} | severity={output.severity.value} | "
            f"driver_contribution={output.driver_contribution.level.value} | "
            f"root_cause={output.root_cause.cause}"
        )
    if agent_name == "driver_risk_analyst":
        return f"risk={output.risk_level.value} ({output.risk_score}) | trend={output.trend.value}"
    return str(output)


def _serialize_result(result: PipelineResult) -> dict:
    trace = []
    for entry in result.trace:
        item = {
            "agent_name": entry.agent_name,
            "status": entry.status.value,
            "item_ref": entry.item_ref,
            "duration_ms": round(entry.duration_ms, 2) if entry.duration_ms is not None else None,
        }
        if entry.status == AgentStatus.FAILED:
            item["error_type"] = entry.error_type
            item["error"] = entry.error
        else:
            item["summary"] = _summarize_output(entry.agent_name, entry.output)
        trace.append(item)

    payload = {"status": result.status.value, "failed_at": result.failed_at, "trace": trace, "result": None}

    if result.status == AgentStatus.SUCCESS:
        driver_risk = result.result
        payload["result"] = {
            "driver_id": driver_risk.driver_id,
            "time_window_days": driver_risk.time_window_days,
            "total_incidents": driver_risk.total_incidents,
            "risk_score": driver_risk.risk_score,
            "risk_level": driver_risk.risk_level.value,
            "trend": driver_risk.trend.value,
            "primary_concern": driver_risk.primary_concern,
            "requires_immediate_attention": driver_risk.requires_immediate_attention,
            "confidence": driver_risk.confidence,
            "recurring_patterns": [
                {
                    "pattern": p.pattern, "occurrences": p.occurrences,
                    "trend": p.trend.value, "explanation": p.explanation,
                }
                for p in driver_risk.recurring_patterns
            ],
            "recommended_focus_areas": driver_risk.recommended_focus_areas,
            "evidence": driver_risk.evidence,
            "limitations": driver_risk.limitations,
        }
    return payload


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

_HISTORY_ITEM_RE = re.compile(r"^/history/([A-Za-z0-9\-]+)$")


def make_handler(orchestrator: Orchestrator, mode_label: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002 - stdlib signature
            pass  # keep the terminal quiet; the browser is the UI

        def do_GET(self):
            if self.path == "/":
                self._send_html(INDEX_HTML)
            elif self.path == "/demo":
                self._send_json({
                    "driver_id": DEMO_DRIVER_ID,
                    "time_window_days": DEMO_TIME_WINDOW_DAYS,
                    "incidents": DEMO_INCIDENTS,
                })
            elif self.path == "/mode":
                self._send_json({"mode": mode_label})
            elif self.path == "/stats":
                with _history_lock:
                    history = _load_history()
                self._send_json(_compute_stats(history))
            elif self.path == "/history":
                with _history_lock:
                    history = _load_history()
                summaries = [_history_summary(r) for r in reversed(history)]
                self._send_json({"runs": summaries})
            elif _HISTORY_ITEM_RE.match(self.path):
                run_id = _HISTORY_ITEM_RE.match(self.path).group(1)
                with _history_lock:
                    history = _load_history()
                record = next((r for r in history if r["id"] == run_id), None)
                if record is None:
                    self.send_error(404)
                    return
                self._send_json(record["response"])
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path == "/run":
                self._handle_run()
            elif self.path == "/history/clear":
                with _history_lock:
                    _save_history([])
                self._send_json({"cleared": True})
            else:
                self.send_error(404)

        def _handle_run(self):
            length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(length)

            try:
                payload = json.loads(raw_body)
                pipeline_input = FleetPipelineInput(
                    driver_id=payload["driver_id"],
                    time_window_days=payload.get("time_window_days", 30),
                    raw_incidents=payload["incidents"],
                )
            except Exception as e:
                self._send_json({"error": f"Invalid input: {e}"}, status=400)
                return

            result = orchestrator.run(pipeline_input)
            response = _serialize_result(result)
            record = _record_run(pipeline_input, response)
            response["run_id"] = record["id"]
            self._send_json(response)

        def _send_html(self, html: str):
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, data: dict, status: int = 200):
            body = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


# ---------------------------------------------------------------------------
# Page (single file, no external requests -- works fully offline).
#
# Design: a deliberately dark "ops console" -- the visual language fleet
# monitoring centers actually use (low-glare, always-on displays). This is a
# committed single theme, not a light/dark toggle. Data colors still follow
# the project's data-viz palette exactly (see the dataviz skill's
# references/palette.md): status colors (good/warning/serious/critical) are
# reserved for risk level and pipeline status and always paired with a text
# label, never color alone; the trend chart and the risk gauge use the
# palette's single sequential/status hues, never a gradient or rainbow.
# Amber is a separate, non-data role: it marks anything the user can act on
# (buttons, the active tab, focus rings) and is never used to encode a value.
# ---------------------------------------------------------------------------

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fleet Safety Pipeline -- Test Console</title>
<style>
  :root {
    --void: #0a0b0d; --panel: #14161b; --panel-raised: #1b1e25; --panel-sunken: #0f1114;
    --hairline: rgba(255,255,255,.08); --hairline-strong: rgba(255,255,255,.16);
    --ink: #f3f4f6; --ink-dim: #98a0ac; --ink-faint: #5b6270;
    --amber: #f2a93b; --amber-ink: #1a1206; --amber-bg: rgba(242,169,59,.13);
    --blue: #3987e5;
    --good: #0ca30c; --warning: #fab219; --serious: #ec835a; --critical: #d03b3b;
    --good-bg: rgba(12,163,12,.14); --warning-bg: rgba(250,178,25,.14);
    --serious-bg: rgba(236,131,90,.14); --critical-bg: rgba(208,59,59,.16);
    --grid: rgba(255,255,255,.07);
    --mono: ui-monospace, "Cascadia Code", "SFMono-Regular", Menlo, Consolas, "Liberation Mono", monospace;
    --sans: system-ui, -apple-system, "Segoe UI", sans-serif;
    --radius: 12px;
  }
  * { box-sizing: border-box; }
  html { color-scheme: dark; }
  body {
    margin: 0; padding: 28px 20px 60px; background: var(--void); color: var(--ink);
    font-family: var(--sans); line-height: 1.5; -webkit-font-smoothing: antialiased;
  }
  body::before {
    content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background: radial-gradient(1100px 480px at 50% -8%, rgba(57,135,229,.10), transparent 60%);
  }
  .shell { max-width: 1040px; margin: 0 auto; position: relative; z-index: 1; }
  a { color: var(--amber); }
  :focus-visible { outline: 2px solid var(--amber); outline-offset: 2px; border-radius: 4px; }

  /* ---------- Topbar ---------- */
  .topbar { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 22px; flex-wrap: wrap; }
  .brand { display: flex; align-items: center; gap: 10px; }
  .pulse-dot {
    width: 9px; height: 9px; border-radius: 50%; background: var(--good); flex-shrink: 0;
    animation: pulseDot 2.2s ease-out infinite;
  }
  @keyframes pulseDot {
    0% { box-shadow: 0 0 0 0 rgba(12,163,12,.55); }
    70% { box-shadow: 0 0 0 9px rgba(12,163,12,0); }
    100% { box-shadow: 0 0 0 0 rgba(12,163,12,0); }
  }
  .brand-text { font-family: var(--mono); font-size: 15px; font-weight: 700; letter-spacing: .02em; }
  .brand-sub { color: var(--ink-faint); font-weight: 500; }
  .mode-tag {
    font-family: var(--mono); font-size: 11px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase;
    padding: 5px 10px; border-radius: 6px; border: 1px solid var(--hairline-strong); background: var(--panel);
    color: var(--ink-dim);
  }
  .mode-tag.live { color: var(--amber); border-color: rgba(242,169,59,.4); background: var(--amber-bg); }

  /* ---------- Tabs ---------- */
  .tabs { position: relative; display: flex; gap: 26px; margin-bottom: 20px; border-bottom: 1px solid var(--hairline); }
  .tab-btn {
    background: none; border: none; padding: 10px 2px 12px; font-size: 13.5px; font-weight: 600;
    font-family: var(--sans); letter-spacing: .01em; color: var(--ink-faint); cursor: pointer;
  }
  .tab-btn.active { color: var(--ink); }
  .tab-btn:hover { color: var(--ink); }
  .tab-indicator { position: absolute; bottom: -1px; height: 2px; background: var(--amber); border-radius: 2px; transition: left .25s ease, width .25s ease; }

  .tab-panel[hidden] { display: none; }

  /* ---------- Panels / cards ---------- */
  .panel {
    background: var(--panel); border: 1px solid var(--hairline); border-radius: var(--radius);
    padding: 20px; margin-bottom: 16px; position: relative; overflow: hidden;
  }
  .eyebrow { font-family: var(--mono); font-size: 10.5px; font-weight: 700; text-transform: uppercase;
    letter-spacing: .09em; color: var(--ink-faint); margin: 0 0 14px; }
  .panel-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; gap: 12px; flex-wrap: wrap; }
  .panel-head .eyebrow { margin: 0; }

  /* ---------- Run tab layout ---------- */
  .run-grid { display: grid; grid-template-columns: 380px 1fr; gap: 18px; align-items: start; }
  @media (max-width: 860px) { .run-grid { grid-template-columns: 1fr; } }

  label { display: block; font-size: 11px; font-weight: 700; color: var(--ink-faint);
    text-transform: uppercase; letter-spacing: .05em; margin-bottom: 6px; }
  .row { display: flex; gap: 12px; margin-bottom: 16px; }
  .row .field { flex: 1; }
  input[type=text], input[type=number] {
    width: 100%; padding: 9px 10px; border: 1px solid var(--hairline-strong); border-radius: 7px;
    font-size: 13.5px; font-family: var(--mono); background: var(--panel-sunken); color: var(--ink);
  }
  input[type=text]:focus, input[type=number]:focus, textarea:focus { border-color: var(--amber); }
  textarea {
    width: 100%; min-height: 300px; padding: 12px; border: 1px solid var(--hairline-strong);
    border-radius: 7px; font-family: var(--mono); font-size: 12px; line-height: 1.55; resize: vertical;
    background: var(--panel-sunken); color: var(--ink);
  }
  .actions { display: flex; gap: 10px; margin-top: 14px; align-items: center; flex-wrap: wrap; }
  button {
    border: none; border-radius: 7px; padding: 10px 16px; font-size: 13.5px; font-weight: 700;
    cursor: pointer; font-family: var(--sans); letter-spacing: .01em;
    display: inline-flex; align-items: center; gap: 8px;
  }
  button.primary { background: var(--amber); color: var(--amber-ink); }
  button.primary:hover:not(:disabled) { filter: brightness(1.06); }
  button.primary:disabled { background: var(--panel-raised); color: var(--ink-faint); cursor: default; }
  button.secondary { background: var(--panel-raised); color: var(--ink); border: 1px solid var(--hairline-strong); }
  button.secondary:hover { border-color: var(--ink-faint); }
  .status-line { font-size: 12.5px; color: var(--ink-dim); }
  .spinner {
    width: 12px; height: 12px; border-radius: 50%; border: 2px solid rgba(26,18,6,.3);
    border-top-color: var(--amber-ink); animation: spin .7s linear infinite; display: none;
  }
  button.primary.loading .spinner { display: inline-block; }
  @keyframes spin { to { transform: rotate(360deg); } }

  details.hint { font-size: 12px; color: var(--ink-dim); margin-top: 10px; }
  details.hint summary { cursor: pointer; font-weight: 600; color: var(--ink-dim); }
  details.hint summary:hover { color: var(--ink); }
  details.hint code, code {
    background: var(--panel-raised); color: var(--ink); padding: 1px 5px; border-radius: 4px;
    font-family: var(--mono); font-size: .92em; border: 1px solid var(--hairline);
  }

  /* ---------- Trace output ---------- */
  .idle-state { padding: 40px 10px; text-align: center; color: var(--ink-faint); font-size: 13px; }
  .idle-state .idle-dot {
    width: 8px; height: 8px; border-radius: 50%; background: var(--ink-faint); margin: 0 auto 12px;
    opacity: .5;
  }

  .pipeline-status {
    display: inline-flex; align-items: center; gap: 6px; font-family: var(--mono); font-weight: 700;
    padding: 4px 10px; border-radius: 6px; font-size: 12px; letter-spacing: .03em;
  }
  .pipeline-status::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
  .pipeline-status.SUCCESS { background: var(--good-bg); color: var(--good); }
  .pipeline-status.FAILED { background: var(--critical-bg); color: var(--critical); }

  .trace-log { margin-top: 14px; }
  .trace-item {
    display: flex; align-items: flex-start; gap: 10px; padding: 10px 0;
    border-bottom: 1px solid var(--hairline); font-size: 13px;
    animation: traceIn .3s ease both;
  }
  .trace-item:last-child { border-bottom: none; }
  @keyframes traceIn { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }
  .pill {
    flex-shrink: 0; width: 40px; text-align: center; font-family: var(--mono); font-size: 10.5px; font-weight: 700;
    padding: 3px 0; border-radius: 5px; letter-spacing: .02em;
  }
  .pill.SUCCESS { background: var(--good-bg); color: var(--good); }
  .pill.FAILED { background: var(--critical-bg); color: var(--critical); }
  .trace-main { flex: 1; min-width: 0; }
  .trace-name { font-family: var(--mono); font-weight: 700; font-size: 12.5px; }
  .trace-ref { color: var(--ink-faint); font-weight: 400; font-family: var(--mono); }
  .trace-detail { color: var(--ink-dim); margin-top: 3px; }
  .trace-detail.error { color: var(--critical); font-family: var(--mono); font-size: 11.5px; }
  .trace-duration { color: var(--ink-faint); font-family: var(--mono); font-size: 11px; white-space: nowrap; font-variant-numeric: tabular-nums; }

  .error-banner {
    background: var(--critical-bg); color: var(--critical); padding: 11px 14px; border-radius: 8px;
    font-size: 12.5px; margin-top: 16px; border: 1px solid rgba(208,59,59,.3);
  }

  /* ---------- Risk gauge (signature element) ---------- */
  .verdict { margin-top: 22px; padding-top: 20px; border-top: 1px solid var(--hairline); }
  .gauge-wrap { position: relative; max-width: 280px; margin: 0 auto 6px; }
  .gauge-svg { width: 100%; height: auto; display: block; }
  .gauge-track { fill: none; stroke: var(--hairline-strong); stroke-width: 13; stroke-linecap: round; }
  .gauge-fill { fill: none; stroke-width: 13; stroke-linecap: round; transition: stroke-dasharray .7s cubic-bezier(.2,.8,.2,1); }
  .gauge-readout { position: absolute; left: 50%; bottom: 4%; transform: translateX(-50%); text-align: center; }
  .gauge-score { font-family: var(--mono); font-size: 38px; font-weight: 700; line-height: 1; font-variant-numeric: tabular-nums; }
  .gauge-outof { font-size: 12px; color: var(--ink-faint); font-family: var(--mono); }
  .gauge-level { margin-top: 6px; font-size: 11.5px; font-weight: 700; letter-spacing: .05em; text-transform: uppercase;
    display: inline-flex; align-items: center; gap: 6px; }

  .attention-banner {
    display: flex; align-items: center; gap: 8px; justify-content: center; margin: 14px auto 0; max-width: 420px;
    padding: 8px 14px; border-radius: 999px; font-size: 12px; font-weight: 700; letter-spacing: .02em;
  }
  .attention-banner.critical { background: var(--critical-bg); color: var(--critical); }
  .attention-banner.clear { background: var(--panel-raised); color: var(--ink-dim); font-weight: 600; }
  .attention-banner .dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
  .attention-banner.critical .dot { animation: attentionBlink 1.3s ease-in-out infinite; }
  @keyframes attentionBlink { 0%, 100% { opacity: 1; } 50% { opacity: .35; } }

  .result-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px 20px; margin: 20px 0 4px; }
  @media (max-width: 560px) { .result-grid { grid-template-columns: repeat(2, 1fr); } }
  .result-grid .k { font-size: 10.5px; color: var(--ink-faint); text-transform: uppercase; letter-spacing: .04em; margin-bottom: 3px; }
  .result-grid .v { font-size: 14px; font-weight: 700; font-family: var(--mono); }

  .tags { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 14px; }
  .tag {
    font-size: 11.5px; padding: 4px 10px; border-radius: 999px; background: var(--panel-raised);
    border: 1px solid var(--hairline-strong); color: var(--ink-dim);
  }

  .patterns { margin-top: 14px; }
  .pattern { padding: 10px 12px; background: var(--panel-sunken); border: 1px solid var(--hairline);
    border-radius: 8px; margin-bottom: 8px; font-size: 12.5px; }
  .pattern .head { font-weight: 700; font-family: var(--mono); font-size: 12px; margin-bottom: 2px; }
  .pattern .head .occ { color: var(--ink-faint); font-weight: 500; }
  .pattern .body { color: var(--ink-dim); }

  details.evidence { margin-top: 16px; font-size: 12.5px; }
  details.evidence summary {
    cursor: pointer; font-weight: 700; color: var(--ink-dim); font-family: var(--mono); font-size: 11px;
    text-transform: uppercase; letter-spacing: .05em;
  }
  details.evidence summary:hover { color: var(--ink); }
  .evidence-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 12px; }
  @media (max-width: 560px) { .evidence-cols { grid-template-columns: 1fr; } }
  .evidence-cols h4 { font-size: 10.5px; text-transform: uppercase; letter-spacing: .04em; color: var(--ink-faint); margin: 0 0 6px; }
  .evidence-cols ul { margin: 0; padding-left: 16px; color: var(--ink-dim); }
  .evidence-cols li { margin-bottom: 4px; }

  /* ---------- Dashboard ---------- */
  .kpi-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px; }
  @media (max-width: 700px) { .kpi-row { grid-template-columns: repeat(2, 1fr); } }
  .stat-tile {
    background: var(--panel); border: 1px solid var(--hairline); border-radius: var(--radius); padding: 16px;
    position: relative; overflow: hidden;
  }
  .stat-tile::before { content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: var(--tile-accent, var(--ink-faint)); }
  .stat-label { font-size: 10.5px; color: var(--ink-faint); text-transform: uppercase; letter-spacing: .05em; margin-bottom: 8px; }
  .stat-value { font-size: 27px; font-weight: 700; font-family: var(--mono); font-variant-numeric: tabular-nums; }
  .stat-value.alert { color: var(--critical); }

  .dash-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 10px; }
  .dash-header h2 { font-size: 12px; margin: 0; font-weight: 700; text-transform: uppercase; letter-spacing: .05em; color: var(--ink-dim); }

  .trend-svg { width: 100%; height: auto; display: block; }
  .grid-line { stroke: var(--grid); stroke-width: 1; }
  .axis-label { fill: var(--ink-faint); font-size: 10px; font-family: var(--mono); }
  .trend-line { fill: none; stroke: var(--blue); stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
  .trend-dot { fill: var(--blue); }
  .trend-dot-ring { stroke: var(--panel); stroke-width: 2; }
  .trend-endlabel { fill: var(--ink); font-size: 12px; font-weight: 700; font-family: var(--mono); }
  .empty-state { color: var(--ink-faint); font-size: 12.5px; padding: 30px 0; text-align: center; }

  .table-scroll { overflow-x: auto; }
  table.history-table { width: 100%; border-collapse: collapse; font-size: 12.5px; min-width: 640px; }
  table.history-table th {
    text-align: left; font-size: 10.5px; text-transform: uppercase; letter-spacing: .04em;
    color: var(--ink-faint); padding: 8px 10px; border-bottom: 1px solid var(--hairline); font-weight: 700;
  }
  table.history-table td { padding: 10px; border-bottom: 1px solid var(--hairline); }
  table.history-table td.mono { font-family: var(--mono); font-size: 12px; color: var(--ink-dim); }
  table.history-table tbody tr { cursor: pointer; transition: background .12s ease; }
  table.history-table tbody tr:hover { background: var(--panel-raised); }
  .level-pill { display: inline-flex; align-items: center; gap: 6px; font-weight: 700; font-size: 12px; font-family: var(--mono); }
  .level-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  .level-dot.LOW { background: var(--good); }
  .level-dot.MEDIUM { background: var(--warning); }
  .level-dot.HIGH { background: var(--serious); }
  .level-dot.CRITICAL { background: var(--critical); }
  .trend-arrow.INCREASING { color: var(--critical); }
  .trend-arrow.DECREASING { color: var(--good); }
  .trend-arrow.STABLE, .trend-arrow.INSUFFICIENT_DATA { color: var(--ink-faint); }
  .attention-flag { color: var(--critical); font-weight: 700; font-family: var(--mono); }
  #dashboardDetail { margin-top: 16px; }
  #dashboardDetail[hidden] { display: none; }
  #dashboardDetail .dash-header h2 { color: var(--ink); }

  @media (prefers-reduced-motion: reduce) {
    .pulse-dot, .trace-item, .attention-banner.critical .dot, .spinner { animation: none !important; }
    .gauge-fill, .tab-indicator { transition: none !important; }
  }
</style>
</head>
<body>
<div class="shell">
  <div class="topbar">
    <div class="brand">
      <span class="pulse-dot" aria-hidden="true"></span>
      <span class="brand-text">FLEET SAFETY <span class="brand-sub">// PIPELINE CONSOLE</span></span>
    </div>
    <div class="mode-tag" id="modeBadge">loading&hellip;</div>
  </div>

  <nav class="tabs" id="tabs">
    <button class="tab-btn active" id="tabBtnRun" type="button">Run Pipeline</button>
    <button class="tab-btn" id="tabBtnDash" type="button">Dashboard</button>
    <div class="tab-indicator" id="tabIndicator"></div>
  </nav>

  <div class="tab-panel" id="tabRun">
    <div class="run-grid">
      <div class="panel">
        <div class="eyebrow">Scenario input</div>
        <div class="row">
          <div class="field">
            <label for="driverId">Driver ID</label>
            <input type="text" id="driverId" value="DEMO-102">
          </div>
          <div class="field" style="max-width:150px;">
            <label for="windowDays">Window (days)</label>
            <input type="number" id="windowDays" value="30" min="1">
          </div>
        </div>

        <label for="incidents">Raw incidents (JSON array)</label>
        <textarea id="incidents" spellcheck="false"></textarea>

        <details class="hint">
          <summary>Incident fields you can use</summary>
          <p>Only <code>incident_id</code> is required &mdash; leave anything else out if you
          don't have it, the agent is built to reason with partial evidence.</p>
          <p><code>incident_id</code>, <code>event_type</code>, <code>timestamp</code>,
          <code>description</code>, <code>vehicle_speed</code>, <code>speed_unit</code>
          (mph/kmh), <code>following_distance</code> (seconds), <code>weather</code>,
          <code>road_condition</code>, <code>location_type</code>, <code>visibility</code>,
          <code>traffic_conditions</code>, <code>visual_observations</code>,
          <code>video_available</code> (true/false), <code>image_available</code> (true/false)</p>
        </details>

        <div class="actions">
          <button class="primary" id="runBtn" type="button"><span class="spinner"></span><span>Run Pipeline</span></button>
          <button class="secondary" id="demoBtn" type="button">Load demo data</button>
        </div>
        <div class="status-line" id="statusLine" style="margin-top:8px;"></div>
      </div>

      <div class="panel">
        <div class="eyebrow">Agent trace &amp; verdict</div>
        <div id="traceBody">
          <div class="idle-state">
            <div class="idle-dot"></div>
            Console idle &mdash; run the pipeline to watch Agent 1 &rarr; Agent 2 execute.
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="tab-panel" id="tabDash" hidden>
    <div class="kpi-row">
      <div class="stat-tile" style="--tile-accent: var(--amber);"><div class="stat-label">Total runs</div><div class="stat-value" id="statTotalRuns">&ndash;</div></div>
      <div class="stat-tile" style="--tile-accent: var(--ink-faint);"><div class="stat-label">Drivers tracked</div><div class="stat-value" id="statDrivers">&ndash;</div></div>
      <div class="stat-tile" style="--tile-accent: var(--blue);"><div class="stat-label">Avg risk score</div><div class="stat-value" id="statAvgRisk">&ndash;</div></div>
      <div class="stat-tile" style="--tile-accent: var(--critical);"><div class="stat-label">High / Critical alerts</div><div class="stat-value alert" id="statAlerts">&ndash;</div></div>
    </div>

    <div class="panel">
      <div class="dash-header"><h2>Risk score trend &mdash; most recent runs</h2></div>
      <div id="trendChart"></div>
    </div>

    <div class="panel">
      <div class="dash-header">
        <h2>Run history</h2>
        <div class="actions" style="margin:0;">
          <button class="secondary" id="refreshBtn" type="button">Refresh</button>
          <button class="secondary" id="clearBtn" type="button">Clear history</button>
        </div>
      </div>
      <div class="table-scroll">
        <table class="history-table">
          <thead>
            <tr><th>Time</th><th>Driver</th><th>Status</th><th>Incidents</th><th>Risk</th><th>Trend</th><th>Attention</th></tr>
          </thead>
          <tbody id="historyBody"></tbody>
        </table>
      </div>
      <div class="empty-state" id="historyEmpty" hidden>No runs yet &mdash; use the Run Pipeline tab to test a scenario.</div>
    </div>

    <div class="panel" id="dashboardDetail" hidden></div>
  </div>
</div>

<script>
const driverIdEl = document.getElementById('driverId');
const windowDaysEl = document.getElementById('windowDays');
const incidentsEl = document.getElementById('incidents');
const runBtn = document.getElementById('runBtn');
const demoBtn = document.getElementById('demoBtn');
const statusLine = document.getElementById('statusLine');
const traceBody = document.getElementById('traceBody');
const modeBadge = document.getElementById('modeBadge');

const tabBtnRun = document.getElementById('tabBtnRun');
const tabBtnDash = document.getElementById('tabBtnDash');
const tabRun = document.getElementById('tabRun');
const tabDash = document.getElementById('tabDash');
const tabIndicator = document.getElementById('tabIndicator');
const dashboardDetail = document.getElementById('dashboardDetail');

function moveIndicator(btn) {
  tabIndicator.style.left = btn.offsetLeft + 'px';
  tabIndicator.style.width = btn.offsetWidth + 'px';
}

function showTab(name) {
  const runActive = name === 'run';
  tabRun.hidden = !runActive;
  tabDash.hidden = runActive;
  tabBtnRun.classList.toggle('active', runActive);
  tabBtnDash.classList.toggle('active', !runActive);
  moveIndicator(runActive ? tabBtnRun : tabBtnDash);
  if (!runActive) refreshDashboard();
}
tabBtnRun.addEventListener('click', () => showTab('run'));
tabBtnDash.addEventListener('click', () => showTab('dash'));
window.addEventListener('resize', () => moveIndicator(tabBtnRun.classList.contains('active') ? tabBtnRun : tabBtnDash));

async function loadDemo() {
  const res = await fetch('/demo');
  const demo = await res.json();
  driverIdEl.value = demo.driver_id;
  windowDaysEl.value = demo.time_window_days;
  incidentsEl.value = JSON.stringify(demo.incidents, null, 2);
}

async function loadMode() {
  try {
    const res = await fetch('/mode');
    const data = await res.json();
    modeBadge.textContent = data.mode;
    modeBadge.classList.toggle('live', data.mode.toUpperCase().startsWith('LIVE'));
  } catch (e) {
    modeBadge.textContent = 'unknown';
  }
}

function pillHtml(status) {
  return `<div class="pill ${status}">${status === 'SUCCESS' ? 'OK' : 'FAIL'}</div>`;
}

const LEVEL_COLOR = { LOW: 'var(--good)', MEDIUM: 'var(--warning)', HIGH: 'var(--serious)', CRITICAL: 'var(--critical)' };

function gaugeHtml(score, level) {
  const path = 'M 26 96 A 74 74 0 0 1 174 96';
  const arcLen = Math.PI * 74;
  const clamped = Math.max(0, Math.min(100, score));
  const filled = arcLen * clamped / 100;
  const color = LEVEL_COLOR[level] || 'var(--ink-faint)';
  return `<div class="gauge-wrap">
    <svg viewBox="0 0 200 112" class="gauge-svg" role="img" aria-label="Risk score ${score} out of 100, ${level || 'unknown'}">
      <path d="${path}" class="gauge-track"/>
      <path d="${path}" class="gauge-fill" style="stroke:${color}; stroke-dasharray:${filled.toFixed(1)} ${(arcLen + 4).toFixed(1)};"/>
    </svg>
    <div class="gauge-readout">
      <div class="gauge-score">${score}<span class="gauge-outof">/100</span></div>
      <div class="gauge-level" style="color:${color};"><span class="level-dot ${level}"></span>${level || 'UNKNOWN'}</div>
    </div>
  </div>`;
}

function traceHtml(data) {
  let html = `<div>
    Pipeline status: <span class="pipeline-status ${data.status}">${data.status}</span>
  </div><div class="trace-log">`;
  data.trace.forEach((entry, i) => {
    const ref = entry.item_ref ? `<span class="trace-ref"> [${entry.item_ref}]</span>` : '';
    const duration = entry.duration_ms !== null ? `${entry.duration_ms}ms` : '';
    let detail = '';
    if (entry.status === 'FAILED') {
      detail = `<div class="trace-detail error">${entry.error_type}: ${entry.error}</div>`;
    } else if (entry.summary) {
      detail = `<div class="trace-detail">${entry.summary}</div>`;
    }
    html += `<div class="trace-item" style="animation-delay:${Math.min(i * 35, 400)}ms;">
      ${pillHtml(entry.status)}
      <div class="trace-main">
        <div><span class="trace-name">${entry.agent_name}</span>${ref}</div>
        ${detail}
      </div>
      <div class="trace-duration">${duration}</div>
    </div>`;
  });
  html += '</div>';

  if (data.status === 'FAILED') {
    html += `<div class="error-banner">Pipeline halted at stage &lsquo;${data.failed_at}&rsquo;
      &mdash; no downstream stage ran, no final result.</div>`;
    return html;
  }

  const r = data.result;
  html += `<div class="verdict">
    ${gaugeHtml(r.risk_score, r.risk_level)}
    ${r.requires_immediate_attention
      ? '<div class="attention-banner critical"><span class="dot"></span>Requires immediate attention</div>'
      : '<div class="attention-banner clear"><span class="dot"></span>No immediate attention required</div>'}
    <div class="result-grid">
      <div><div class="k">Total incidents</div><div class="v">${r.total_incidents}</div></div>
      <div><div class="k">Trend</div><div class="v">${r.trend}</div></div>
      <div><div class="k">Confidence</div><div class="v">${r.confidence}</div></div>
      <div><div class="k">Primary concern</div><div class="v">${r.primary_concern}</div></div>
      <div><div class="k">Time window</div><div class="v">${r.time_window_days}d</div></div>
      <div><div class="k">Driver</div><div class="v">${r.driver_id}</div></div>
    </div>`;

  if (r.recommended_focus_areas && r.recommended_focus_areas.length) {
    html += '<div class="tags">' + r.recommended_focus_areas.map(a => `<span class="tag">${a}</span>`).join('') + '</div>';
  }

  if (r.recurring_patterns.length) {
    html += '<div class="patterns">';
    for (const p of r.recurring_patterns) {
      html += `<div class="pattern">
        <div class="head">${p.pattern} <span class="occ">&mdash; ${p.occurrences}x, ${p.trend}</span></div>
        <div class="body">${p.explanation}</div>
      </div>`;
    }
    html += '</div>';
  }

  if ((r.evidence && r.evidence.length) || (r.limitations && r.limitations.length)) {
    html += `<details class="evidence"><summary>Supporting evidence &amp; limitations</summary>
      <div class="evidence-cols">
        <div><h4>Evidence</h4><ul>${(r.evidence || []).map(e => `<li>${e}</li>`).join('') || '<li>&ndash;</li>'}</ul></div>
        <div><h4>Limitations</h4><ul>${(r.limitations || []).map(l => `<li>${l}</li>`).join('') || '<li>&ndash;</li>'}</ul></div>
      </div>
    </details>`;
  }

  return html;
}

function renderResults(data) {
  traceBody.innerHTML = traceHtml(data);
}

async function runPipeline() {
  let incidents;
  try {
    incidents = JSON.parse(incidentsEl.value);
  } catch (e) {
    statusLine.textContent = 'Incidents field is not valid JSON: ' + e.message;
    statusLine.style.color = 'var(--critical)';
    return;
  }

  runBtn.disabled = true;
  runBtn.classList.add('loading');
  statusLine.style.color = 'var(--ink-dim)';
  statusLine.textContent = 'Running…';

  try {
    const res = await fetch('/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        driver_id: driverIdEl.value,
        time_window_days: parseInt(windowDaysEl.value, 10) || 30,
        incidents: incidents,
      }),
    });
    const data = await res.json();
    if (!res.ok) {
      statusLine.textContent = data.error || 'Request failed.';
      statusLine.style.color = 'var(--critical)';
      return;
    }
    statusLine.textContent = '';
    renderResults(data);
  } catch (e) {
    statusLine.textContent = 'Could not reach the pipeline server: ' + e.message;
    statusLine.style.color = 'var(--critical)';
  } finally {
    runBtn.disabled = false;
    runBtn.classList.remove('loading');
  }
}

runBtn.addEventListener('click', runPipeline);
demoBtn.addEventListener('click', loadDemo);

// ---------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------

function fmtTime(iso) {
  const d = new Date(iso);
  return d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

const TREND_ARROW = { INCREASING: '&uarr;', DECREASING: '&darr;', STABLE: '&rarr;', INSUFFICIENT_DATA: '&ndash;' };

function renderTrendChart(runs) {
  const container = document.getElementById('trendChart');
  const successful = runs.filter(r => r.status === 'SUCCESS' && r.risk_score !== null).slice(0, 10).reverse();

  if (successful.length === 0) {
    container.innerHTML = '<div class="empty-state">No successful runs yet &mdash; run the pipeline to see a trend here.</div>';
    return;
  }

  const W = 640, H = 200, padL = 32, padR = 16, padT = 16, padB = 30;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const n = successful.length;
  const xStep = n > 1 ? plotW / (n - 1) : 0;
  const yFor = (score) => padT + plotH - (score / 100) * plotH;
  const xFor = (i) => padL + i * xStep;

  let gridlines = '';
  [0, 25, 50, 75, 100].forEach(v => {
    const y = yFor(v);
    gridlines += `<line x1="${padL}" y1="${y.toFixed(1)}" x2="${W - padR}" y2="${y.toFixed(1)}" class="grid-line"/>`;
    gridlines += `<text x="${padL - 6}" y="${(y + 3).toFixed(1)}" class="axis-label" text-anchor="end">${v}</text>`;
  });

  const points = successful.map((r, i) => `${xFor(i).toFixed(1)},${yFor(r.risk_score).toFixed(1)}`).join(' ');

  let dots = '';
  successful.forEach((r, i) => {
    const x = xFor(i), y = yFor(r.risk_score);
    dots += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="7" fill="none" class="trend-dot-ring"/>
      <circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="4" class="trend-dot">
        <title>${r.driver_id}: ${r.risk_score} (${r.risk_level})</title>
      </circle>`;
  });

  const xLabels = successful.map((r, i) => {
    const x = xFor(i);
    const label = r.driver_id.length > 9 ? r.driver_id.slice(0, 8) + '…' : r.driver_id;
    return `<text x="${x.toFixed(1)}" y="${H - 8}" class="axis-label" text-anchor="middle">${label}</text>`;
  }).join('');

  const last = successful[n - 1];
  const lastX = xFor(n - 1), lastY = yFor(last.risk_score);

  container.innerHTML = `<svg viewBox="0 0 ${W} ${H}" class="trend-svg" role="img" aria-label="Risk score trend across recent runs">
    ${gridlines}
    <polyline points="${points}" class="trend-line"/>
    ${dots}
    <text x="${lastX.toFixed(1)}" y="${(lastY - 12).toFixed(1)}" class="trend-endlabel" text-anchor="middle">${last.risk_score}</text>
    ${xLabels}
  </svg>`;
}

function levelPillHtml(level) {
  if (!level) return '<span class="empty-state" style="padding:0;">&ndash;</span>';
  return `<span class="level-pill"><span class="level-dot ${level}"></span>${level}</span>`;
}

function renderHistoryTable(runs) {
  const body = document.getElementById('historyBody');
  const empty = document.getElementById('historyEmpty');
  if (runs.length === 0) {
    body.innerHTML = '';
    empty.hidden = false;
    return;
  }
  empty.hidden = true;
  body.innerHTML = runs.map(r => {
    const statusHtml = r.status === 'SUCCESS'
      ? '<span class="pipeline-status SUCCESS">SUCCESS</span>'
      : '<span class="pipeline-status FAILED">FAILED</span>';
    const trendHtml = r.trend
      ? `<span class="trend-arrow ${r.trend}">${TREND_ARROW[r.trend] || ''}</span> ${r.trend.replace('_', ' ')}`
      : '&ndash;';
    const attentionHtml = r.requires_immediate_attention ? '<span class="attention-flag">Yes</span>' : '&ndash;';
    return `<tr data-run-id="${r.id}" tabindex="0">
      <td class="mono">${fmtTime(r.timestamp)}</td>
      <td class="mono">${r.driver_id}</td>
      <td>${statusHtml}</td>
      <td class="mono">${r.incident_count}</td>
      <td>${r.risk_score !== null ? r.risk_score + ' ' : ''}${levelPillHtml(r.risk_level)}</td>
      <td>${trendHtml}</td>
      <td>${attentionHtml}</td>
    </tr>`;
  }).join('');

  body.querySelectorAll('tr').forEach(row => {
    row.addEventListener('click', () => showRunDetail(row.dataset.runId));
    row.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); showRunDetail(row.dataset.runId); }
    });
  });
}

async function showRunDetail(runId) {
  const res = await fetch(`/history/${runId}`);
  if (!res.ok) return;
  const data = await res.json();
  dashboardDetail.hidden = false;
  dashboardDetail.innerHTML = `<div class="dash-header"><h2>Run detail &mdash; ${runId}</h2></div>` + traceHtml(data);
  dashboardDetail.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function renderStats(stats) {
  document.getElementById('statTotalRuns').textContent = stats.total_runs;
  document.getElementById('statDrivers').textContent = stats.unique_drivers;
  document.getElementById('statAvgRisk').textContent = stats.avg_risk_score !== null ? stats.avg_risk_score : '–';
  document.getElementById('statAlerts').textContent = stats.high_critical_count;
}

async function refreshDashboard() {
  const [statsRes, historyRes] = await Promise.all([fetch('/stats'), fetch('/history')]);
  const stats = await statsRes.json();
  const history = await historyRes.json();
  renderStats(stats);
  renderTrendChart(history.runs);
  renderHistoryTable(history.runs);
}

document.getElementById('refreshBtn').addEventListener('click', refreshDashboard);
document.getElementById('clearBtn').addEventListener('click', async () => {
  if (!confirm('Clear all run history? This cannot be undone.')) return;
  await fetch('/history/clear', { method: 'POST' });
  dashboardDetail.hidden = true;
  refreshDashboard();
});

loadMode();
loadDemo();
moveIndicator(tabBtnRun);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true", help="Use the real Anthropic API for both agents.")
    parser.add_argument("--port", type=int, default=8765, help="Port to serve on (default 8765).")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open a browser tab.")
    args = parser.parse_args()

    if args.live:
        from fleet_safety.llm.anthropic_client import AnthropicLLMClient
        incident_agent = FleetSafetyIncidentAnalyst(AnthropicLLMClient())
        driver_agent = DriverRiskAnalyst(AnthropicLLMClient())
        mode_label = "LIVE (Anthropic API)"
    else:
        from fleet_safety.llm.mock_client import DriverRiskMockLLMClient, MockLLMClient
        incident_agent = FleetSafetyIncidentAnalyst(MockLLMClient())
        driver_agent = DriverRiskAnalyst(DriverRiskMockLLMClient())
        mode_label = "MOCK (offline)"

    orchestrator = build_fleet_safety_pipeline(incident_agent, driver_agent)
    handler_cls = make_handler(orchestrator, mode_label)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_cls)
    url = f"http://127.0.0.1:{args.port}"

    print(f"[mode] {mode_label}")
    print(f"Fleet Safety pipeline test console running at {url}")
    print(f"Run history is saved to {HISTORY_PATH}")
    print("Press Ctrl+C to stop.\n")

    if not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        server.shutdown()


if __name__ == "__main__":
    main()
