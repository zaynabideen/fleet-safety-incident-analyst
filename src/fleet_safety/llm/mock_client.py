"""
MockLLMClient — a deterministic, offline stand-in for a real LLM backend.

WHAT THIS IS: a rule engine that walks the same twelve-step investigation
the system prompt asks a real model to perform (identify event -> scene
context -> driver behaviour -> contributing factors -> facts vs inference
-> driver contribution -> severity -> confidence -> root cause -> evidence
-> recommended action -> human review flag), and emits IncidentOutput-
shaped JSON.

WHAT THIS IS NOT: a substitute for the reasoning quality of an actual LLM.
It cannot read images, infer from free-text nuance the way a language
model does, or handle a phrasing of the input it wasn't written to expect.
It exists so that:

  1. The agent, schema validation, retry logic, and orchestration can be
     tested with zero API cost and zero network dependency.
  2. This project's success criteria (different scenarios -> different,
     evidence-based conclusions, not one templated answer) can be
     demonstrated end-to-end without requiring an ANTHROPIC_API_KEY.

Swap in AnthropicLLMClient for real reasoning; the Agent class doesn't
change either way.
"""

from __future__ import annotations

import json
from typing import Any


class MockLLMClient:
    """Implements the LLMClient protocol (see base.py) without inheriting
    it, to keep this file importable with zero dependency on the rest of
    the package — useful for isolated unit tests of just this engine."""

    def complete(self, system_prompt: str, user_message: str) -> str:
        incident: dict[str, Any] = json.loads(user_message)
        result = self._analyze(incident)
        return json.dumps(result)

    # ------------------------------------------------------------------
    # The rule engine
    # ------------------------------------------------------------------

    def _analyze(self, inc: dict[str, Any]) -> dict[str, Any]:
        facts: list[str] = []
        factors: list[dict[str, Any]] = []
        limitations: list[str] = []

        event_type = inc.get("event_type") or "unknown"
        speed = inc.get("vehicle_speed")
        unit = inc.get("speed_unit") or "mph"
        following_distance = inc.get("following_distance")
        weather = inc.get("weather")
        road_condition = inc.get("road_condition")
        location_type = inc.get("location_type")
        description = (inc.get("description") or "").lower()
        video_available = bool(inc.get("video_available"))
        visual_obs = (inc.get("visual_observations") or "").lower()
        driver_cam = bool(inc.get("driver_facing_camera_available"))

        # --- STEP 1/2: observed facts from whatever fields exist ---
        if speed is not None:
            facts.append(f"Vehicle speed was {speed} {unit}.")
        if following_distance is not None:
            facts.append(f"Following distance was approximately {following_distance} seconds.")
        if road_condition:
            facts.append(f"Road condition was reported as {road_condition}.")
        if weather:
            facts.append(f"Weather was reported as {weather}.")
        if location_type:
            facts.append(f"Incident occurred at a {location_type}.")
        if inc.get("description"):
            facts.append(inc["description"].strip().rstrip(".") + ".")
        if visual_obs:
            facts.append(f"Visual review noted: {inc['visual_observations'].strip()}.")
        if not video_available:
            limitations.append("No dashcam video was provided, so visual confirmation of the surrounding scene was not possible.")
        if not driver_cam:
            limitations.append("No driver-facing camera evidence was available, so driver attention/distraction could not be directly observed.")
        if following_distance is None:
            limitations.append("Following distance telemetry was not provided.")
        if not weather and not road_condition:
            limitations.append("Environmental conditions (weather, road surface) were not reported.")

        # --- external-hazard language in the description ---
        sudden_external = any(
            kw in description
            for kw in ["pedestrian", "cyclist", "animal", "child", "cut in", "cut him off", "cut me off", "ran a red", "ran the light"]
        ) or any(
            kw in description for kw in ["suddenly stopped", "stopped suddenly", "swerved"]
        )
        vehicle_ahead_stopped = "stopped suddenly" in description or "suddenly stopped" in description

        # --- STEP 4: contributing factors ---
        short_following = following_distance is not None and following_distance < 2.0
        if short_following:
            impact = "HIGH" if following_distance < 1.5 else "MEDIUM"
            factors.append({
                "factor": "Short following distance",
                "impact": impact,
                "evidence": f"Reported following distance was approximately {following_distance} seconds.",
            })

        if road_condition and road_condition.lower() in ("wet", "icy", "snow", "snowy"):
            factors.append({
                "factor": f"{road_condition.capitalize()} road conditions",
                "impact": "MEDIUM",
                "evidence": f"Road condition was reported as {road_condition}.",
            })

        if sudden_external:
            factors.append({
                "factor": "Sudden external hazard (pedestrian/vehicle/other road user)",
                "impact": "HIGH",
                "evidence": inc.get("description", "").strip(),
            })

        if "phone" in description or "distract" in description:
            if driver_cam or "confirmed" in description:
                factors.append({
                    "factor": "Driver distraction",
                    "impact": "HIGH",
                    "evidence": inc.get("description", "").strip(),
                })
            else:
                # mentioned but not visually confirmed -> flag as
                # unconfirmed, do NOT treat as an established factor
                limitations.append(
                    "Description suggests possible distraction, but this is not confirmed by driver-facing camera evidence."
                )

        if not factors:
            factors.append({
                "factor": "UNKNOWN",
                "impact": "UNKNOWN",
                "evidence": "Insufficient information was provided to identify specific contributing factors.",
            })

        # --- STEP 6: driver contribution ---
        if sudden_external and not short_following:
            driver_level, driver_expl = "MINOR", (
                "The available evidence points primarily to an external hazard; the driver's response "
                "appears to be a defensive reaction rather than a behaviour that created the risk."
            )
        elif sudden_external and short_following:
            driver_level, driver_expl = "MODERATE", (
                "An external hazard appears to be the immediate trigger, but the short following distance "
                "would have reduced the driver's available reaction time."
            )
        elif short_following and following_distance < 1.5:
            driver_level, driver_expl = "SIGNIFICANT", (
                "The following distance was well below a safe margin, which is a direct driver-controlled factor."
            )
        elif short_following:
            driver_level, driver_expl = "MODERATE", (
                "The following distance was shorter than recommended, which may have reduced available reaction time."
            )
        elif not facts or (speed is None and following_distance is None and not description):
            driver_level, driver_expl = "UNKNOWN", (
                "Insufficient information was provided to assess driver behaviour."
            )
        else:
            driver_level, driver_expl = "UNKNOWN", (
                "Available evidence does not clearly indicate whether driver behaviour contributed."
            )

        # --- STEP 7: severity ---
        is_collision = "collision" in event_type.lower() or "collision" in description or "crash" in description
        if is_collision:
            severity = "CRITICAL"
        elif driver_level == "SIGNIFICANT":
            severity = "HIGH"
        elif driver_level in ("MODERATE",):
            severity = "MEDIUM"
        elif driver_level == "MINOR":
            severity = "LOW"
        else:
            severity = "MEDIUM" if factors and factors[0]["factor"] != "UNKNOWN" else "LOW"

        # --- STEP 8: confidence ---
        confidence = 40
        if speed is not None:
            confidence += 10
        if following_distance is not None:
            confidence += 15
        if weather or road_condition:
            confidence += 10
        if inc.get("description"):
            confidence += 10
        if video_available:
            confidence += 15
        if driver_cam:
            confidence += 5
        confidence = min(confidence, 95)

        conflicting = inc.get("_conflict_flag") is True
        if visual_obs and sudden_external and any(
            kw in visual_obs for kw in ["no vehicle ahead", "empty road", "contradicts", "does not show", "clear road"]
        ):
            conflicting = True
        requires_review = False
        if conflicting:
            confidence = max(confidence - 30, 10)
            requires_review = True
            limitations.append("Video and telemetry evidence conflict; this assessment has reduced confidence pending review.")

        # --- STEP 9: root cause ---
        if factors and factors[0]["factor"] != "UNKNOWN":
            primary = max(
                factors,
                key=lambda f: {"HIGH": 2, "MEDIUM": 1, "LOW": 0, "UNKNOWN": -1}[f["impact"]],
            )
            root_cause_text = primary["factor"]
            root_cause_conf = confidence
            root_cause_expl = primary["evidence"]
        else:
            root_cause_text = "Unknown"
            root_cause_conf = min(confidence, 40)
            root_cause_expl = "Insufficient evidence to determine root cause."

        # --- STEP 11: recommended action ---
        if severity == "CRITICAL":
            action, reason = "SAFETY_INVESTIGATION", "Collision or serious safety-critical event requires a full investigation."
        elif severity == "HIGH":
            action, reason = "MANAGER_REVIEW", "High-severity event with meaningful driver contribution warrants manager review."
        elif severity == "MEDIUM" and driver_level in ("MODERATE", "SIGNIFICANT"):
            action, reason = "DRIVER_COACHING", "Moderate-severity event with an identifiable, coachable driver behaviour."
        elif root_cause_text == "Unknown":
            action, reason = "MONITOR", "Insufficient evidence to justify a targeted action; continue monitoring for a recurring pattern."
        elif driver_level in ("NONE", "MINOR", "UNKNOWN"):
            action, reason = "NO_ACTION", "Available evidence does not indicate driver behaviour requiring intervention."
        else:
            action, reason = "MONITOR", "Low-to-moderate risk signal; monitor for recurrence."

        # --- STEP 12: human review ---
        if severity in ("HIGH", "CRITICAL"):
            requires_review = True
        if confidence < 55:
            requires_review = True
        if driver_level == "UNKNOWN" and severity != "LOW":
            requires_review = True

        summary = self._summarize(inc, event_type, sudden_external, short_following, road_condition)

        return {
            "incident_id": inc.get("incident_id", ""),
            "incident_summary": summary,
            "event_type": event_type,
            "severity": severity,
            "confidence": confidence,
            "observed_facts": facts,
            "contributing_factors": factors,
            "driver_contribution": {"level": driver_level, "explanation": driver_expl},
            "root_cause": {
                "cause": root_cause_text,
                "confidence": root_cause_conf,
                "explanation": root_cause_expl,
            },
            "evidence": facts[:5] if facts else ["No structured evidence was available."],
            "recommended_action": {"action": action, "reason": reason},
            "requires_human_review": requires_review,
            "limitations": limitations or ["None identified."],
        }

    @staticmethod
    def _summarize(inc, event_type, sudden_external, short_following, road_condition) -> str:
        subject = f"The vehicle recorded a {event_type.replace('_', ' ')} event" if event_type != "unknown" else "The vehicle recorded a safety event"
        clause = []
        if sudden_external and inc.get("description"):
            clause.append(f"after {inc['description'][0].lower() + inc['description'][1:].rstrip('.')}")
        elif inc.get("description"):
            clause.append(f"({inc['description'].rstrip('.')})")
        if road_condition:
            clause.append(f"on a {road_condition} road")
        if inc.get("location_type"):
            loc = inc["location_type"]
            article = "an" if loc[:1].lower() in "aeiou" else "a"
            clause.append(f"at {article} {loc}")
        tail = " ".join(clause)
        summary = f"{subject} {tail}.".replace("  ", " ").strip()
        if not tail:
            summary = f"{subject}. Limited supporting detail was provided."
        return summary
