"""
Realistic driver-history scenarios for Agent 2, expressed directly as
Agent-1-shaped incident dicts (as if Agent 1 already ran on each one) so
these tests exercise Agent 2's own logic in isolation. The end-to-end
Agent 1 -> Agent 2 pipeline is covered separately, in
test_pipeline_integration.py.
"""


def _incident(
    incident_id, event_type, severity, driver_level, root_cause, timestamp,
    confidence=80,
):
    return {
        "incident_id": incident_id,
        "incident_summary": f"{event_type} event.",
        "event_type": event_type,
        "severity": severity,
        "confidence": confidence,
        "observed_facts": [],
        "contributing_factors": [],
        "driver_contribution": {"level": driver_level, "explanation": "test fixture"},
        "root_cause": {"cause": root_cause, "confidence": confidence, "explanation": "test fixture"},
        "evidence": [],
        "recommended_action": {"action": "MONITOR", "reason": "test fixture"},
        "requires_human_review": severity in ("HIGH", "CRITICAL"),
        "limitations": [],
        "timestamp": timestamp,
    }


DRIVER_SCENARIOS = [
    {
        "name": "stable_low_risk_driver",
        "description": "A handful of low-severity incidents spread evenly across the window — should stay LOW/STABLE, not flagged for attention.",
        "input": {
            "driver_id": "D-100",
            "time_window_days": 30,
            "incidents": [
                _incident("I-1", "harsh_braking", "LOW", "MINOR", "Sudden external hazard", "2026-08-02T09:00:00"),
                _incident("I-2", "harsh_braking", "LOW", "NONE", "Sudden external hazard", "2026-08-09T09:00:00"),
                _incident("I-3", "harsh_braking", "LOW", "MINOR", "Sudden external hazard", "2026-08-16T09:00:00"),
                _incident("I-4", "harsh_braking", "LOW", "NONE", "Sudden external hazard", "2026-08-23T09:00:00"),
            ],
        },
        "expect": {
            "risk_level_in": ["LOW", "MEDIUM"],
            "trend_in": ["STABLE"],
            "requires_immediate_attention": False,
        },
    },
    {
        "name": "escalating_high_risk_driver",
        "description": "Severity and driver contribution both worsen across the window — should be HIGH/CRITICAL, INCREASING, flagged for attention.",
        "input": {
            "driver_id": "D-102",
            "time_window_days": 30,
            "incidents": [
                _incident("I-10", "harsh_braking", "LOW", "MINOR", "Short following distance", "2026-08-01T09:00:00"),
                _incident("I-11", "tailgating", "MEDIUM", "MODERATE", "Short following distance", "2026-08-03T09:00:00"),
                _incident("I-12", "harsh_braking", "MEDIUM", "MODERATE", "Short following distance", "2026-08-15T09:00:00"),
                _incident("I-13", "tailgating", "HIGH", "SIGNIFICANT", "Short following distance", "2026-08-24T09:00:00"),
                _incident("I-14", "harsh_braking", "HIGH", "SIGNIFICANT", "Short following distance", "2026-08-27T09:00:00"),
                _incident("I-15", "tailgating", "HIGH", "SIGNIFICANT", "Short following distance", "2026-08-29T09:00:00"),
            ],
        },
        "expect": {
            "risk_level_in": ["HIGH", "CRITICAL"],
            "trend_in": ["INCREASING"],
            "requires_immediate_attention": True,
            "primary_concern_in": ["harsh_braking", "tailgating"],
            "has_recurring_pattern": "Short following distance",
        },
    },
    {
        "name": "defensive_driver_not_at_fault",
        "description": "Several HIGH-severity events, but driver_contribution is NONE/MINOR throughout (external hazards) — risk score must stay low despite the raw severity, mirroring Agent 1's 'don't auto-blame the driver' rule.",
        "input": {
            "driver_id": "D-103",
            "time_window_days": 30,
            "incidents": [
                _incident("I-20", "harsh_braking", "HIGH", "NONE", "Sudden external hazard", "2026-08-02T09:00:00"),
                _incident("I-21", "harsh_braking", "HIGH", "MINOR", "Sudden external hazard", "2026-08-10T09:00:00"),
                _incident("I-22", "harsh_braking", "MEDIUM", "NONE", "Sudden external hazard", "2026-08-18T09:00:00"),
                _incident("I-23", "harsh_braking", "HIGH", "MINOR", "Sudden external hazard", "2026-08-26T09:00:00"),
            ],
        },
        "expect": {
            "risk_level_in": ["LOW", "MEDIUM"],
            "requires_immediate_attention": False,
        },
    },
    {
        "name": "insufficient_history",
        "description": "Only two incidents in the window — too few to call a trend.",
        "input": {
            "driver_id": "D-104",
            "time_window_days": 30,
            "incidents": [
                _incident("I-30", "harsh_braking", "MEDIUM", "MODERATE", "Short following distance", "2026-08-05T09:00:00"),
                _incident("I-31", "harsh_braking", "MEDIUM", "MODERATE", "Short following distance", "2026-08-20T09:00:00"),
            ],
        },
        "expect": {
            "trend_in": ["INSUFFICIENT_DATA"],
        },
    },
    {
        "name": "no_incidents",
        "description": "A driver with a completely clean record in the window.",
        "input": {
            "driver_id": "D-105",
            "time_window_days": 30,
            "incidents": [],
        },
        "expect": {
            "risk_level_in": ["LOW"],
            "total_incidents": 0,
            "requires_immediate_attention": False,
        },
    },
]
