"""
Realistic test scenarios for Agent 1, covering the success criteria from
the project brief: the agent must reach different, evidence-based
conclusions for different incidents, not one templated answer for every
"harsh_braking" event. Each scenario states what a correct analysis
should NOT get wrong — used both as documentation and by test_incident_analyst.py.
"""

SCENARIOS = [
    {
        "name": "sudden_stop_wet_junction",
        "description": "Vehicle ahead stopped suddenly at a wet junction; following distance was short.",
        "input": {
            "incident_id": "INC-001",
            "event_type": "harsh_braking",
            "vehicle_speed": 42,
            "speed_unit": "mph",
            "following_distance": 1.4,
            "weather": "rain",
            "road_condition": "wet",
            "location_type": "junction",
            "timestamp": "2026-08-31T10:30:00",
            "description": "Vehicle ahead stopped suddenly.",
            "video_available": False,
            "image_available": False,
        },
        "expect": {
            "severity_in": ["MEDIUM", "HIGH"],
            "driver_contribution_in": ["MODERATE", "SIGNIFICANT"],
            "requires_human_review": None,  # not asserted either way
            "root_cause_not_unknown": True,
        },
    },
    {
        "name": "pedestrian_defensive_braking",
        "description": "Harsh braking triggered by a pedestrian suddenly entering the road — should NOT default to blaming the driver.",
        "input": {
            "incident_id": "INC-002",
            "event_type": "harsh_braking",
            "vehicle_speed": 25,
            "speed_unit": "mph",
            "following_distance": 3.5,
            "weather": "clear",
            "road_condition": "dry",
            "location_type": "residential street",
            "description": "A pedestrian suddenly entered the road from between parked cars.",
            "video_available": True,
            "visual_observations": "Dashcam shows a pedestrian stepping into the lane approximately 2 seconds before braking; no other vehicles nearby.",
        },
        "expect": {
            "severity_in": ["LOW", "MEDIUM"],
            "driver_contribution_in": ["MINOR", "NONE"],
            "recommended_action_not_in": ["MANAGER_REVIEW", "SAFETY_INVESTIGATION", "IMMEDIATE_INTERVENTION"],
        },
    },
    {
        "name": "tailgating_wet_no_external_trigger",
        "description": "Harsh braking on a wet road with an unsafe following distance and no external hazard mentioned — driver behaviour should carry more weight.",
        "input": {
            "incident_id": "INC-003",
            "event_type": "harsh_braking",
            "vehicle_speed": 38,
            "speed_unit": "mph",
            "following_distance": 0.9,
            "weather": "rain",
            "road_condition": "wet",
            "location_type": "highway",
            "description": "Harsh braking event detected during highway driving.",
            "video_available": False,
        },
        "expect": {
            "driver_contribution_in": ["SIGNIFICANT", "MODERATE"],
            "root_cause_not_unknown": True,
        },
    },
    {
        "name": "bare_event_only",
        "description": "Only 'harsh braking detected' — no telemetry, no description, no context. Agent must not invent a cause.",
        "input": {
            "incident_id": "INC-004",
            "event_type": "harsh_braking",
        },
        "expect": {
            "root_cause_is_unknown_or_insufficient": True,
            "driver_contribution_in": ["UNKNOWN"],
            "confidence_below": 60,
        },
    },
    {
        "name": "suspected_distraction_no_visual_evidence",
        "description": "Description hints at distraction but there is no driver-facing camera to confirm it — must not assert distraction as fact.",
        "input": {
            "incident_id": "INC-005",
            "event_type": "lane_departure",
            "vehicle_speed": 55,
            "speed_unit": "mph",
            "road_condition": "dry",
            "location_type": "highway",
            "description": "Vehicle drifted out of lane briefly; driver may have been distracted by phone.",
            "video_available": False,
            "driver_facing_camera_available": False,
        },
        "expect": {
            "driver_contribution_not_in": ["SIGNIFICANT"],
            "limitations_mention_distraction_unconfirmed": True,
        },
    },
    {
        "name": "collision_event",
        "description": "An actual collision — should be CRITICAL severity and routed for safety investigation, always requiring human review.",
        "input": {
            "incident_id": "INC-006",
            "event_type": "collision",
            "vehicle_speed": 30,
            "speed_unit": "mph",
            "following_distance": 0.8,
            "road_condition": "dry",
            "location_type": "intersection",
            "description": "Vehicle collided with another vehicle at an intersection.",
            "video_available": True,
            "visual_observations": "Dashcam confirms impact with another vehicle crossing the intersection.",
        },
        "expect": {
            "severity_in": ["CRITICAL"],
            "requires_human_review_true": True,
            "recommended_action_in": ["SAFETY_INVESTIGATION", "MANAGER_REVIEW", "IMMEDIATE_INTERVENTION"],
        },
    },
    {
        "name": "conflicting_video_and_description",
        "description": "The written description claims a vehicle ahead stopped suddenly, but the video review contradicts that — confidence should drop and human review should trigger.",
        "input": {
            "incident_id": "INC-007",
            "event_type": "harsh_braking",
            "vehicle_speed": 40,
            "speed_unit": "mph",
            "following_distance": 1.2,
            "road_condition": "dry",
            "location_type": "junction",
            "description": "Vehicle ahead stopped suddenly.",
            "video_available": True,
            "visual_observations": "Video review shows an empty road ahead; footage does not show a vehicle ahead of the driver at the time of braking, which contradicts the incident description.",
        },
        "expect": {
            "requires_human_review_true": True,
            "confidence_below": 66,
        },
    },
]
