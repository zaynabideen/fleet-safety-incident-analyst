"""
Pydantic schemas for the Fleet Safety Incident Analyst (Agent 1).

Two schemas matter here:
  - IncidentInput:  what the agent accepts. Every field except incident_id
                     is optional, because real fleet data is incomplete —
                     a text-only event has different fields available than
                     one with video + full telemetry.
  - IncidentOutput: what the agent returns. This is the CONTRACT the rest
                     of the multi-agent system (Driver Risk Analyst, Fleet
                     Risk Analyst, Coaching Agent, Action Agent) will be
                     built against, so it is deliberately strict: enums
                     instead of free strings wherever the value drives
                     downstream logic, and Field(ge=0, le=100) instead of
                     trusting the model to stay in range.

Keep this file the single source of truth for the schema. Both LLM backends
(AnthropicLLMClient, MockLLMClient) and every future agent import from here.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums — every field that downstream agents will branch on is an enum, not
# a free string. An LLM that returns "Medium-High" instead of "HIGH" fails
# validation loudly instead of silently corrupting a driver's risk score
# three agents later.
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ImpactLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


class DriverContributionLevel(str, Enum):
    SIGNIFICANT = "SIGNIFICANT"
    MODERATE = "MODERATE"
    MINOR = "MINOR"
    NONE = "NONE"
    UNKNOWN = "UNKNOWN"


class RecommendedActionType(str, Enum):
    NO_ACTION = "NO_ACTION"
    MONITOR = "MONITOR"
    DRIVER_COACHING = "DRIVER_COACHING"
    TARGETED_TRAINING = "TARGETED_TRAINING"
    MANAGER_REVIEW = "MANAGER_REVIEW"
    SAFETY_INVESTIGATION = "SAFETY_INVESTIGATION"
    IMMEDIATE_INTERVENTION = "IMMEDIATE_INTERVENTION"


class SpeedUnit(str, Enum):
    MPH = "mph"
    KMH = "kmh"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class TrendDirection(str, Enum):
    INCREASING = "INCREASING"
    DECREASING = "DECREASING"
    STABLE = "STABLE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------

class IncidentInput(BaseModel):
    """
    Structured incident data. Only incident_id and event_type are required.
    Everything else may be None — the agent must reason with whatever subset
    is actually present and must never invent a value for a missing field.

    Fields are grouped to match the brief: incident, vehicle/telemetry,
    environment, visual, and forward-looking placeholders for data this
    agent doesn't require yet (driver/vehicle history) but the schema
    should not have to change to accept later.
    """

    model_config = {"extra": "allow"}  # forward-compatible: unknown future
    # fields (e.g. a new telemetry channel) are preserved rather than
    # silently dropped, even though this agent doesn't act on them yet.

    # --- incident ---
    incident_id: str
    event_type: Optional[str] = None
    timestamp: Optional[str] = None
    description: Optional[str] = None
    event_duration_seconds: Optional[float] = Field(default=None, ge=0)
    reported_severity: Optional[str] = None  # upstream system's own
    # classification, if any — the agent may flag this as inconsistent
    # with the evidence, it must not just defer to it.

    # --- vehicle / telemetry ---
    vehicle_id: Optional[str] = None
    vehicle_speed: Optional[float] = Field(default=None, ge=0)
    speed_unit: Optional[SpeedUnit] = SpeedUnit.MPH
    acceleration: Optional[float] = None
    braking_intensity: Optional[float] = Field(default=None, ge=0, le=1)
    following_distance: Optional[float] = Field(default=None, ge=0)  # seconds
    steering_info: Optional[str] = None
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None

    # --- environment ---
    weather: Optional[str] = None
    road_condition: Optional[str] = None
    visibility: Optional[str] = None
    traffic_conditions: Optional[str] = None
    location_type: Optional[str] = None
    time_of_day: Optional[str] = None

    # --- visual ---
    video_available: bool = False
    image_available: bool = False
    # Until real perception (frame extraction / VLM) is wired in, visual
    # evidence arrives as a human- or upstream-model-written description of
    # what's visible. None means no visual evidence at all — NOT "nothing
    # notable was seen."
    visual_observations: Optional[str] = None
    driver_facing_camera_available: bool = False

    # --- forward-looking placeholders (not used by Agent 1) ---
    driver_history: Optional[dict] = None
    vehicle_history: Optional[dict] = None

    @field_validator("incident_id")
    @classmethod
    def incident_id_not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("incident_id must not be blank")
        return v


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class ContributingFactor(BaseModel):
    factor: str
    impact: ImpactLevel
    evidence: str


class DriverContribution(BaseModel):
    level: DriverContributionLevel
    explanation: str


class RootCause(BaseModel):
    cause: str
    confidence: int = Field(ge=0, le=100)
    explanation: str


class RecommendedAction(BaseModel):
    action: RecommendedActionType
    reason: str


class IncidentOutput(BaseModel):
    incident_id: str
    incident_summary: str
    event_type: str
    severity: Severity
    confidence: int = Field(ge=0, le=100)

    observed_facts: list[str] = Field(default_factory=list)
    contributing_factors: list[ContributingFactor] = Field(default_factory=list)
    driver_contribution: DriverContribution
    root_cause: RootCause
    evidence: list[str] = Field(default_factory=list)
    recommended_action: RecommendedAction
    requires_human_review: bool
    limitations: list[str] = Field(default_factory=list)

    # Carried over from IncidentInput.timestamp by the agent (not decided by
    # the LLM) — added for Agent 2 (Driver Risk Analyst), which needs to
    # order a driver's incidents in time to detect a worsening/improving
    # trend. Optional because older incidents, or a caller that never had a
    # timestamp, must still work.
    timestamp: Optional[str] = None


# ---------------------------------------------------------------------------
# Agent 2 — Driver Risk Analyst — schemas
#
# Agent 2 consumes a LIST of Agent 1's IncidentOutput for one driver, not raw
# incidents. This is deliberate: it should never have to re-derive facts,
# severity, or root cause that Agent 1 already established — it only asks
# "across this driver's already-analyzed history, what pattern emerges?"
# ---------------------------------------------------------------------------

class CategoryCount(BaseModel):
    category: str
    count: int = Field(ge=0)


class RecurringPattern(BaseModel):
    pattern: str
    occurrences: int = Field(ge=1)
    trend: TrendDirection
    explanation: str
    example_incident_ids: list[str] = Field(default_factory=list)


class DriverRiskInput(BaseModel):
    model_config = {"extra": "allow"}

    driver_id: str
    time_window_days: int = Field(default=30, ge=1)
    incidents: list[IncidentOutput] = Field(default_factory=list)

    @field_validator("driver_id")
    @classmethod
    def driver_id_not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("driver_id must not be blank")
        return v


class DriverRiskOutput(BaseModel):
    driver_id: str
    time_window_days: int

    total_incidents: int = Field(ge=0)
    incident_breakdown: list[CategoryCount] = Field(default_factory=list)

    risk_score: int = Field(ge=0, le=100)
    risk_level: RiskLevel
    trend: TrendDirection

    primary_concern: str
    recurring_patterns: list[RecurringPattern] = Field(default_factory=list)

    evidence: list[str] = Field(default_factory=list)
    confidence: int = Field(ge=0, le=100)

    recommended_focus_areas: list[str] = Field(default_factory=list)
    requires_immediate_attention: bool
    limitations: list[str] = Field(default_factory=list)
