"""
Deterministic risk-scoring engine for Agent 2 (Driver Risk Analyst).

DESIGN DECISION, stated explicitly because it's the one choice in this
project most worth defending in an interview: the NUMBERS in a driver risk
score (frequency counts, the weighted score, whether the trend is up or
down) are computed here, in plain Python — not asked of an LLM.

Why: an LLM is good at reading messy, unstructured evidence and producing a
qualitative judgment (that's what Agent 1 does). It is not the right tool
for "count how many of these 8 incidents were HIGH severity" or "is the
second half of this window worse than the first half" — those are exact
arithmetic operations with one correct answer, and asking a language model
to eyeball them risks an occasionally wrong number that looks exactly as
confident as a right one. A fleet safety risk score that ranks drivers for
a manager's attention has to be reproducible and auditable: same input,
same score, every time, with a formula someone can check by hand.

So Agent 2's LLM (see prompts/driver_risk_analyst.py) is only ever handed
the numbers *already computed here* and asked to write the human-readable
explanation grounded in them — never to produce the numbers itself.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from ..schemas import DriverContributionLevel, IncidentOutput, RiskLevel, Severity, TrendDirection

# Points contributed per incident by severity. Roughly geometric: each step
# up is meaningfully worse, CRITICAL dominates a risk score the way an
# actual collision should.
SEVERITY_WEIGHT: dict[Severity, float] = {
    Severity.LOW: 5,
    Severity.MEDIUM: 15,
    Severity.HIGH: 35,
    Severity.CRITICAL: 70,
}

# Scales the severity weight by how much the driver actually contributed.
# This deliberately mirrors Agent 1's Rule 2 ("never automatically blame the
# driver") into the scoring layer: a defensive-braking incident the driver
# didn't cause should barely move their risk score, even if it was a HIGH-
# severity event to investigate.
CONTRIBUTION_MULTIPLIER: dict[DriverContributionLevel, float] = {
    DriverContributionLevel.NONE: 0.2,
    DriverContributionLevel.MINOR: 0.5,
    DriverContributionLevel.UNKNOWN: 0.7,
    DriverContributionLevel.MODERATE: 1.0,
    DriverContributionLevel.SIGNIFICANT: 1.4,
}

NORMALIZATION_WINDOW_DAYS = 30  # the score is expressed as "per 30 days",
# so a driver evaluated over a 7-day window and one evaluated over 90 days
# are still comparable.

RISK_LEVEL_THRESHOLDS = [
    (75, RiskLevel.CRITICAL),
    (50, RiskLevel.HIGH),
    (25, RiskLevel.MEDIUM),
    (0, RiskLevel.LOW),
]

MIN_INCIDENTS_FOR_TREND = 4  # below this, "increasing vs decreasing" isn't
# a meaningfully different claim from noise.


@dataclass
class DriverRiskStats:
    total_incidents: int
    breakdown: list[tuple[str, int]]  # (event_type, count), sorted desc
    risk_score: int
    risk_level: RiskLevel
    trend: TrendDirection
    primary_concern: str
    primary_concern_weight: float
    recurring: list["RecurringGroup"]
    evidence: list[str]
    confidence: int
    undated_count: int


@dataclass
class RecurringGroup:
    pattern: str
    occurrences: int
    trend: TrendDirection
    incident_ids: list[str] = field(default_factory=list)


def _weighted_score(incident: IncidentOutput) -> float:
    return SEVERITY_WEIGHT[incident.severity] * CONTRIBUTION_MULTIPLIER[incident.driver_contribution.level]


def _parse_timestamp(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def compute_stats(incidents: list[IncidentOutput], time_window_days: int) -> DriverRiskStats:
    total = len(incidents)

    if total == 0:
        return DriverRiskStats(
            total_incidents=0,
            breakdown=[],
            risk_score=0,
            risk_level=RiskLevel.LOW,
            trend=TrendDirection.INSUFFICIENT_DATA,
            primary_concern="None",
            primary_concern_weight=0.0,
            recurring=[],
            evidence=["No incidents were provided for this driver in the given window."],
            confidence=20,
            undated_count=0,
        )

    # --- frequency breakdown by event_type ---
    type_counts = Counter(i.event_type for i in incidents)
    breakdown = sorted(type_counts.items(), key=lambda kv: kv[1], reverse=True)

    # --- weighted risk score, normalized to a NORMALIZATION_WINDOW_DAYS basis ---
    raw = sum(_weighted_score(i) for i in incidents)
    scale = NORMALIZATION_WINDOW_DAYS / max(time_window_days, 1)
    risk_score = min(100, round(raw * scale))

    risk_level = next(level for threshold, level in RISK_LEVEL_THRESHOLDS if risk_score >= threshold)

    # --- primary concern: category with the highest total weighted contribution ---
    weight_by_type: dict[str, float] = defaultdict(float)
    for i in incidents:
        weight_by_type[i.event_type] += _weighted_score(i)
    primary_concern, primary_weight = max(weight_by_type.items(), key=lambda kv: kv[1])

    # --- trend: compare weighted score in the first half of the window vs the second half ---
    dated = [(i, _parse_timestamp(i.timestamp)) for i in incidents]
    undated_count = sum(1 for _, ts in dated if ts is None)
    dated_valid = [(i, ts) for i, ts in dated if ts is not None]

    if total < MIN_INCIDENTS_FOR_TREND or len(dated_valid) < MIN_INCIDENTS_FOR_TREND:
        trend = TrendDirection.INSUFFICIENT_DATA
    else:
        dated_valid.sort(key=lambda pair: pair[1])
        midpoint = dated_valid[0][1] + (dated_valid[-1][1] - dated_valid[0][1]) / 2
        first_half = [i for i, ts in dated_valid if ts <= midpoint]
        second_half = [i for i, ts in dated_valid if ts > midpoint]
        if not first_half or not second_half:
            trend = TrendDirection.INSUFFICIENT_DATA
        else:
            first_score = sum(_weighted_score(i) for i in first_half) / len(first_half)
            second_score = sum(_weighted_score(i) for i in second_half) / len(second_half)
            if first_score == 0:
                trend = TrendDirection.INCREASING if second_score > 0 else TrendDirection.STABLE
            elif second_score >= first_score * 1.2:
                trend = TrendDirection.INCREASING
            elif second_score <= first_score * 0.8:
                trend = TrendDirection.DECREASING
            else:
                trend = TrendDirection.STABLE

    # --- recurring patterns: group by root_cause.cause, keep those seen 2+ times ---
    by_cause: dict[str, list[IncidentOutput]] = defaultdict(list)
    for i in incidents:
        cause = i.root_cause.cause
        if cause and cause.strip().lower() not in ("unknown", ""):
            by_cause[cause].append(i)

    recurring: list[RecurringGroup] = []
    for cause, group in sorted(by_cause.items(), key=lambda kv: len(kv[1]), reverse=True):
        if len(group) < 2:
            continue
        group_dated = [(i, _parse_timestamp(i.timestamp)) for i in group]
        group_dated_valid = [(i, ts) for i, ts in group_dated if ts is not None]
        if len(group_dated_valid) >= MIN_INCIDENTS_FOR_TREND:
            # Split by the TIME midpoint of this group's own date range, not
            # by index — an index split (sorted list cut in half by count)
            # trivially always gives ~equal halves and can never actually
            # detect clustering later in the window.
            group_dated_valid.sort(key=lambda p: p[1])
            g_start, g_end = group_dated_valid[0][1], group_dated_valid[-1][1]
            g_mid = g_start + (g_end - g_start) / 2
            first_n = sum(1 for _, ts in group_dated_valid if ts <= g_mid)
            second_n = sum(1 for _, ts in group_dated_valid if ts > g_mid)
            if first_n == 0:
                group_trend = TrendDirection.INCREASING if second_n > 0 else TrendDirection.STABLE
            elif second_n >= first_n * 1.2:
                group_trend = TrendDirection.INCREASING
            elif second_n <= first_n * 0.8:
                group_trend = TrendDirection.DECREASING
            else:
                group_trend = TrendDirection.STABLE
        else:
            group_trend = TrendDirection.INSUFFICIENT_DATA
        recurring.append(RecurringGroup(
            pattern=cause,
            occurrences=len(group),
            trend=group_trend,
            incident_ids=[i.incident_id for i in group],
        ))

    # --- confidence: more incidents and full time coverage = more confidence ---
    confidence = min(95, 30 + total * 7)
    if undated_count > 0:
        confidence = max(20, confidence - undated_count * 10)

    # --- evidence: plain factual statements a manager can verify against the source incidents ---
    evidence = [f"{total} incident(s) recorded for this driver in the last {time_window_days} days."]
    for event_type, count in breakdown[:4]:
        evidence.append(f"{event_type.replace('_', ' ')}: {count} occurrence(s).")
    high_or_critical = sum(1 for i in incidents if i.severity in (Severity.HIGH, Severity.CRITICAL))
    if high_or_critical:
        evidence.append(f"{high_or_critical} of {total} incidents were rated HIGH or CRITICAL severity.")
    significant_contribution = sum(
        1 for i in incidents
        if i.driver_contribution.level in (DriverContributionLevel.SIGNIFICANT, DriverContributionLevel.MODERATE)
    )
    if significant_contribution:
        evidence.append(
            f"Driver behaviour was assessed as a moderate or significant contributor in "
            f"{significant_contribution} of {total} incidents."
        )
    if undated_count:
        evidence.append(f"{undated_count} incident(s) had no timestamp and were excluded from trend analysis.")

    return DriverRiskStats(
        total_incidents=total,
        breakdown=breakdown,
        risk_score=risk_score,
        risk_level=risk_level,
        trend=trend,
        primary_concern=primary_concern,
        primary_concern_weight=primary_weight,
        recurring=recurring,
        evidence=evidence,
        confidence=confidence,
        undated_count=undated_count,
    )
