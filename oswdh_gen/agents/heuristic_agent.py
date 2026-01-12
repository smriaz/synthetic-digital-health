from __future__ import annotations
import numpy as np

def heuristic_decision(daily_row: dict) -> tuple[str, dict, dict]:
    # Simple, transparent policy
    stress = float(daily_row.get("stress_index", 0.5))
    sleep = float(daily_row.get("sleep_duration_min", 0)) / 480.0
    conf = float(daily_row.get("feature_confidence", 0.5))

    tools_used = ["trend_analyzer_v1","risk_model_v1"]

    if conf < 0.3:
        return ("do_nothing", {}, {"plan":"Low confidence day; avoid acting."})

    if stress > 0.7 and sleep < 0.7:
        return ("send_survey", {"survey_type":"ema","intensity":"low"}, {"plan":"High stress + low sleep -> brief EMA and wind-down recommendation."})
    if sleep < 0.5:
        return ("recommend", {"intervention_id":"sleep_wind_down"}, {"plan":"Low sleep -> recommend wind-down routine."})
    if stress > 0.8:
        return ("recommend", {"intervention_id":"stress_breathing"}, {"plan":"High stress -> recommend breathing routine."})
    return ("do_nothing", {}, {"plan":"No action needed."})

def reward_from_outcomes(outcome_count: int, alerts_sent: int) -> dict:
    # Simple reward proxy: fewer outcomes is better; fewer alerts is better
    health = 0.05 if outcome_count == 0 else -0.10 * outcome_count
    fatigue_penalty = -0.02 * alerts_sent
    cost = -0.01 * alerts_sent
    total = health + fatigue_penalty + cost
    return {"total": float(total), "health": float(health), "fatigue_penalty": float(fatigue_penalty), "cost": float(cost)}
