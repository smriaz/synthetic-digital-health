from __future__ import annotations
import numpy as np
import pandas as pd

def generate_outcomes(rng: np.random.Generator, time_index: pd.DataFrame, latents: pd.DataFrame, participant_profile: dict,
                      tracks: dict) -> pd.DataFrame:
    # Features-only v1 minimal: generate sparse events from hazards driven by latents.
    events = []
    ts = time_index["timestamp_utc_ms"].to_numpy()
    day_index = time_index["day_index"].to_numpy()

    # Cardio rhythm: AF-like hazard from arrhythmia_propensity and sleep_debt and stress
    if tracks.get("cardio_rhythm", {}).get("enabled", True):
        base_prev = float(tracks["cardio_rhythm"].get("prevalence_0_1", 0.06))
        has_risk = rng.random() < base_prev + 0.15*(participant_profile["age_years"]/90.0)
        if has_risk:
            arr = latents["arrhythmia_propensity_0_1"].to_numpy()
            stress = latents["stress_load_0_1"].to_numpy()
            debt = latents["sleep_debt_0_1"].to_numpy()
            hazard = np.clip(0.000005 + 0.00004*arr + 0.00002*stress + 0.00002*debt, 0, 0.001)
            # sample events
            for i in range(len(ts)):
                if rng.random() < hazard[i]:
                    events.append({
                        "timestamp_utc_ms": int(ts[i]),
                        "day_index": int(day_index[i]),
                        "outcome_type": "af_like",
                        "severity_0_1": float(np.clip(0.3 + 0.7*arr[i],0,1)),
                        "outcome_window_start_ts_utc_ms": int(ts[i]),
                        "outcome_window_end_ts_utc_ms": int(ts[min(i+60, len(ts)-1)]),
                        "label_source": "synthetic_truth",
                        "label_noise_flag": False,
                        "label_delay_days": 0,
                        "notes": ""
                    })
                    # limit events per participant for small demo
                    if len(events) > 10:
                        break

    # Infection-like onset: rare spikes
    if tracks.get("infection_like", {}).get("enabled", True):
        yearly = float(tracks["infection_like"].get("episodes_per_year_mean", 1.0))
        # scale by duration
        expected = yearly * (len(time_index) / (365*24*60))
        n_epi = int(rng.poisson(expected))
        for _ in range(n_epi):
            i = int(rng.integers(0, len(ts)))
            events.append({
                "timestamp_utc_ms": int(ts[i]),
                "day_index": int(day_index[i]),
                "outcome_type": "ili_like_onset",
                "severity_0_1": float(np.clip(0.4 + rng.random()*0.6,0,1)),
                "outcome_window_start_ts_utc_ms": int(ts[i]),
                "outcome_window_end_ts_utc_ms": int(ts[min(i+24*60, len(ts)-1)]),
                "label_source": "synthetic_truth",
                "label_noise_flag": False,
                "label_delay_days": 0,
                "notes": ""
            })

    return pd.DataFrame(events)
