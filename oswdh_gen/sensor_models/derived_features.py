from __future__ import annotations
import numpy as np
import pandas as pd

def minute_features_from_latents(rng: np.random.Generator, time_index: pd.DataFrame, latents: pd.DataFrame,
                                base_noise_0_1: float) -> pd.DataFrame:
    n = len(time_index)
    # activity intensity: higher during daytime; random bursts
    circ = latents["circadian_phase_0_1"].to_numpy()
    day_mask = (circ > 0.25) & (circ < 0.85)
    acc = rng.beta(1.2, 6.0, size=n).astype("float32")
    acc[day_mask] += rng.beta(2.0, 4.0, size=day_mask.sum()).astype("float32")*0.35
    acc = np.clip(acc, 0, 1)

    # heart rate influenced by stress, sleep debt, activity
    stress = latents["stress_load_0_1"].to_numpy()
    sleep_debt = latents["sleep_debt_0_1"].to_numpy()
    hr = 55 + 45*acc + 18*stress + 10*sleep_debt + rng.normal(0, 3 + 8*base_noise_0_1, size=n)
    hr = np.clip(hr, 25, 240).astype("float32")

    # HRV proxy decreases with stress and sleep debt
    hrv = 65 - 40*stress - 25*sleep_debt + rng.normal(0, 5 + 10*base_noise_0_1, size=n)
    hrv = np.clip(hrv, 0, 200).astype("float32")

    # respiration proxy correlates with activity and stress
    rr = 12 + 10*acc + 6*stress + rng.normal(0, 1 + 3*base_noise_0_1, size=n)
    rr = np.clip(rr, 5, 40).astype("float32")

    # temperature influenced by infection burden (currently zero in v0 generator)
    temp = 33.0 + 0.8*stress + rng.normal(0, 0.1 + 0.3*base_noise_0_1, size=n)
    temp = np.clip(temp, 25, 40).astype("float32")

    # quality/confidence
    ppg_quality = np.clip(1.0 - (0.5*acc + base_noise_0_1*rng.random(n)), 0, 1).astype("float32")
    hr_conf = np.clip(ppg_quality - 0.1*rng.random(n), 0, 1).astype("float32")

    # activity/sleep state predictions (simple)
    activity_state = np.full(n, "sedentary", dtype=object)
    activity_state[acc > 0.6] = "vigorous"
    activity_state[(acc > 0.35) & (acc <= 0.6)] = "moderate"
    activity_state[(acc > 0.15) & (acc <= 0.35)] = "light"
    # synthetic sleep: night window
    sleep_state = np.full(n, "unknown", dtype=object)
    night = (circ < 0.20) | (circ > 0.90)
    sleep_state[night] = "N2"
    activity_state[night] = "sleep"

    df = pd.DataFrame({
        "acc_intensity": acc.astype("float32"),
        "hr_bpm": hr,
        "hr_confidence": hr_conf,
        "ppg_quality": ppg_quality,
        "skin_temp_c": temp,
        "spo2_pct": np.clip(97 + rng.normal(0, 0.7, size=n), 70, 100).astype("float32"),
        "activity_state_pred": activity_state,
        "sleep_state_pred": sleep_state,
    })
    return df

def daily_aggregate(minute_df: pd.DataFrame) -> dict:
    # expects a single day slice
    steps = int((minute_df["acc_intensity"].to_numpy() * 120).sum())  # rough
    wear_minutes = int((~minute_df["missing_flag"]).sum())
    nonwear_minutes = int((minute_df["missing_flag"]).sum())
    artifact_score = float(1.0 - minute_df["ppg_quality"].mean())
    conf = float(minute_df["hr_confidence"].mean())

    # sleep minutes: activity_state == sleep
    sleep_mask = minute_df["activity_state_pred"].astype(str) == "sleep"
    sleep_minutes = int(sleep_mask.sum())

    # sleep staging approx
    rem = int((minute_df["sleep_state_pred"].astype(str) == "REM").sum())
    deep = int((minute_df["sleep_state_pred"].astype(str) == "N3").sum())

    resting_hr = float(minute_df.loc[~sleep_mask, "hr_bpm"].quantile(0.1)) if (~sleep_mask).any() else float(minute_df["hr_bpm"].median())
    avg_hr = float(minute_df["hr_bpm"].mean())
    hrv_proxy = float(np.clip(65 - 30*(1.0-conf) - 20*artifact_score, 0, 200))
    rr_proxy = float(minute_df["hr_bpm"].mean()/6.0)

    sleep_eff = float(np.clip(sleep_minutes / 480.0, 0, 1))  # proxy
    stress_index = float(np.clip(1.0 - minute_df["hrv_rmssd_ms_proxy"].mean()/120.0 if "hrv_rmssd_ms_proxy" in minute_df else 0.5, 0, 1))

    return {
        "wear_minutes": wear_minutes,
        "nonwear_minutes": nonwear_minutes,
        "artifact_score": float(np.clip(artifact_score,0,1)),
        "feature_confidence": float(np.clip(conf,0,1)),
        "steps": steps,
        "active_minutes": int((minute_df["acc_intensity"]>0.35).sum()),
        "sedentary_minutes": int((minute_df["acc_intensity"]<=0.15).sum()),
        "resting_hr_bpm": float(resting_hr),
        "avg_hr_bpm": float(avg_hr),
        "hrv_rmssd_ms_proxy": float(hrv_proxy),
        "resp_rate_rpm_proxy": float(np.clip(rr_proxy,5,40)),
        "skin_temp_c": float(minute_df["skin_temp_c"].mean()),
        "sleep_duration_min": sleep_minutes,
        "sleep_efficiency": sleep_eff,
        "rem_min": rem,
        "deep_min": deep,
        "awakenings_count": int(np.clip((1.0-sleep_eff)*10,0,50)),
        "strain_index": float(np.clip(minute_df["acc_intensity"].mean(),0,1)),
        "recovery_index": float(np.clip(conf*(1.0-artifact_score),0,1)),
        "stress_index": float(np.clip(stress_index,0,1)),
    }
