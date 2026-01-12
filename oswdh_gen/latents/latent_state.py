from __future__ import annotations
import numpy as np
import pandas as pd

def smooth_random_walk(rng: np.random.Generator, n: int, start: float, step_sd: float, low=0.0, high=1.0) -> np.ndarray:
    x = np.empty(n, dtype="float32")
    x[0] = start
    for i in range(1, n):
        x[i] = np.clip(x[i-1] + rng.normal(0, step_sd), low, high)
    return x

def generate_latents_minute(rng: np.random.Generator, time_index: pd.DataFrame, participant_row: dict) -> pd.DataFrame:
    n = len(time_index)
    # baseline anchors per participant
    base_stress = float(np.clip(0.25 + 0.6*participant_row["comorbidity_index"] + rng.normal(0,0.05), 0, 1))
    base_arr = float(np.clip(0.15 + 0.7*(participant_row["age_years"]/90.0) + rng.normal(0,0.05), 0, 1))
    base_fit = float(np.clip(participant_row["baseline_fitness_index"] + rng.normal(0,0.05), 0, 1))

    circ_phase = ((time_index["minute_index"].to_numpy() / (24*60)) % 1.0).astype("float32")
    circ_amp = smooth_random_walk(rng, n, start=float(np.clip(0.6 + rng.normal(0,0.05),0,1)), step_sd=0.0015)
    stress = smooth_random_walk(rng, n, start=base_stress, step_sd=0.003)
    inflammation = smooth_random_walk(rng, n, start=float(np.clip(0.2+participant_row["comorbidity_index"]*0.3,0,1)), step_sd=0.0018)
    arr_prop = smooth_random_walk(rng, n, start=base_arr, step_sd=0.002)
    fit = smooth_random_walk(rng, n, start=base_fit, step_sd=0.0005)
    infection = np.zeros(n, dtype="float32")
    sleep_debt = smooth_random_walk(rng, n, start=float(np.clip(0.3+rng.normal(0,0.07),0,1)), step_sd=0.0012)
    auton = smooth_random_walk(rng, n, start=float(np.clip(0.5+rng.normal(0,0.07),0,1)), step_sd=0.002)

    df = pd.DataFrame({
        "circadian_phase_0_1": circ_phase,
        "circadian_amplitude_0_1": circ_amp,
        "sleep_debt_0_1": sleep_debt,
        "autonomic_balance_0_1": auton,
        "stress_load_0_1": stress,
        "inflammatory_state_0_1": inflammation,
        "arrhythmia_propensity_0_1": arr_prop,
        "fitness_index_0_1": fit,
        "infection_burden_0_1": infection,
        "latent_confidence": np.full(n, 1.0, dtype="float32")
    })
    return df
