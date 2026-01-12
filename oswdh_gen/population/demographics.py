from __future__ import annotations
import numpy as np
import pandas as pd

def sample_sex(rng: np.random.Generator, dist: dict) -> str:
    keys = list(dist.keys())
    probs = np.array([dist[k] for k in keys], dtype=float)
    probs = probs / probs.sum()
    return rng.choice(keys, p=probs).item()

def beta01(rng: np.random.Generator, a: float, b: float) -> float:
    return float(rng.beta(a, b))

def generate_participants(rng: np.random.Generator, *, n: int, age_range: list, sex_dist: dict,
                          socio_params: dict, comorb_params: dict) -> pd.DataFrame:
    ages = rng.uniform(age_range[0], age_range[1], size=n).astype("float32")
    sex = [sample_sex(rng, sex_dist) for _ in range(n)]
    height = rng.normal(170, 10, size=n).clip(120, 220).astype("float32")
    weight = rng.normal(75, 15, size=n).clip(35, 200).astype("float32")
    bmi = (weight / (height/100.0)**2).astype("float32")
    fitness = rng.beta(2.2, 2.2, size=n).astype("float32")

    socio = np.array([beta01(rng, socio_params.get("a",2.0), socio_params.get("b",2.5)) for _ in range(n)], dtype="float32")
    comorb = np.array([beta01(rng, comorb_params.get("a",1.8), comorb_params.get("b",3.0)) for _ in range(n)], dtype="float32")

    alert_tol = rng.beta(2.0, 2.0, size=n).astype("float32")
    survey_tol = rng.beta(2.0, 2.5, size=n).astype("float32")

    smoking = rng.choice(["never","former","current","unknown"], size=n, p=[0.55,0.25,0.15,0.05])

    return pd.DataFrame({
        "sex": sex,
        "age_years": ages,
        "height_cm": height,
        "weight_kg": weight,
        "bmi": bmi,
        "smoking_status": smoking,
        "baseline_fitness_index": fitness,
        "alert_tolerance": alert_tol,
        "survey_burden_tolerance": survey_tol,
        "socioeconomic_index": socio,
        "comorbidity_index": comorb
    })
