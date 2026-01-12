from __future__ import annotations
import json
import numpy as np
import pandas as pd

def build_explanations(rng: np.random.Generator, outcomes: pd.DataFrame, latents: pd.DataFrame,
                       time_index: pd.DataFrame, top_k: int, counterfactuals_per_event: int) -> pd.DataFrame:
    if outcomes.empty:
        return pd.DataFrame(columns=["event_id","participant_id","timestamp_utc_ms","outcome_type","true_drivers_json","precursors_json","mechanism_ids_json","counterfactuals_json"])

    # Build a quick lookup from timestamp to latent row
    idx = pd.Series(range(len(time_index)), index=time_index["timestamp_utc_ms"].astype("int64"))
    records = []
    for _, row in outcomes.iterrows():
        t = int(row["timestamp_utc_ms"])
        i = int(idx.get(t, 0))
        # driver contributions (synthetic truth): choose latents depending on outcome type
        if row["outcome_type"] == "af_like":
            drivers = [
                ("arrhythmia_propensity_0_1", float(latents.loc[i,"arrhythmia_propensity_0_1"])),
                ("sleep_debt_0_1", float(latents.loc[i,"sleep_debt_0_1"])),
                ("stress_load_0_1", float(latents.loc[i,"stress_load_0_1"])),
            ]
            mech_ids = ["M_CARDIO_001","M_SLEEP_001","M_STRESS_001"]
        elif row["outcome_type"] == "ili_like_onset":
            drivers = [
                ("inflammatory_state_0_1", float(latents.loc[i,"inflammatory_state_0_1"])),
                ("stress_load_0_1", float(latents.loc[i,"stress_load_0_1"])),
                ("sleep_debt_0_1", float(latents.loc[i,"sleep_debt_0_1"])),
            ]
            mech_ids = ["M_INF_001","M_STRESS_001","M_SLEEP_001"]
        else:
            drivers = [
                ("stress_load_0_1", float(latents.loc[i,"stress_load_0_1"])),
                ("sleep_debt_0_1", float(latents.loc[i,"sleep_debt_0_1"])),
            ]
            mech_ids = ["M_STRESS_001","M_SLEEP_001"]

        # normalize contributions
        vals = np.array([max(0.0, d[1]) for d in drivers], dtype=float)
        if vals.sum() > 0:
            vals = vals / vals.sum()
        drivers_json = json.dumps([{"latent":drivers[j][0], "contribution": float(vals[j])} for j in range(min(top_k,len(drivers)))])

        # simple counterfactuals: reduce top driver by 0.2
        cfs = []
        for j in range(min(counterfactuals_per_event, len(drivers))):
            latent_name = drivers[j][0]
            cfs.append({"do": {latent_name: max(0.0, drivers[j][1]-0.2)}, "risk_delta": float(-0.05 - 0.10*vals[j])})
        counterfactuals_json = json.dumps(cfs)

        records.append({
            "true_drivers_json": drivers_json,
            "precursors_json": json.dumps([]),
            "mechanism_ids_json": json.dumps(mech_ids),
            "counterfactuals_json": counterfactuals_json
        })

    out = outcomes[["event_id","participant_id","timestamp_utc_ms","outcome_type"]].copy()
    for k in ["true_drivers_json","precursors_json","mechanism_ids_json","counterfactuals_json"]:
        out[k] = [r[k] for r in records]
    return out
