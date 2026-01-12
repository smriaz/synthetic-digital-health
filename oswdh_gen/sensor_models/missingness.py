from __future__ import annotations
import numpy as np
import pandas as pd

def apply_missingness(rng: np.random.Generator, minute_df: pd.DataFrame, *, daily_prob_nonwear: float,
                      mean_episode_minutes: int, daily_prob_upload: float) -> tuple[pd.DataFrame, list[dict]]:
    logs = []
    n = len(minute_df)
    minute_df = minute_df.copy()
    minute_df["missing_flag"] = False
    minute_df["missing_reason"] = "none"

    # Nonwear episode
    if rng.random() < daily_prob_nonwear:
        dur = int(max(5, rng.exponential(mean_episode_minutes)))
        dur = min(dur, n)
        start = int(rng.integers(0, n-dur+1))
        end = start + dur
        minute_df.loc[start:end-1, "missing_flag"] = True
        minute_df.loc[start:end-1, "missing_reason"] = "nonwear"
        logs.append({"missing_type":"nonwear","start_idx":start,"end_idx":end,"severity_0_1":min(1.0,dur/n)})
    # Upload failure (entire day)
    if rng.random() < daily_prob_upload:
        minute_df["missing_flag"] = True
        minute_df["missing_reason"] = "upload"
        logs.append({"missing_type":"upload_failure","start_idx":0,"end_idx":n,"severity_0_1":1.0})
    return minute_df, logs
