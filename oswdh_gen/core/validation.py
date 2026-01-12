from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

REQUIRED_FILES = [
    "participants.parquet","devices.parquet","time_index.parquet",
    "wearables_features_daily.parquet","wearables_features_minute.parquet",
    "sleep_episodes.parquet","activity_bouts.parquet","symptoms_surveys.parquet",
    "clinical_events.parquet","outcomes.parquet","missingness_log.parquet",
    "ground_truth_latents.parquet","explanations.parquet",
    "federated_partitions.json"
]

def validate_dataset(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    missing = [f for f in REQUIRED_FILES if not (out_dir / f).exists()]
    report = {"ok": len(missing)==0, "missing_files": missing, "checks": {}}

    # Basic referential integrity check (participants referenced elsewhere)
    if report["ok"]:
        participants = pd.read_parquet(out_dir/"participants.parquet")[["participant_id"]]
        pset = set(participants["participant_id"].astype(str).tolist())

        def check_participants(file: str, col: str):
            df = pd.read_parquet(out_dir/file, columns=[col])
            bad = df[~df[col].astype(str).isin(pset)]
            report["checks"][f"{file}:{col}"] = {"bad_rows": int(len(bad))}
            return len(bad)==0

        ok1 = check_participants("wearables_features_daily.parquet","participant_id")
        ok2 = check_participants("wearables_features_minute.parquet","participant_id")
        ok3 = check_participants("outcomes.parquet","participant_id")
        report["ok"] = report["ok"] and ok1 and ok2 and ok3

    return report
