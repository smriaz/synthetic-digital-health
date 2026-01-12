from __future__ import annotations
from pathlib import Path
import pandas as pd

def write_stats_report(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    daily = pd.read_parquet(out_dir/"wearables_features_daily.parquet")
    outcomes = pd.read_parquet(out_dir/"outcomes.parquet")
    participants = pd.read_parquet(out_dir/"participants.parquet")

    html = []
    html.append("<h1>OSWDH Stats Report</h1>")
    html.append(f"<p>Participants: {len(participants):,}</p>")
    html.append(f"<p>Daily rows: {len(daily):,}</p>")
    html.append(f"<p>Outcomes: {len(outcomes):,}</p>")

    html.append("<h2>Daily summary (selected columns)</h2>")
    html.append(daily[["steps","resting_hr_bpm","hrv_rmssd_ms_proxy","sleep_duration_min","stress_index"]].describe().to_html())

    html.append("<h2>Outcomes by type</h2>")
    html.append(outcomes["outcome_type"].value_counts().to_frame("count").to_html())

    path = out_dir/"stats_report.html"
    path.write_text("\n".join(html), encoding="utf-8")
    return path
