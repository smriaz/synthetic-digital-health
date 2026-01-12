from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import pandas as pd

@dataclass(frozen=True)
class Timebase:
    start_date_utc: str  # YYYY-MM-DD
    duration_days: int

    def build_minute_index(self) -> pd.DataFrame:
        start = datetime.fromisoformat(self.start_date_utc).replace(tzinfo=timezone.utc)
        minutes = self.duration_days * 24 * 60
        ts = [start + timedelta(minutes=i) for i in range(minutes)]
        df = pd.DataFrame({
            "timestamp_utc_ms": [int(t.timestamp() * 1000) for t in ts],
            "day_index": [i // (24*60) for i in range(minutes)],
            "minute_index": [i % (24*60) for i in range(minutes)],
        })
        # weekday 0=Mon..6=Sun
        df["weekday"] = [(t.weekday()) for t in ts]
        return df

    def build_day_index(self) -> pd.DataFrame:
        start = datetime.fromisoformat(self.start_date_utc).replace(tzinfo=timezone.utc)
        ts = [start + timedelta(days=i) for i in range(self.duration_days)]
        return pd.DataFrame({
            "day_index": list(range(self.duration_days)),
            "day_start_ts_utc_ms": [int(t.timestamp() * 1000) for t in ts],
        })
