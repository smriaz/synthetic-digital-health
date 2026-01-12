from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

from oswdh_gen.core.rng import RNG
from oswdh_gen.core.timebase import Timebase
from oswdh_gen.core.ids import make_id
from oswdh_gen.core.manifest import write_manifest
from oswdh_gen.core.validation import validate_dataset
from oswdh_gen.core.stats_report import write_stats_report

from oswdh_gen.population.demographics import generate_participants
from oswdh_gen.latents.latent_state import generate_latents_minute
from oswdh_gen.sensor_models.derived_features import minute_features_from_latents
from oswdh_gen.sensor_models.missingness import apply_missingness
from oswdh_gen.disease_models.outcomes import generate_outcomes
from oswdh_gen.xai.explanations import build_explanations
from oswdh_gen.federated.partitions import build_partitions
from oswdh_gen.agents.heuristic_agent import heuristic_decision, reward_from_outcomes

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def cmd_generate(cfg_path: Path, out_dir: Path) -> None:
    cfg_text = Path(cfg_path).read_text(encoding="utf-8")
    cfg = yaml.safe_load(cfg_text)
    config_sha = sha256_text(cfg_text)

    seed = int(cfg.get("seed", 1337))
    rng_wrap = RNG(seed)
    rng_pop = rng_wrap.child("population")
    rng_lat = rng_wrap.child("latents")
    rng_sens = rng_wrap.child("sensors")
    rng_out = rng_wrap.child("outcomes")
    rng_xai = rng_wrap.child("xai")
    rng_agent = rng_wrap.child("agent")

    tb = Timebase(start_date_utc=cfg["study"]["start_date_utc"], duration_days=int(cfg["study"]["duration_days"]))
    minute_index = tb.build_minute_index()
    day_index = tb.build_day_index()

    # Sites
    n_sites = int(cfg.get("sites", {}).get("n_sites", 5))
    site_ids = [f"site_{i:03d}" for i in range(n_sites)]

    # Devices
    vendors = cfg.get("devices", {}).get("vendors", {"VendorA":0.5,"VendorB":0.3,"VendorC":0.2})
    vendor_keys = list(vendors.keys())
    vendor_probs = np.array([vendors[k] for k in vendor_keys], dtype=float)
    vendor_probs = vendor_probs / vendor_probs.sum()
    # Make a device per participant for simplicity in v1 generator
    n = int(cfg["population"]["n_participants"])
    device_ids = [make_id("dev", str(i), str(seed)) for i in range(n)]
    device_vendor = rng_pop.choice(vendor_keys, size=n, p=vendor_probs)
    devices_df = pd.DataFrame({
        "device_id": device_ids,
        "vendor": device_vendor,
        "model": ["model_1"]*n,
        "sensor_profile_id": ["sp_1"]*n,
        "firmware_schedule_id": ["fw_1"]*n,
        "has_acc": [True]*n,
        "has_ppg": [True]*n,
        "has_temp": [True]*n,
        "has_spo2": [True]*n,
        "acc_hz": [50]*n,
        "ppg_hz": [50]*n,
        "feature_algo_version": ["fa_v1"]*n
    })

    # Participants
    demo_df = generate_participants(
        rng_pop,
        n=n,
        age_range=cfg["population"]["age_range"],
        sex_dist=cfg["population"]["sex_distribution"],
        socio_params=cfg["population"].get("socioeconomic_index", {}),
        comorb_params=cfg["population"].get("comorbidity_index", {}),
    )
    participant_ids = [make_id("p", str(i), str(seed)) for i in range(n)]
    participants_df = demo_df.copy()
    participants_df.insert(0, "participant_id", participant_ids)
    participants_df["site_id"] = rng_pop.choice(site_ids, size=n)
    participants_df["device_id"] = device_ids
    # enrollment / end timestamps
    start_ms = int(minute_index["timestamp_utc_ms"].iloc[0])
    end_ms = int(minute_index["timestamp_utc_ms"].iloc[-1])
    participants_df["enrollment_ts_utc_ms"] = start_ms
    participants_df["study_end_ts_utc_ms"] = end_ms

    # Export base tables
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    participants_df.to_parquet(out_dir/"participants.parquet", index=False)
    devices_df.to_parquet(out_dir/"devices.parquet", index=False)
    minute_index.to_parquet(out_dir/"time_index.parquet", index=False)

    # Generate per participant minute features, daily aggregates, outcomes, logs
    daily_rows = []
    minute_rows = []
    sleep_rows = []
    act_rows = []
    survey_rows = []
    clinical_rows = []
    outcome_rows = []
    missing_rows = []
    latent_rows = []
    expl_rows = []
    agent_logs = []

    base_noise = float(cfg.get("devices", {}).get("noise_profiles", {}).get("base_noise_0_1", 0.25))
    miss_cfg = cfg.get("missingness", {})
    nonwear_p = float(miss_cfg.get("nonwear", {}).get("daily_probability", 0.15))
    nonwear_mean = int(miss_cfg.get("nonwear", {}).get("mean_episode_minutes", 180))
    upload_p = float(miss_cfg.get("upload_failure", {}).get("daily_probability", 0.03))

    tracks = cfg.get("tracks", {})
    xai_cfg = cfg.get("xai", {})
    top_k = int(xai_cfg.get("explanations", {}).get("top_k_drivers", 5))
    cf_n = int(xai_cfg.get("explanations", {}).get("counterfactuals_per_event", 3))

    for pi, prow in participants_df.iterrows():
        pid = prow["participant_id"]
        # participant-specific noise multiplier by vendor
        vendor = devices_df.loc[pi, "vendor"]
        vendor_mult = {"VendorA":1.0,"VendorB":1.15,"VendorC":1.3}.get(vendor, 1.0)
        pn = base_noise * vendor_mult

        # latents
        lat = generate_latents_minute(rng_lat, minute_index, prow.to_dict())
        lat.insert(0, "participant_id", pid)
        lat.insert(1, "timestamp_utc_ms", minute_index["timestamp_utc_ms"].to_numpy())
        lat.insert(2, "day_index", minute_index["day_index"].to_numpy())
        latent_rows.append(lat)

        # features
        feat = minute_features_from_latents(rng_sens, minute_index, lat.drop(columns=["participant_id","timestamp_utc_ms","day_index"]), pn)
        # missingness per day
        feat = pd.concat([minute_index[["timestamp_utc_ms","day_index","minute_index"]].reset_index(drop=True), feat.reset_index(drop=True)], axis=1)

        # apply missingness day by day
        for d in range(int(cfg["study"]["duration_days"])):
            sl = feat[feat["day_index"]==d].copy().reset_index(drop=True)
            sl2, logs = apply_missingness(rng_sens, sl, daily_prob_nonwear=nonwear_p, mean_episode_minutes=nonwear_mean, daily_prob_upload=upload_p)
            feat.loc[feat["day_index"]==d, "missing_flag"] = sl2["missing_flag"].to_numpy()
            feat.loc[feat["day_index"]==d, "missing_reason"] = sl2["missing_reason"].to_numpy()
            # missingness log rows
            for lg in logs:
                start_ts = int(sl2.loc[lg["start_idx"], "timestamp_utc_ms"])
                end_ts = int(sl2.loc[lg["end_idx"]-1, "timestamp_utc_ms"])
                missing_rows.append({
                    "missingness_id": make_id("miss", pid, str(d), lg["missing_type"]),
                    "participant_id": pid,
                    "start_ts_utc_ms": start_ts,
                    "end_ts_utc_ms": end_ts,
                    "missing_type": lg["missing_type"],
                    "severity_0_1": float(lg["severity_0_1"]),
                    "affected_stream": "minute_features",
                    "notes": ""
                })

        # mask observed values if missing
        miss = feat["missing_flag"].to_numpy()
        for col in ["hr_bpm","skin_temp_c","spo2_pct"]:
            feat.loc[miss, col] = np.nan
        feat.loc[miss, "hr_confidence"] = 0.0
        feat.loc[miss, "ppg_quality"] = 0.0
        feat.insert(0, "participant_id", pid)
        minute_rows.append(feat)

        # daily aggregates
        for d in range(int(cfg["study"]["duration_days"])):
            day_slice = feat[feat["day_index"]==d].copy()
            # minimal daily aggregation
            wear_minutes = int((~day_slice["missing_flag"]).sum())
            nonwear_minutes = int((day_slice["missing_flag"]).sum())
            artifact_score = float(1.0 - day_slice["ppg_quality"].mean()) if len(day_slice) else 1.0
            conf = float(day_slice["hr_confidence"].mean()) if len(day_slice) else 0.0
            acc_mean = float(day_slice["acc_intensity"].mean()) if len(day_slice) else 0.0
            steps = int((day_slice["acc_intensity"].fillna(0).to_numpy() * 120).sum())
            sleep_mask = (day_slice["activity_state_pred"].astype(str)=="sleep")
            sleep_minutes = int(sleep_mask.sum())
            resting_hr = float(day_slice.loc[~sleep_mask, "hr_bpm"].quantile(0.1)) if (~sleep_mask).any() else float(np.nanmedian(day_slice["hr_bpm"].to_numpy()))
            avg_hr = float(np.nanmean(day_slice["hr_bpm"].to_numpy()))
            stress_index = float(np.clip(lat.loc[lat["day_index"]==d, "stress_load_0_1"].mean(),0,1))
            # HRV proxy (store day-level)
            hrv_proxy = float(np.clip(70 - 35*stress_index - 20*float(lat.loc[lat["day_index"]==d, "sleep_debt_0_1"].mean()),0,200))

            daily_rows.append({
                "participant_id": pid,
                "day_index": d,
                "day_start_ts_utc_ms": int(day_index.loc[day_index["day_index"]==d, "day_start_ts_utc_ms"].iloc[0]),
                "wear_minutes": wear_minutes,
                "nonwear_minutes": nonwear_minutes,
                "artifact_score": float(np.clip(artifact_score,0,1)),
                "feature_confidence": float(np.clip(conf,0,1)),
                "steps": steps,
                "active_minutes": int((day_slice["acc_intensity"]>0.35).sum()),
                "sedentary_minutes": int((day_slice["acc_intensity"]<=0.15).sum()),
                "resting_hr_bpm": float(np.clip(resting_hr,30,120)) if not np.isnan(resting_hr) else 60.0,
                "avg_hr_bpm": float(np.clip(avg_hr,30,220)) if not np.isnan(avg_hr) else 70.0,
                "hrv_rmssd_ms_proxy": hrv_proxy,
                "resp_rate_rpm_proxy": float(np.clip(avg_hr/6.0 if not np.isnan(avg_hr) else 12.0,5,40)),
                "skin_temp_c": float(np.nanmean(day_slice["skin_temp_c"])) if len(day_slice) else 33.0,
                "sleep_duration_min": sleep_minutes,
                "sleep_efficiency": float(np.clip(sleep_minutes/480.0,0,1)),
                "rem_min": 0,
                "deep_min": 0,
                "awakenings_count": int(np.clip((1.0-(sleep_minutes/480.0))*10,0,50)),
                "strain_index": float(np.clip(acc_mean,0,1)),
                "recovery_index": float(np.clip(conf*(1.0-artifact_score),0,1)),
                "stress_index": float(stress_index),
            })

        # outcomes from latents
        outs = generate_outcomes(rng_out, minute_index, lat.drop(columns=["participant_id","timestamp_utc_ms","day_index"]), prow.to_dict(), tracks)
        if not outs.empty:
            outs.insert(0, "participant_id", pid)
            outs["event_id"] = [make_id("evt", pid, str(i), str(int(t))) for i,t in enumerate(outs["timestamp_utc_ms"].tolist())]
            outcome_rows.append(outs)

    # concatenate and write
    daily_df = pd.DataFrame(daily_rows)
    minute_df = pd.concat(minute_rows, ignore_index=True)
    latents_df = pd.concat(latent_rows, ignore_index=True)
    outcomes_df = pd.concat(outcome_rows, ignore_index=True) if len(outcome_rows) else pd.DataFrame(columns=[
        "participant_id","event_id","timestamp_utc_ms","day_index","outcome_type","severity_0_1",
        "outcome_window_start_ts_utc_ms","outcome_window_end_ts_utc_ms","label_source","label_noise_flag","label_delay_days","notes"
    ])

    # placeholder empty tables for v1 contract (populated more fully as you extend)
    sleep_df = pd.DataFrame(columns=["sleep_episode_id","participant_id","start_ts_utc_ms","end_ts_utc_ms","day_index",
                                     "sleep_onset_latency_min","wake_after_sleep_onset_min","rem_min","deep_min","light_min",
                                     "awake_min","efficiency","confidence"])
    activity_df = pd.DataFrame(columns=["activity_bout_id","participant_id","start_ts_utc_ms","end_ts_utc_ms","day_index",
                                        "activity_type","intensity","steps","confidence"])
    surveys_df = pd.DataFrame(columns=["survey_id","participant_id","timestamp_utc_ms","day_index","survey_type",
                                       "fatigue_0_10","pain_0_10","palpitations_0_10","stress_0_10","mood_0_10",
                                       "adherence_self_report_0_10","completed","missing_reason"])
    clinical_df = pd.DataFrame(columns=["clinical_event_id","participant_id","site_id","timestamp_utc_ms","event_type",
                                        "code_system","code","value_num","value_text","unit","label_delay_days","confidence"])
    missing_df = pd.DataFrame(missing_rows) if len(missing_rows) else pd.DataFrame(columns=[
        "missingness_id","participant_id","start_ts_utc_ms","end_ts_utc_ms","missing_type","severity_0_1","affected_stream","notes"
    ])

    # explanations
    explanations_df = build_explanations(rng_xai, outcomes_df, latents_df[latents_df["participant_id"]==latents_df["participant_id"]], minute_index, top_k, cf_n)
    # NOTE: build_explanations expects per-participant in later refactor; for now it works for empty/placeholder.
    # We'll build a simpler explanations generation for now:
    if not outcomes_df.empty:
        # make one explanation per event using global latents slice at matching timestamp
        idx = pd.Series(range(len(minute_index)), index=minute_index["timestamp_utc_ms"].astype("int64"))
        recs = []
        for _, r in outcomes_df.iterrows():
            t = int(r["timestamp_utc_ms"])
            i = int(idx.get(t, 0))
            # choose drivers
            drivers = [
                {"latent":"arrhythmia_propensity_0_1","contribution":0.5},
                {"latent":"sleep_debt_0_1","contribution":0.3},
                {"latent":"stress_load_0_1","contribution":0.2},
            ]
            recs.append({
                "event_id": r["event_id"],
                "participant_id": r["participant_id"],
                "timestamp_utc_ms": r["timestamp_utc_ms"],
                "outcome_type": r["outcome_type"],
                "true_drivers_json": json.dumps(drivers),
                "precursors_json": json.dumps([]),
                "mechanism_ids_json": json.dumps(["M_CARDIO_001","M_SLEEP_001","M_STRESS_001"]),
                "counterfactuals_json": json.dumps([{"do":{"sleep_debt_0_1":0.1},"risk_delta":-0.1}])
            })
        explanations_df = pd.DataFrame(recs)
    else:
        explanations_df = pd.DataFrame(columns=["event_id","participant_id","timestamp_utc_ms","outcome_type","true_drivers_json","precursors_json","mechanism_ids_json","counterfactuals_json"])

    # partitions
    fed_cfg = cfg.get("federated", {})
    partitions = build_partitions(participants_df, devices_df, strategies=fed_cfg.get("strategies",["by_site"]), splits=fed_cfg.get("splits",{}), seed=seed)
    (out_dir/"federated_partitions.json").write_text(json.dumps(partitions, indent=2), encoding="utf-8")

    # agent logs from daily + outcomes
    if cfg.get("agents", {}).get("enabled", True):
        agent_dir = out_dir/"agent_logs/heuristic_agent"
        agent_dir.mkdir(parents=True, exist_ok=True)
        # Build outcome counts per participant-day
        if not outcomes_df.empty:
            oc = outcomes_df.groupby(["participant_id","day_index"]).size().to_dict()
        else:
            oc = {}
        # Write one jsonl file
        path = agent_dir/"episodes.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for _, dr in daily_df.iterrows():
                pid = dr["participant_id"]
                d = int(dr["day_index"])
                action_type, action_payload, meta = heuristic_decision(dr.to_dict())
                alerts = 1 if action_type in ("send_survey","recommend","escalate") else 0
                reward = reward_from_outcomes(int(oc.get((pid,d),0)), alerts)
                obj = {
                    "episode_id": make_id("ep", pid, str(d), str(seed)),
                    "participant_id": pid,
                    "day_index": d,
                    "timestamp_utc_ms": int(dr["day_start_ts_utc_ms"]),
                    "observation": {"daily": {k: (float(v) if isinstance(v,(int,float,np.floating)) and not np.isnan(v) else v) for k,v in dr.to_dict().items()}},
                    "plan_text": meta["plan"],
                    "tools_used": ["trend_analyzer_v1","risk_model_v1"],
                    "action": {"type": action_type, **action_payload},
                    "reward": reward,
                    "safety_flags": {"alert_fatigue_risk": False, "over_escalation": False}
                }
                f.write(json.dumps(obj) + "\n")

    # write parquet files
    daily_df.to_parquet(out_dir/"wearables_features_daily.parquet", index=False)
    minute_df.to_parquet(out_dir/"wearables_features_minute.parquet", index=False)
    sleep_df.to_parquet(out_dir/"sleep_episodes.parquet", index=False)
    activity_df.to_parquet(out_dir/"activity_bouts.parquet", index=False)
    surveys_df.to_parquet(out_dir/"symptoms_surveys.parquet", index=False)
    clinical_df.to_parquet(out_dir/"clinical_events.parquet", index=False)
    outcomes_df.to_parquet(out_dir/"outcomes.parquet", index=False)
    missing_df.to_parquet(out_dir/"missingness_log.parquet", index=False)
    latents_df.to_parquet(out_dir/"ground_truth_latents.parquet", index=False)
    explanations_df.to_parquet(out_dir/"explanations.parquet", index=False)

    # optional report + manifest
    if cfg.get("export", {}).get("write_stats_report", True):
        write_stats_report(out_dir)

    if cfg.get("export", {}).get("write_manifest", True):
        write_manifest(
            out_dir,
            dataset_name="OSWDH",
            dataset_version=str(cfg.get("dataset_version","1.0")),
            generator_version="gen-1.0.0",
            git_commit="unknown",
            config_sha256=config_sha,
            schema_version=str(cfg.get("schema_version","1.0")),
            seed=seed,
            tracks_enabled=list(cfg.get("tracks", {}).keys()),
            partitions=fed_cfg.get("strategies",[]),
        )

def cmd_validate_dataset(out_dir: Path) -> None:
    rep = validate_dataset(Path(out_dir))
    print(json.dumps(rep, indent=2))
    if not rep["ok"]:
        raise SystemExit(2)

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("config")
    g.add_argument("--out", required=True)
    v = sub.add_parser("validate-dataset")
    v.add_argument("out_dir")
    args = ap.parse_args()

    if args.cmd == "generate":
        cmd_generate(Path(args.config), Path(args.out))
    elif args.cmd == "validate-dataset":
        cmd_validate_dataset(Path(args.out_dir))

if __name__ == "__main__":
    main()
