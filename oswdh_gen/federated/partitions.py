from __future__ import annotations
import json
import numpy as np

def make_splits(rng: np.random.Generator, participant_ids: list[str], splits: dict) -> dict:
    p = participant_ids.copy()
    rng.shuffle(p)
    n = len(p)
    n_train = int(n * splits.get("train", 0.7))
    n_val = int(n * splits.get("val", 0.1))
    train = p[:n_train]
    val = p[n_train:n_train+n_val]
    test = p[n_train+n_val:]
    return {"train": train, "val": val, "test": test}

def build_partitions(participants_df, devices_df, *, strategies: list[str], splits: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    out = {"schema_version":"1.0","partition_strategies":{}}
    pids = participants_df["participant_id"].astype(str).tolist()

    if "by_site" in strategies:
        clients = {}
        for site, g in participants_df.groupby("site_id"):
            clients[str(site)] = g["participant_id"].astype(str).tolist()
        out["partition_strategies"]["by_site"] = {"clients": clients, "splits": {k: make_splits(rng, v, splits) for k,v in clients.items()}}

    if "by_vendor" in strategies:
        dev_vendor = devices_df.set_index("device_id")["vendor"].to_dict()
        clients = {}
        for _, row in participants_df.iterrows():
            vendor = dev_vendor.get(row["device_id"], "unknown")
            clients.setdefault(str(vendor), []).append(str(row["participant_id"]))
        out["partition_strategies"]["by_vendor"] = {"clients": clients, "splits": {k: make_splits(rng, v, splits) for k,v in clients.items()}}

    if "hybrid" in strategies:
        dev_vendor = devices_df.set_index("device_id")["vendor"].to_dict()
        clients = {}
        for _, row in participants_df.iterrows():
            key = f"{row['site_id']}|{dev_vendor.get(row['device_id'],'unknown')}"
            clients.setdefault(str(key), []).append(str(row["participant_id"]))
        out["partition_strategies"]["hybrid"] = {"clients": clients, "splits": {k: make_splits(rng, v, splits) for k,v in clients.items()}}

    return out
