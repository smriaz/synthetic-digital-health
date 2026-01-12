from __future__ import annotations
import hashlib, json
from pathlib import Path
from datetime import datetime, timezone

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def write_manifest(out_dir: Path, *, dataset_name: str, dataset_version: str, generator_version: str,
                   git_commit: str, config_sha256: str, schema_version: str, seed: int, tracks_enabled: list[str],
                   partitions: list[str]) -> Path:
    tables = {}
    for p in out_dir.rglob("*.parquet"):
        rel = str(p.relative_to(out_dir))
        tables[rel] = {"rows": None, "sha256": sha256_file(p)}
    manifest = {
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "generator_version": generator_version,
        "generator_git_commit": git_commit,
        "config_sha256": config_sha256,
        "schema_version": schema_version,
        "seed": seed,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "tables": tables,
        "tracks_enabled": tracks_enabled,
        "partitions": partitions,
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
