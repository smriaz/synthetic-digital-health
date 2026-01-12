from pathlib import Path
import subprocess, json, os

def test_smoke_generate_and_validate(tmp_path):
    out = tmp_path/"data"
    cfg = Path("configs/templates/small.yaml")
    subprocess.check_call(["python","-m","oswdh_gen.cli","generate", str(cfg), "--out", str(out)])
    subprocess.check_call(["python","-m","oswdh_gen.cli","validate-dataset", str(out)])
