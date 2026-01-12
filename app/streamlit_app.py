import streamlit as st
from pathlib import Path
import yaml
import subprocess
import pandas as pd

st.set_page_config(page_title="OSWDH Generator", layout="wide")

st.title("OSWDH — Synthetic Wearables Digital Health Dataset Generator")
st.caption("Features-only • v1.0 schema • Streamlit Community Cloud compatible")

st.sidebar.header("Configuration")
cfg_path = st.sidebar.selectbox(
    "Choose a config",
    ["configs/published_v1.yaml", "configs/templates/small.yaml"],
    index=0
)

out_dir = st.sidebar.text_input("Output folder (relative)", "data/oswdh_run")

with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

st.sidebar.subheader("Quick knobs (safe)")
cfg["population"]["n_participants"] = st.sidebar.slider("Participants", 50, 5000, int(cfg["population"]["n_participants"]), step=50)
cfg["study"]["duration_days"] = st.sidebar.slider("Duration (days)", 7, 180, int(cfg["study"]["duration_days"]), step=1)
cfg["seed"] = st.sidebar.number_input("Seed", min_value=1, max_value=10_000_000, value=int(cfg.get("seed", 1337)))

st.subheader("Config preview")
st.code(yaml.safe_dump(cfg, sort_keys=False), language="yaml")

run = st.button("Generate dataset")
if run:
    # Write a temp config for this run
    tmp_cfg = Path("configs") / "_streamlit_run.yaml"
    tmp_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    st.info("Generating... (this may take a bit for larger settings on Community Cloud)")
    cmd = ["python", "-m", "oswdh_gen.cli", "generate", str(tmp_cfg), "--out", out_dir]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        st.error("Generation failed")
        st.code(res.stderr)
    else:
        st.success("Generation complete")
        st.code(res.stdout if res.stdout else "(no stdout)")

        # Show a small preview
        p = Path(out_dir)
        if (p/"participants.parquet").exists():
            participants = pd.read_parquet(p/"participants.parquet").head(20)
            st.subheader("Participants (preview)")
            st.dataframe(participants, use_container_width=True)

        if (p/"wearables_features_daily.parquet").exists():
            daily = pd.read_parquet(p/"wearables_features_daily.parquet").head(20)
            st.subheader("Daily features (preview)")
            st.dataframe(daily, use_container_width=True)

        if (p/"outcomes.parquet").exists():
            outcomes = pd.read_parquet(p/"outcomes.parquet").head(20)
            st.subheader("Outcomes (preview)")
            st.dataframe(outcomes, use_container_width=True)

st.markdown("---")
st.subheader("Notes for Streamlit Community Cloud")
st.markdown(
"""
- Keep default settings small for responsiveness.
- Use the CLI locally for very large datasets, then upload artifacts to your storage.
- This app stays thin: it writes a config and calls the generator.
"""
)
