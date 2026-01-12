# OSWDH (Open Synthetic Wearables for Digital Health)

This repository contains:

- **oswdh_gen/**: a config-driven synthetic dataset generator (features-only, v1.0 schema)
- **app/**: a Streamlit app compatible with **Streamlit Community Cloud**
- **schemas/**: dataset schemas and causal metadata (v1.0)
- **configs/**: published config snapshot and templates
- **tests/**: lightweight CI checks (schema + determinism + integrity)

## Quickstart (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Generate dataset
python -m oswdh_gen.cli generate configs/published_v1.yaml --out data/oswdh_v1

# Validate dataset
python -m oswdh_gen.cli validate-dataset data/oswdh_v1
```

## Streamlit App

```bash
streamlit run app/streamlit_app.py
```

## Licenses
- Code: Apache-2.0
- Dataset outputs: CC BY-NC 4.0 (see docs/dataset_card.md)
