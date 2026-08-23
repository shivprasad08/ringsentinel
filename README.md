# RingSentinel — Explainable Abuse-Ring Detection

Detects coordinated abuse rings (accounts sharing hidden device/
instrument/address infrastructure) on a payment platform, via a
three-stage graph pipeline (hard-link connected components → Louvain
soft-link clustering → GBM ring scorer), with full explainability and
a defense-only, human-review-gated action policy.

Built for the Razorpay AI Buildathon, Track 2: AI Risk Manager.

## Repo structure

```
ringsentinel/
├── pipeline/                  # detection pipeline (batch)
│   ├── config.py               single source of truth for all params
│   ├── utils.py                 shared ID/hash helpers
│   ├── generate_dataset.py      Stage 1 — synthetic data + injected rings
│   ├── build_graph.py           Stage 2 — entity graph construction
│   ├── hard_link_detection.py   Stage 3 — connected components
│   ├── louvain_detection.py     Stage 4 — soft-link community detection
│   ├── gbm_scorer.py            Stage 5 — GBM classifier + held-out eval
│   └── audit_layer.py           Stage 6 — explainability + audit trail
├── run_pipeline.py             orchestrator — runs all 6 stages in order
├── api/                        # serving layer (FastAPI)
│   ├── app/main.py               /rings, /audit/{id}, /score-ring
│   ├── Dockerfile
│   └── requirements.txt
├── notebooks/
│   └── colab_full_pipeline_backup.py   single-file version (Colab-era)
├── data/                       # generated — gitignored, not committed
├── DATASET_SPEC.md             synthetic dataset design doc
└── requirements.txt             pipeline dependencies
```

## Running it

```bash
# 1. Pipeline — generates data, trains, evaluates, writes audit log
pip install -r requirements.txt
python run_pipeline.py

# 2. API — serves the trained model + audit log
cd api
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Or run a single stage while iterating: `python -m pipeline.gbm_scorer`.

## Key design decisions (for the pitch / Q&A)

- **Held-out evaluation discipline**: `labels_HELD_OUT.csv` is loaded
  ONLY in `gbm_scorer.py` (to train) and `audit_layer.py` (to compute
  case evidence for already-flagged clusters). No detection stage
  (hard-link, Louvain) ever touches it.
- **Reproducibility**: all randomness — including generated IDs — goes
  through the seeded `random` module, not `uuid.uuid4()` (which
  ignores seeding). Same `CONFIG["seed"]` always produces byte-identical
  output.
- **Defense-only**: every audit record's `action` field is hard-coded
  to `FLAGGED_FOR_HUMAN_REVIEW`. No code path in this repo can trigger
  an automated block/freeze.
- **Serving pattern**: the API loads pre-trained artifacts at startup
  and scores on request — it never retrains per-request.

See `DATASET_SPEC.md` for the synthetic data design (including the
easy/medium/hard difficulty presets for stress-testing the eval).
