# RingSentinel API — Deployment

## What's verified vs. what you need to check yourself

The app logic (`app/main.py`) has been tested directly — all four
endpoints (`/health`, `/rings`, `/audit/{cluster_id}`, `/score-ring`)
work correctly against real pipeline output. The **Dockerfile has not
been build-tested** (no Docker available in the environment this was
built in) — it follows standard, low-risk conventions, but run the
build yourself before you trust it in your pitch/demo.

## Local run (no Docker) — do this first

```bash
cd api
pip install -r requirements.txt
# data/ must already exist — copy it from wherever the pipeline
# script (ringsentinel_full_pipeline.py) wrote it:
cp -r ../data .
uvicorn app.main:app --reload --port 8000
```

Then check:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/rings?limit=5
```

## Docker

```bash
cd api
docker build -t ringsentinel-api .
docker run -v $(pwd)/data:/app/data -p 8000:8000 ringsentinel-api
```

If the build fails on the `xgboost` install specifically, the usual
fix is switching the base image from `python:3.11-slim` to
`python:3.11` (non-slim has more build tooling preinstalled) — slim
images occasionally need `apt-get install -y libgomp1` for xgboost's
OpenMP dependency. Worth testing this now, not during your demo.

## Endpoints

- `GET /health` — liveness + how many flagged clusters are loaded
- `GET /rings?limit=50&min_risk_score=0.0` — list flagged clusters, highest risk first
- `GET /audit/{cluster_id}` — full case file: evidence, features, anomalies
- `POST /score-ring` — body `{"account_ids": ["acc_...", "acc_...", ...]}`, scores an arbitrary candidate group on demand

## What NOT to change

`action` is always `FLAGGED_FOR_HUMAN_REVIEW` or `NOT_FLAGGED`. Don't
add an auto-block/auto-freeze action to this service, even for the
demo — it's the one hard compliance line in the track brief.
