"""
RingSentinel — Serving API (Step 7)
======================================

Wraps the trained pipeline (gbm_model.pkl + the flagged-cluster audit
log) behind three endpoints. Does NOT retrain or re-run the pipeline on
request — that's a batch job (run_pipeline / the Colab script), not a
request-time operation. This is the standard, defensible production
pattern: train/batch-score offline, serve pre-computed + light on-demand
scoring online.

Endpoints:
  GET  /health                    liveness check
  GET  /rings                     list all flagged clusters (summary)
  GET  /audit/{cluster_id}        full case file for one flagged cluster
  POST /score-ring                score an arbitrary candidate group of
                                   account_ids on demand

Every response's `action` field is hard-coded to FLAGGED_FOR_HUMAN_REVIEW
or NOT_FLAGGED — there is no code path in this service that can trigger
an automated block/freeze/ban. That's a structural choice, not a config
flag, matching the track's defense-only requirement.

Run locally:
    pip install -r requirements.txt
    uvicorn app.main:app --reload --port 8000

Data dependency: expects a populated ./data/ directory (from the
pipeline script) at DATA_DIR below — mount it as a volume in Docker
rather than baking it into the image, since the dataset/model is a
separate, regenerable artifact from the service code.
"""

import json
import os
import pickle
from collections import defaultdict
from itertools import combinations
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

DATA_DIR = os.environ.get("RINGSENTINEL_DATA_DIR", "data")
RISK_THRESHOLD = 0.5

FEATURE_COLS = ["size", "density", "avg_edge_weight", "account_age_std_days",
                 "txn_velocity_per_day", "chargeback_rate", "avg_txn_amount"]

app = FastAPI(
    title="RingSentinel API",
    description="Explainable abuse-ring detection — defense-only, human-review-gated.",
    version="1.0.0",
)

# Demo-only: allows the static dashboard (served from a different origin/port)
# to call this API from a browser. In real production this would be locked
# to the dashboard's actual deployed origin, not "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Loaded once at startup, not per-request. If data/ is missing or the
# pipeline hasn't been run, fail LOUD at startup, not silently at the
# first request — this is a deliberate production habit, not
# boilerplate: a service that starts "successfully" with no data would
# return confusing 500s later instead of a clear startup error now.
# ------------------------------------------------------------------
_state = {}


@app.on_event("startup")
def load_artifacts():
    required = ["accounts.csv", "devices.csv", "payment_instruments.csv",
                "addresses.csv", "transactions.csv", "gbm_model.pkl",
                "audit_log.jsonl", "baseline_stats.json"]
    missing = [f for f in required if not os.path.exists(f"{DATA_DIR}/{f}")]
    if missing:
        raise RuntimeError(
            f"Missing required files in {DATA_DIR}/: {missing}. "
            f"Run the pipeline script first to generate them."
        )

    _state["accounts"] = pd.read_csv(f"{DATA_DIR}/accounts.csv", parse_dates=["created_at"])
    _state["devices"] = pd.read_csv(f"{DATA_DIR}/devices.csv")
    _state["instruments"] = pd.read_csv(f"{DATA_DIR}/payment_instruments.csv")
    _state["addresses"] = pd.read_csv(f"{DATA_DIR}/addresses.csv")
    _state["transactions"] = pd.read_csv(f"{DATA_DIR}/transactions.csv", parse_dates=["timestamp"])

    with open(f"{DATA_DIR}/gbm_model.pkl", "rb") as f:
        _state["model"] = pickle.load(f)

    with open(f"{DATA_DIR}/baseline_stats.json") as f:
        _state["baselines"] = json.load(f)

    audit_records = []
    with open(f"{DATA_DIR}/audit_log.jsonl") as f:
        for line in f:
            audit_records.append(json.loads(line))
    _state["audit_by_cluster"] = {r["cluster_id"]: r for r in audit_records}
    _state["audit_list"] = audit_records

    # NOTE: labels_HELD_OUT.csv is deliberately never loaded here. The
    # serving layer has no access to ground truth — same discipline as
    # every detection stage in the pipeline.
    print(f"RingSentinel API ready. {len(audit_records)} flagged clusters loaded from {DATA_DIR}/.")


# ------------------------------------------------------------------
# Request/response models
# ------------------------------------------------------------------
class ScoreRingRequest(BaseModel):
    account_ids: List[str] = Field(..., min_items=2, description="Candidate group of account IDs to score")


class RingSummary(BaseModel):
    cluster_id: str
    risk_score: float
    cluster_size: int
    action: str


class ScoreRingResponse(BaseModel):
    risk_score: float
    cluster_size: int
    action: str
    feature_snapshot: dict
    shared_entity_evidence: list
    anomalous_features: list
    note: Optional[str] = None


# ------------------------------------------------------------------
# Shared feature-computation logic (mirrors stage5/stage6 of the
# pipeline, but for an arbitrary account list supplied at request time)
# ------------------------------------------------------------------
def _shared_entity_map(df, id_col, account_ids_set):
    m = {}
    subset = df[df["account_id"].isin(account_ids_set)]
    for entity_id, group in subset.groupby(id_col):
        accts = set(group["account_id"])
        if len(accts) > 1:
            m[entity_id] = accts
    return m


def compute_features_for_accounts(account_ids: List[str]):
    accounts, devices = _state["accounts"], _state["devices"]
    instruments, addresses = _state["instruments"], _state["addresses"]
    txns = _state["transactions"]

    account_ids_set = set(account_ids)
    unknown = account_ids_set - set(accounts["account_id"])
    if unknown:
        raise HTTPException(status_code=404, detail=f"Unknown account_ids: {sorted(unknown)[:5]}")

    accounts_idx = accounts.set_index("account_id")
    member_accounts = accounts_idx.loc[list(account_ids_set)]
    member_txns = txns[txns["account_id"].isin(account_ids_set)]

    device_map = _shared_entity_map(devices, "device_id", account_ids_set)
    instr_map = _shared_entity_map(instruments, "instrument_id", account_ids_set)
    addr_map = _shared_entity_map(addresses, "address_id", account_ids_set)

    edge_signals = defaultdict(set)
    for signal_name, entity_map in (("device", device_map), ("instrument", instr_map), ("address", addr_map)):
        for entity_id, accts in entity_map.items():
            for a, b in combinations(sorted(accts), 2):
                edge_signals[(a, b)].add(signal_name)

    n = len(account_ids_set)
    max_edges = n * (n - 1) / 2
    density = len(edge_signals) / max_edges if max_edges else 0.0
    avg_weight = (sum(len(s) for s in edge_signals.values()) / len(edge_signals)) if edge_signals else 0.0

    age_days = (member_accounts["created_at"].max() - member_accounts["created_at"]).dt.days
    age_std = float(age_days.std()) if len(age_days) > 1 else 0.0

    if len(member_txns) > 0:
        span_days = max((member_txns["timestamp"].max() - member_txns["timestamp"].min()).days, 1)
        txn_velocity = len(member_txns) / span_days
        chargeback_rate = float((member_txns["status"] == "chargeback").mean())
        avg_amount = float(member_txns["amount_inr"].mean())
    else:
        txn_velocity, chargeback_rate, avg_amount = 0.0, 0.0, 0.0

    features = {
        "size": float(n), "density": density, "avg_edge_weight": avg_weight,
        "account_age_std_days": age_std, "txn_velocity_per_day": txn_velocity,
        "chargeback_rate": chargeback_rate, "avg_txn_amount": avg_amount,
    }

    evidence = []
    for signal_name, entity_map in (("device", device_map), ("instrument", instr_map), ("address", addr_map)):
        for entity_id, accts in entity_map.items():
            if len(accts) >= 2:
                evidence.append({"signal": signal_name, "entity_id": entity_id, "accounts_sharing": sorted(accts)})

    return features, evidence


def explain_anomalies(features: dict) -> List[str]:
    baselines = _state["baselines"]
    out = []
    for feat in FEATURE_COLS:
        base = baselines.get(feat, 0)
        val = features[feat]
        if base > 0:
            ratio = val / base
            if ratio >= 1.5 or ratio <= 0.67:
                out.append(f"{feat}={val:.3f} ({ratio:.1f}x candidate-population avg {base:.3f})")
    return out


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "flagged_clusters_loaded": len(_state.get("audit_list", []))}


@app.get("/rings", response_model=List[RingSummary])
def list_rings(limit: int = 50, min_risk_score: float = 0.0):
    records = [r for r in _state["audit_list"] if r["risk_score"] >= min_risk_score]
    records = sorted(records, key=lambda r: r["risk_score"], reverse=True)[:limit]
    return [RingSummary(cluster_id=r["cluster_id"], risk_score=r["risk_score"],
                         cluster_size=r["cluster_size"], action=r["action"]) for r in records]


@app.get("/audit/{cluster_id}")
def get_audit(cluster_id: str):
    record = _state["audit_by_cluster"].get(cluster_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No flagged cluster with id '{cluster_id}'")
    return record


@app.post("/score-ring", response_model=ScoreRingResponse)
def score_ring(req: ScoreRingRequest):
    features, evidence = compute_features_for_accounts(req.account_ids)

    X = pd.DataFrame([features])[FEATURE_COLS]
    risk_score = float(_state["model"].predict_proba(X)[0, 1])
    action = "FLAGGED_FOR_HUMAN_REVIEW" if risk_score >= RISK_THRESHOLD else "NOT_FLAGGED"

    note = None
    if not evidence:
        note = ("No shared device/instrument/address found among these accounts. "
                "Score is based on behavioral features only — treat with extra caution.")

    return ScoreRingResponse(
        risk_score=round(risk_score, 4),
        cluster_size=len(req.account_ids),
        action=action,
        feature_snapshot={k: round(v, 4) for k, v in features.items()},
        shared_entity_evidence=evidence,
        anomalous_features=explain_anomalies(features),
        note=note,
    )