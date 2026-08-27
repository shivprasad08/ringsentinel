"""
RingSentinel — Anomaly Detection Layer (Module 1)
=====================================================

Unsupervised IsolationForest over cluster-level features, run in
PARALLEL to the GBM scorer. Two call modes:

  1. From run_pipeline.py (in-process): main(gbm_model=..., cluster_features=...)
     Uses the already-trained model and already-built features — no
     disk reload, no re-fitting the GBM.

  2. Standalone: python -m pipeline.anomaly_layer [--features a,b,c] [--contamination 0.15]
     Loads gbm_model.pkl and rebuilds cluster_features from data/ CSVs.

Both modes save data/combined_flags.csv and print the same value
assessment (does the anomaly layer catch true rings the GBM missed).
"""

import argparse
import pickle

import pandas as pd
from sklearn.ensemble import IsolationForest

from .config import DATA_DIR, FEATURE_COLS

DEFAULT_CONTAMINATION = 0.15


def _load_from_disk(data_dir):
    """Standalone-mode fallback: rebuild what the orchestrator would
    otherwise hand us directly."""
    from .gbm_scorer import build_cluster_features, attach_labels

    flags_df = pd.read_csv(f"{data_dir}/louvain_flags.csv")
    accounts = pd.read_csv(f"{data_dir}/accounts.csv", parse_dates=["created_at"])
    txns = pd.read_csv(f"{data_dir}/transactions.csv", parse_dates=["timestamp"])
    labels = pd.read_csv(f"{data_dir}/labels_HELD_OUT.csv")  # eval-only

    cluster_features = attach_labels(build_cluster_features(flags_df, accounts, txns), labels)

    with open(f"{data_dir}/gbm_model.pkl", "rb") as f:
        gbm_model = pickle.load(f)

    return gbm_model, cluster_features


def main(gbm_model=None, cluster_features=None, data_dir=DATA_DIR,
         feature_cols=None, contamination=DEFAULT_CONTAMINATION):
    if gbm_model is None or cluster_features is None:
        gbm_model, cluster_features = _load_from_disk(data_dir)

    feature_cols = feature_cols or FEATURE_COLS
    unknown = set(feature_cols) - set(FEATURE_COLS)
    if unknown:
        raise ValueError(f"Unknown feature(s) {unknown}. Valid: {FEATURE_COLS}")

    cf = cluster_features.copy()

    iso_model = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    iso_model.fit(cf[feature_cols])
    # higher = more anomalous (flip sklearn's default sign convention)
    cf["anomaly_score"] = -iso_model.decision_function(cf[feature_cols])
    cf["is_anomaly_flagged"] = iso_model.predict(cf[feature_cols]) == -1

    gbm_proba = gbm_model.predict_proba(cf[FEATURE_COLS])[:, 1]
    cf["gbm_flagged"] = gbm_proba >= 0.5
    cf["gbm_risk_score"] = gbm_proba

    def reason(row):
        if row["gbm_flagged"] and row["is_anomaly_flagged"]:
            return "BOTH"
        if row["gbm_flagged"]:
            return "GBM"
        if row["is_anomaly_flagged"]:
            return "ANOMALY"
        return "NONE"

    cf["flag_reason"] = cf.apply(reason, axis=1)

    combined_flags = cf[["cluster_id", "gbm_flagged", "gbm_risk_score",
                          "is_anomaly_flagged", "anomaly_score", "flag_reason"]].copy()
    combined_flags.to_csv(f"{data_dir}/combined_flags.csv", index=False)

    breakdown = cf["flag_reason"].value_counts().to_dict()
    total_flagged = (cf["flag_reason"] != "NONE").sum()
    print(f"Features used: {feature_cols}")
    print(f"Contamination: {contamination}")
    print(f"Saved combined_flags.csv ({len(cf)} clusters)")
    print(f"Flag breakdown: {breakdown}")
    print(f"Total clusters flagged: {total_flagged} / {len(cf)}")

    eval_stats = {}
    if "is_real_ring" in cf.columns:
        ring_cf = cf[cf["is_real_ring"] == 1]
        both = int(((ring_cf["gbm_flagged"]) & (ring_cf["is_anomaly_flagged"])).sum())
        gbm_only = int(((ring_cf["gbm_flagged"]) & (~ring_cf["is_anomaly_flagged"])).sum())
        anomaly_only = int(((~ring_cf["gbm_flagged"]) & (ring_cf["is_anomaly_flagged"])).sum())
        missed = int(((~ring_cf["gbm_flagged"]) & (~ring_cf["is_anomaly_flagged"])).sum())
        noise_cf = cf[cf["is_real_ring"] == 0]
        fp = int(noise_cf["is_anomaly_flagged"].sum())

        eval_stats = {"both": both, "gbm_only": gbm_only, "anomaly_only": anomaly_only,
                       "missed": missed, "anomaly_false_positives": fp}

        print(f"\n--- Anomaly-layer value assessment (on {len(ring_cf)} true-ring clusters) ---")
        print(f"  Caught by BOTH        : {both}")
        print(f"  Caught by GBM only    : {gbm_only}")
        print(f"  Caught by ANOMALY only: {anomaly_only}  << this is the module's justification")
        print(f"  Missed by both        : {missed}")
        if anomaly_only == 0:
            print("  [!] Anomaly detector caught ZERO rings that GBM missed on this run.")
        else:
            print(f"  [OK] Anomaly detector added {anomaly_only} true-ring catch(es) GBM missed.")
        print(f"  Anomaly false positives: {fp} noise cluster(s) flagged")

    return iso_model, combined_flags, eval_stats


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=str, default=None,
                    help="Comma-separated feature columns (default: all)")
    p.add_argument("--contamination", type=float, default=DEFAULT_CONTAMINATION)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    feature_cols = args.features.split(",") if args.features else None
    main(feature_cols=feature_cols, contamination=args.contamination)