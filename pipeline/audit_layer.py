"""RingSentinel — Stage 6: Explainability + Audit Trail.

Scores EVERY candidate cluster (not just the held-out test split — that
was for honest evaluation; this is the deployment decision) and builds
a self-contained case file per flagged cluster: exact shared entities,
feature snapshot, and which features are anomalous vs. the population
baseline. `action` is always FLAGGED_FOR_HUMAN_REVIEW — never anything
else. This is the artifact the API's /audit/{cluster_id} serves.

Run standalone (after gbm_scorer.py has run in the same session, or via
run_pipeline.py which chains everything):  python -m pipeline.audit_layer
"""

import json
from datetime import datetime, timezone

import pandas as pd

from .config import DATA_DIR, FEATURE_COLS, RISK_THRESHOLD


def build_audit(model, cluster_features, devices, instruments, addresses, threshold=RISK_THRESHOLD):
    X_all = cluster_features[FEATURE_COLS]
    proba_all = model.predict_proba(X_all)[:, 1]
    cf = cluster_features.copy()
    cf["risk_score"] = proba_all
    cf["final_flag"] = (proba_all >= threshold).astype(int)

    baselines = cf[FEATURE_COLS].mean()

    def shared_entity_map(df, id_col):
        m = {}
        for entity_id, group in df.groupby(id_col):
            accts = set(group["account_id"])
            if len(accts) > 1:
                m[entity_id] = accts
        return m

    device_map = shared_entity_map(devices, "device_id")
    instr_map = shared_entity_map(instruments, "instrument_id")
    addr_map = shared_entity_map(addresses, "address_id")

    records = []
    for _, row in cf.iterrows():
        if row["final_flag"] != 1:
            continue
        members = set(row["_member_ids"])

        evidence = []
        for signal_name, entity_map in (("device", device_map), ("instrument", instr_map), ("address", addr_map)):
            for entity_id, accts in entity_map.items():
                overlap = accts & members
                if len(overlap) >= 2:
                    evidence.append({"signal": signal_name, "entity_id": entity_id,
                                      "accounts_sharing": sorted(overlap)})

        anomalous = []
        for feat in FEATURE_COLS:
            base = baselines[feat]
            val = row[feat]
            if base > 0:
                ratio = val / base
                if ratio >= 1.5 or ratio <= 0.67:
                    anomalous.append(f"{feat}={val:.3f} ({ratio:.1f}x candidate-population avg {base:.3f})")

        records.append({
            "cluster_id": row["cluster_id"],
            "flagged_at": datetime.now(timezone.utc).isoformat(),
            "risk_score": round(float(row["risk_score"]), 4),
            "cluster_size": int(row["size"]),
            "member_account_ids": sorted(members),
            "shared_entity_evidence": evidence,
            "feature_snapshot": {f: round(float(row[f]), 4) for f in FEATURE_COLS},
            "anomalous_features": anomalous,
            "detection_method": "louvain_soft_link -> gbm_ring_scorer",
            "action": "FLAGGED_FOR_HUMAN_REVIEW",
            "model_version": "gbm_v1_xgboost",
        })

    return records, cf


def write_top_cases_report(audit_records, path, top_n=5):
    top_cases = sorted(audit_records, key=lambda r: r["risk_score"], reverse=True)[:top_n]
    lines = ["# RingSentinel — Top Flagged Cases (Audit Report)\n"]
    for rec in top_cases:
        lines.append(f"## {rec['cluster_id']}  —  risk score {rec['risk_score']:.2f}")
        lines.append(f"- **Accounts flagged ({rec['cluster_size']}):** {', '.join(rec['member_account_ids'][:6])}"
                      f"{' ...' if rec['cluster_size'] > 6 else ''}")
        lines.append(f"- **Action:** {rec['action']} (never auto-actioned)")
        if rec["shared_entity_evidence"]:
            ev = rec["shared_entity_evidence"][0]
            lines.append(f"- **Primary evidence:** {len(ev['accounts_sharing'])} accounts share the same "
                          f"{ev['signal']} (`{ev['entity_id']}`)")
        if rec["anomalous_features"]:
            lines.append(f"- **Anomalous signals:** {'; '.join(rec['anomalous_features'][:3])}")
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main(model=None, cluster_features=None, data_dir=DATA_DIR):
    import pickle
    if model is None:
        with open(f"{data_dir}/gbm_model.pkl", "rb") as f:
            model = pickle.load(f)
    if cluster_features is None:
        # Rebuild from saved artifacts if not passed in-process
        from .gbm_scorer import build_cluster_features, attach_labels
        flags = pd.read_csv(f"{data_dir}/louvain_flags.csv")
        accounts = pd.read_csv(f"{data_dir}/accounts.csv", parse_dates=["created_at"])
        txns = pd.read_csv(f"{data_dir}/transactions.csv", parse_dates=["timestamp"])
        labels = pd.read_csv(f"{data_dir}/labels_HELD_OUT.csv")
        cluster_features = attach_labels(build_cluster_features(flags, accounts, txns), labels)

    devices = pd.read_csv(f"{data_dir}/devices.csv")
    instruments = pd.read_csv(f"{data_dir}/payment_instruments.csv")
    addresses = pd.read_csv(f"{data_dir}/addresses.csv")

    audit_records, scored_clusters = build_audit(model, cluster_features, devices, instruments, addresses)
    print(f"Clusters flagged for human review: {len(audit_records)} "
          f"({sum(r['cluster_size'] for r in audit_records)} accounts)")

    with open(f"{data_dir}/audit_log.jsonl", "w") as f:
        for rec in audit_records:
            f.write(json.dumps(rec) + "\n")

    write_top_cases_report(audit_records, f"{data_dir}/audit_report_top_cases.md")

    baseline_stats = cluster_features[FEATURE_COLS].mean().to_dict()
    with open(f"{data_dir}/baseline_stats.json", "w") as f:
        json.dump(baseline_stats, f, indent=2)

    print(f"Saved audit_log.jsonl, audit_report_top_cases.md, baseline_stats.json to {data_dir}/")
    return audit_records


if __name__ == "__main__":
    main()
