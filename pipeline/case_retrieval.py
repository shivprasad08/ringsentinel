"""
RingSentinel — Similar-Case Retrieval (RAG over your own reviewed cases)
============================================================================

Retrieves past REVIEWED cases structurally similar to a new flagged
cluster, so the narrative can ground itself in precedent ("2 similar
past cases were confirmed fraud") instead of describing each case in
isolation. This is the system's memory — it gets more useful the more
cases get reviewed, with zero extra infrastructure: it reads only the
two artifacts the pipeline and API already produce:

  - data/audit_log.jsonl    -> every flagged case's feature_snapshot
  - data/decisions.jsonl    -> every reviewer decision ever recorded

No vector DB, no embedding API call, no new data source. Similarity is
computed over the SAME numeric features (FEATURE_COLS) the GBM/anomaly
layer already use — z-score normalized so no single feature (e.g.
avg_txn_amount, which is in the thousands) dominates cosine similarity
over a 0-1 scale feature like density.

Cold-start handling: if decisions.jsonl doesn't exist yet, or has zero
overlap with flagged cases, this returns an empty list — callers must
handle "no precedent yet" gracefully, not treat it as an error.
"""

import json
import os

import numpy as np

from .config import DATA_DIR, FEATURE_COLS


def load_case_pool(data_dir=DATA_DIR):
    """Join audit_log.jsonl (features) with decisions.jsonl (verdicts)
    on cluster_id. Only cases with BOTH a feature snapshot AND at least
    one recorded decision are retrievable — that's the whole point,
    this is precedent, not just any flagged case."""
    audit_by_id = {}
    with open(f"{data_dir}/audit_log.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            audit_by_id[rec["cluster_id"]] = rec["feature_snapshot"]

    decisions_path = f"{data_dir}/decisions.jsonl"
    if not os.path.exists(decisions_path):
        return []  # cold start — no reviews recorded yet, nothing to retrieve

    decisions_by_id = {}
    with open(decisions_path) as f:
        for line in f:
            d = json.loads(line)
            decisions_by_id.setdefault(d["cluster_id"], []).append(d)

    pool = []
    for cluster_id, decisions in decisions_by_id.items():
        if cluster_id not in audit_by_id:
            continue  # decision recorded for a case not in the current audit log
        latest = max(decisions, key=lambda d: d["decided_at"])
        pool.append({
            "cluster_id": cluster_id,
            "features": audit_by_id[cluster_id],
            "decision": latest["decision"],
            "reviewer_note": latest.get("reviewer_note", ""),
            "decided_at": latest["decided_at"],
        })
    return pool


def find_similar_cases(target_features: dict, pool: list, feature_cols=FEATURE_COLS,
                        k=3, exclude_cluster_id=None, min_similarity=0.3):
    """Cosine similarity over z-score-normalized features. Returns up
    to k results, each with a similarity score, sorted highest first.
    Filters out weak matches (below min_similarity) rather than always
    returning k regardless of quality — a poor match is worse than no
    match at all for grounding a narrative."""
    candidates = [c for c in pool if c["cluster_id"] != exclude_cluster_id]
    if not candidates:
        return []

    all_rows = [[c["features"][f] for f in feature_cols] for c in candidates]
    all_rows.append([target_features[f] for f in feature_cols])
    matrix = np.array(all_rows, dtype=float)

    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0) + 1e-9
    normed = (matrix - mean) / std

    target_vec = normed[-1]
    cand_vecs = normed[:-1]
    norms = np.linalg.norm(cand_vecs, axis=1) * np.linalg.norm(target_vec) + 1e-9
    sims = (cand_vecs @ target_vec) / norms

    order = np.argsort(-sims)
    results = []
    for i in order:
        if sims[i] < min_similarity:
            break  # sims is sorted descending, so we can stop early
        c = candidates[i]
        results.append({
            "cluster_id": c["cluster_id"],
            "decision": c["decision"],
            "reviewer_note": c["reviewer_note"],
            "similarity": float(sims[i]),
        })
        if len(results) >= k:
            break
    return results


if __name__ == "__main__":
    # Standalone smoke test
    pool = load_case_pool()
    print(f"Reviewed case pool size: {len(pool)}")
    if pool:
        target = pool[0]
        similar = find_similar_cases(target["features"], pool, exclude_cluster_id=target["cluster_id"])
        print(f"\nMost similar cases to {target['cluster_id']} (excluding itself):")
        for s in similar:
            print(f"  {s['cluster_id']}: {s['similarity']:.0%} similar, reviewed as '{s['decision']}'")
        if not similar:
            print("  (none above similarity threshold)")
    else:
        print("No reviewed cases yet — this is expected on a fresh pipeline run with no decisions recorded.")