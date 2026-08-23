"""
RingSentinel — FULL PIPELINE, ONE CELL
=========================================
Paste this ENTIRE file into a single Colab cell and run it. Nothing else
needed above or below it — no separate files, no imports between scripts,
no ordering to get wrong. This is the fix for the stale-file / stale-cell
problem: everything runs in one process, one execution, top to bottom.

First cell (run once):
    !pip install faker networkx xgboost scikit-learn -q

Then paste this whole file into the next cell and run it.
"""

import hashlib
import json
import os
import pickle
import random
import shutil
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from itertools import combinations

import numpy as np
import pandas as pd
import networkx as nx
from faker import Faker
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import xgboost as xgb

# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    "seed": 42,
    "n_normal_accounts": 6000,
    "n_rings": 60,
    "ring_size_range": (4, 14),
    "ring_shares_device_prob": 0.55,
    "ring_shares_instrument_prob": 0.45,
    "ring_shares_address_prob": 0.35,
    "ring_signal_coverage": 0.7,
    "ring_activity_window_days": (1, 5),
    "ring_staggered_fraction": 0.35,
    "legit_coincidence_rate": 0.04,
    "legit_coincidence_cluster_size": (2, 4),
    "noise_burst_fraction": 0.3,
    "normal_txns_per_account": (1, 8),
    "ring_txns_per_account": (2, 10),
    "normal_amount_range": (99, 5999),
    "ring_amount_range": (299, 6999),
    "chargeback_rate_normal": 0.01,
    "chargeback_rate_ring": 0.22,
    "dataset_start": "2025-11-01",
    "dataset_days": 120,
    "output_dir": "data",
}

DATA_DIR = CONFIG["output_dir"]
MIN_HARD_LINK_WEIGHT = 2
MIN_CLUSTER_SIZE = 3
LOUVAIN_RESOLUTION = 1.0
TEST_FRACTION = 0.35
POSITIVE_LABEL_THRESHOLD = 0.5
FEATURE_COLS = ["size", "density", "avg_edge_weight", "account_age_std_days",
                 "txn_velocity_per_day", "chargeback_rate", "avg_txn_amount"]


# ============================================================
# STAGE 1: SYNTHETIC DATA GENERATOR
# ============================================================
def _id(prefix):
    return f"{prefix}_" + "".join(random.choices("0123456789abcdef", k=10))


def _hash_instrument(raw):
    return "instr_" + hashlib.sha256(raw.encode()).hexdigest()[:16]


def stage1_generate(cfg):
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    fake = Faker("en_IN")
    Faker.seed(cfg["seed"])

    start_date = datetime.fromisoformat(cfg["dataset_start"])
    accounts, devices, instruments, addresses, transactions, labels = [], [], [], [], [], []

    def random_timestamp(window_start=None, window_days=None):
        if window_start is not None:
            return window_start + timedelta(days=random.uniform(0, window_days))
        return start_date + timedelta(days=random.uniform(0, cfg["dataset_days"]))

    def new_account(created_at):
        acc_id = _id("acc")
        accounts.append({"account_id": acc_id, "created_at": created_at.isoformat(),
                          "name": fake.name(), "email": fake.free_email(),
                          "phone": fake.phone_number(),
                          "kyc_status": random.choices(["verified", "pending", "unverified"],
                                                        weights=[0.75, 0.15, 0.10])[0]})
        return acc_id

    def new_device(acc_id, ts, shared=None):
        dev_id = shared or _id("dev")
        devices.append({"account_id": acc_id, "device_id": dev_id,
                         "ip_address": fake.ipv4_public(), "first_seen": ts.isoformat()})
        return dev_id

    def new_instrument(acc_id, ts, shared=None):
        instr_id = shared or _hash_instrument(fake.credit_card_number())
        instruments.append({"account_id": acc_id, "instrument_id": instr_id,
                             "instrument_type": random.choice(["card", "upi", "netbanking"]),
                             "first_seen": ts.isoformat()})
        return instr_id

    def new_address(acc_id, ts, shared=None):
        addr_id = shared or _id("addr")
        addresses.append({"account_id": acc_id, "address_id": addr_id,
                           "city": fake.city(), "pincode": fake.postcode()})
        return addr_id

    def add_transactions(acc_id, device_id, instr_id, addr_id, n_range, amount_range,
                          chargeback_rate, window_start=None, window_days=None):
        for _ in range(random.randint(*n_range)):
            ts = random_timestamp(window_start, window_days)
            status = "success"
            if random.random() < chargeback_rate:
                status = "chargeback"
            elif random.random() < 0.05:
                status = "failed"
            transactions.append({"transaction_id": _id("txn"), "account_id": acc_id,
                                  "timestamp": ts.isoformat(),
                                  "amount_inr": round(random.uniform(*amount_range), 2),
                                  "device_id": device_id, "instrument_id": instr_id,
                                  "address_id": addr_id, "status": status})

    # Normal population
    for _ in range(cfg["n_normal_accounts"]):
        ts = random_timestamp()
        acc_id = new_account(ts)
        dev_id, instr_id, addr_id = new_device(acc_id, ts), new_instrument(acc_id, ts), new_address(acc_id, ts)
        add_transactions(acc_id, dev_id, instr_id, addr_id, cfg["normal_txns_per_account"],
                          cfg["normal_amount_range"], cfg["chargeback_rate_normal"])
        labels.append({"account_id": acc_id, "ring_id": None, "is_ring_member": 0})

    # Coincidental noise clusters (false-positive stress)
    n_noise = max(1, int(cfg["n_normal_accounts"] * cfg["legit_coincidence_rate"]
                          / np.mean(cfg["legit_coincidence_cluster_size"])))
    for _ in range(n_noise):
        cluster_size = random.randint(*cfg["legit_coincidence_cluster_size"])
        ts = random_timestamp()
        shared_kind = random.choice(["device", "address"])
        shared_val = _id("dev") if shared_kind == "device" else _id("addr")
        is_bursty = random.random() < cfg["noise_burst_fraction"]
        for _ in range(cluster_size):
            acc_ts = ts + timedelta(hours=random.uniform(0, 72))
            acc_id = new_account(acc_ts)
            dev_id = new_device(acc_id, acc_ts, shared_val if shared_kind == "device" else None)
            instr_id = new_instrument(acc_id, acc_ts)
            addr_id = new_address(acc_id, acc_ts, shared_val if shared_kind == "address" else None)
            if is_bursty:
                add_transactions(acc_id, dev_id, instr_id, addr_id, cfg["ring_txns_per_account"],
                                  cfg["normal_amount_range"], cfg["chargeback_rate_normal"],
                                  window_start=ts, window_days=3)
            else:
                add_transactions(acc_id, dev_id, instr_id, addr_id, cfg["normal_txns_per_account"],
                                  cfg["normal_amount_range"], cfg["chargeback_rate_normal"])
            labels.append({"account_id": acc_id, "ring_id": None, "is_ring_member": 0})

    # Injected abuse rings
    for ring_idx in range(cfg["n_rings"]):
        ring_id = f"ring_{ring_idx:03d}"
        ring_size = random.randint(*cfg["ring_size_range"])
        is_staggered = random.random() < cfg["ring_staggered_fraction"]
        window_days = cfg["dataset_days"] if is_staggered else random.uniform(*cfg["ring_activity_window_days"])
        window_start = start_date if is_staggered else random_timestamp()

        shares_device = random.random() < cfg["ring_shares_device_prob"]
        shares_instr = random.random() < cfg["ring_shares_instrument_prob"]
        shares_addr = random.random() < cfg["ring_shares_address_prob"]
        shared_device_id = _id("dev") if shares_device else None
        shared_instr_id = _hash_instrument(_id("raw")) if shares_instr else None
        shared_addr_id = _id("addr") if shares_addr else None

        for _ in range(ring_size):
            acc_ts = window_start + timedelta(days=random.uniform(0, window_days))
            acc_id = new_account(acc_ts)
            use_dev = shares_device and random.random() < cfg["ring_signal_coverage"]
            use_instr = shares_instr and random.random() < cfg["ring_signal_coverage"]
            use_addr = shares_addr and random.random() < cfg["ring_signal_coverage"]
            dev_id = new_device(acc_id, acc_ts, shared_device_id if use_dev else None)
            instr_id = new_instrument(acc_id, acc_ts, shared_instr_id if use_instr else None)
            addr_id = new_address(acc_id, acc_ts, shared_addr_id if use_addr else None)
            add_transactions(acc_id, dev_id, instr_id, addr_id, cfg["ring_txns_per_account"],
                              cfg["ring_amount_range"], cfg["chargeback_rate_ring"],
                              window_start=window_start, window_days=window_days)
            labels.append({"account_id": acc_id, "ring_id": ring_id, "is_ring_member": 1})

    return (pd.DataFrame(accounts), pd.DataFrame(devices), pd.DataFrame(instruments),
            pd.DataFrame(addresses), pd.DataFrame(transactions), pd.DataFrame(labels))


# ============================================================
# STAGE 2: GRAPH CONSTRUCTION
# ============================================================
def stage2_build_graph(devices, instruments, addresses):
    G = nx.Graph()
    signal_tables = {"device": (devices, "device_id"), "instrument": (instruments, "instrument_id"),
                      "address": (addresses, "address_id")}

    entity_to_accounts = defaultdict(lambda: defaultdict(set))
    for signal, (df, id_col) in signal_tables.items():
        for entity_id, group in df.groupby(id_col):
            accts = set(group["account_id"])
            if len(accts) > 1:
                entity_to_accounts[signal][entity_id] = accts

    all_accounts = set()
    for df, _ in signal_tables.values():
        all_accounts |= set(df["account_id"])
    G.add_nodes_from(all_accounts)

    edge_signals = defaultdict(set)
    for signal, entity_map in entity_to_accounts.items():
        for entity_id, accts in entity_map.items():
            for a, b in combinations(sorted(accts), 2):
                edge_signals[(a, b)].add(signal)

    for (a, b), signals in edge_signals.items():
        G.add_edge(a, b, weight=len(signals), signals=sorted(signals))

    return G


# ============================================================
# STAGE 3: HARD-LINK DETECTION
# ============================================================
def stage3_hard_link(G, min_weight=MIN_HARD_LINK_WEIGHT, min_cluster_size=MIN_CLUSTER_SIZE):
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, d in G.edges(data=True):
        if d.get("weight", 0) >= min_weight:
            H.add_edge(u, v, **d)

    flags = []
    for component in nx.connected_components(H):
        if len(component) < min_cluster_size:
            continue
        sub = H.subgraph(component)
        signals_seen = set()
        for _, _, d in sub.edges(data=True):
            signals_seen.update(d.get("signals", []))
        cluster_id = f"hardlink_{min(component)}"
        for acc in component:
            flags.append({"account_id": acc, "cluster_id": cluster_id, "cluster_size": len(component),
                           "shared_signals": ",".join(sorted(signals_seen)),
                           "detection_method": "hard_link_connected_components"})
    return pd.DataFrame(flags)


# ============================================================
# STAGE 4: LOUVAIN SOFT-LINK DETECTION
# ============================================================
def stage4_louvain(G, resolution=LOUVAIN_RESOLUTION, seed=42, min_size=MIN_CLUSTER_SIZE):
    connected_nodes = [n for n in G.nodes() if G.degree(n) > 0]
    G_active = G.subgraph(connected_nodes)
    communities = nx.algorithms.community.louvain_communities(G_active, weight="weight",
                                                                resolution=resolution, seed=seed)
    flags = []
    for i, community in enumerate(communities):
        if len(community) < min_size:
            continue
        sub = G_active.subgraph(community)
        signals_seen = set()
        for _, _, d in sub.edges(data=True):
            signals_seen.update(d.get("signals", []))
        n = len(community)
        max_edges = n * (n - 1) / 2
        density = sub.number_of_edges() / max_edges if max_edges else 0
        avg_weight = (sum(d["weight"] for _, _, d in sub.edges(data=True)) / sub.number_of_edges()
                      if sub.number_of_edges() else 0)
        cluster_id = f"louvain_{i:04d}"
        for acc in community:
            flags.append({"account_id": acc, "cluster_id": cluster_id, "cluster_size": n,
                           "shared_signals": ",".join(sorted(signals_seen)),
                           "internal_density": round(density, 3), "avg_edge_weight": round(avg_weight, 2),
                           "detection_method": "louvain_soft_link"})
    return pd.DataFrame(flags), G_active


# ============================================================
# STAGE 5: GBM RING SCORER + EVAL HARNESS
# ============================================================
def stage5_build_features(flags, accounts, txns):
    accounts_idx = accounts.set_index("account_id")
    rows = []
    for cluster_id, group in flags.groupby("cluster_id"):
        member_ids = group["account_id"].tolist()
        member_accounts = accounts_idx.loc[member_ids]
        member_txns = txns[txns["account_id"].isin(member_ids)]

        age_days = (member_accounts["created_at"].max() - member_accounts["created_at"]).dt.days
        age_std = age_days.std() if len(age_days) > 1 else 0.0

        if len(member_txns) > 0:
            span_days = max((member_txns["timestamp"].max() - member_txns["timestamp"].min()).days, 1)
            txn_velocity = len(member_txns) / span_days
            chargeback_rate = (member_txns["status"] == "chargeback").mean()
            avg_amount = member_txns["amount_inr"].mean()
        else:
            txn_velocity, chargeback_rate, avg_amount = 0.0, 0.0, 0.0

        rows.append({"cluster_id": cluster_id, "size": group["cluster_size"].iloc[0],
                      "density": group["internal_density"].iloc[0], "avg_edge_weight": group["avg_edge_weight"].iloc[0],
                      "account_age_std_days": age_std, "txn_velocity_per_day": txn_velocity,
                      "chargeback_rate": chargeback_rate, "avg_txn_amount": avg_amount,
                      "_member_ids": member_ids})
    return pd.DataFrame(rows)


def stage5_attach_labels(cluster_features, labels):
    ring_accounts = set(labels.loc[labels["is_ring_member"] == 1, "account_id"])
    fractions, binary = [], []
    for members in cluster_features["_member_ids"]:
        frac = sum(m in ring_accounts for m in members) / len(members)
        fractions.append(frac)
        binary.append(1 if frac >= POSITIVE_LABEL_THRESHOLD else 0)
    cluster_features = cluster_features.copy()
    cluster_features["_true_ring_fraction"] = fractions
    cluster_features["is_real_ring"] = binary
    return cluster_features


def stage5_train_eval(cluster_df, seed=42):
    train_df, test_df = train_test_split(
        cluster_df, test_size=TEST_FRACTION, random_state=seed,
        stratify=cluster_df["is_real_ring"] if cluster_df["is_real_ring"].nunique() > 1 else None)

    model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                               random_state=seed, eval_metric="logloss")
    model.fit(train_df[FEATURE_COLS], train_df["is_real_ring"])

    test_pred = model.predict(test_df[FEATURE_COLS])
    precision = precision_score(test_df["is_real_ring"], test_pred, zero_division=0)
    recall = recall_score(test_df["is_real_ring"], test_pred, zero_division=0)
    f1 = f1_score(test_df["is_real_ring"], test_pred, zero_division=0)

    test_df = test_df.copy()
    test_df["predicted"] = test_pred
    fp_clusters = test_df[(test_df["predicted"] == 1) & (test_df["is_real_ring"] == 0)]
    fp_accounts = sum(len(m) for m in fp_clusters["_member_ids"])

    return model, {"test_clusters": len(test_df), "precision": precision, "recall": recall, "f1": f1,
                    "fp_clusters": len(fp_clusters), "fp_accounts": fp_accounts}, test_df


# ============================================================
# STAGE 6: EXPLAINABILITY + AUDIT TRAIL
# ============================================================
def stage6_build_audit(model, cluster_features, devices, instruments, addresses, threshold=0.5):
    """Score EVERY candidate cluster (not just the held-out test split —
    that split was for honest evaluation; this is the actual deployment
    decision) and, for every cluster the model flags, build a self
    -contained case file: exact shared entities (not just signal TYPE —
    the actual device/instrument/address ID and which accounts share
    it), the feature values that drove the score, and which of those
    features are anomalous vs. the candidate population baseline.

    This is what /audit/{ring_id} would serve in the API layer, and
    it's the artifact a merchant ops reviewer actually needs: not a
    number, but "here is exactly why."

    Never auto-acts. Every record's action is FLAGGED_FOR_HUMAN_REVIEW,
    full stop — that's the defense-only line from the track brief.
    """
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



def main():
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
        print(f"Cleared stale {DATA_DIR}/ from any previous run.\n")
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 60, "\nSTAGE 1: Generating synthetic dataset\n" + "=" * 60)
    accounts, devices, instruments, addresses, txns, labels = stage1_generate(CONFIG)
    accounts.to_csv(f"{DATA_DIR}/accounts.csv", index=False)
    devices.to_csv(f"{DATA_DIR}/devices.csv", index=False)
    instruments.to_csv(f"{DATA_DIR}/payment_instruments.csv", index=False)
    addresses.to_csv(f"{DATA_DIR}/addresses.csv", index=False)
    txns.to_csv(f"{DATA_DIR}/transactions.csv", index=False)
    labels.to_csv(f"{DATA_DIR}/labels_HELD_OUT.csv", index=False)
    n_ring = labels["is_ring_member"].sum()
    print(f"Accounts: {len(accounts)} | Ring accounts: {n_ring} ({n_ring/len(accounts):.1%}) "
          f"| Rings: {CONFIG['n_rings']} | Transactions: {len(txns)}")

    print("\n" + "=" * 60, "\nSTAGE 2: Building graph\n" + "=" * 60)
    account_graph = stage2_build_graph(devices, instruments, addresses)
    n_hard = sum(1 for _, _, d in account_graph.edges(data=True) if d["weight"] >= 2)
    n_soft = sum(1 for _, _, d in account_graph.edges(data=True) if d["weight"] == 1)
    print(f"Account graph: {account_graph.number_of_nodes()} accounts, {account_graph.number_of_edges()} edges "
          f"({n_hard} hard-link, {n_soft} soft-link)")

    print("\n" + "=" * 60, "\nSTAGE 3: Hard-link detection (connected components)\n" + "=" * 60)
    hard_flags = stage3_hard_link(account_graph)
    hard_flags.to_csv(f"{DATA_DIR}/hard_link_flags.csv", index=False)
    print(f"Flagged {hard_flags['cluster_id'].nunique() if not hard_flags.empty else 0} clusters, "
          f"{len(hard_flags)} accounts")
    if not hard_flags.empty:
        flagged = set(hard_flags["account_id"])
        ring_accts = set(labels.loc[labels["is_ring_member"] == 1, "account_id"])
        tp, fp, fn = len(flagged & ring_accts), len(flagged - ring_accts), len(ring_accts - flagged)
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        print(f"[sanity check] precision={prec:.1%} recall={rec:.1%}")

    print("\n" + "=" * 60, "\nSTAGE 4: Louvain soft-link detection\n" + "=" * 60)
    louvain_flags, _ = stage4_louvain(account_graph)
    louvain_flags.to_csv(f"{DATA_DIR}/louvain_flags.csv", index=False)
    print(f"Flagged {louvain_flags['cluster_id'].nunique() if not louvain_flags.empty else 0} clusters, "
          f"{len(louvain_flags)} accounts")
    if not louvain_flags.empty:
        flagged = set(louvain_flags["account_id"])
        ring_accts = set(labels.loc[labels["is_ring_member"] == 1, "account_id"])
        tp, fp, fn = len(flagged & ring_accts), len(flagged - ring_accts), len(ring_accts - flagged)
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        print(f"[sanity check] precision={prec:.1%} recall={rec:.1%}")

    print("\n" + "=" * 60, "\nSTAGE 5: GBM ring scorer + held-out eval\n" + "=" * 60)
    accounts_dt = pd.read_csv(f"{DATA_DIR}/accounts.csv", parse_dates=["created_at"])
    txns_dt = pd.read_csv(f"{DATA_DIR}/transactions.csv", parse_dates=["timestamp"])
    cluster_features = stage5_build_features(louvain_flags, accounts_dt, txns_dt)
    cluster_features = stage5_attach_labels(cluster_features, labels)
    print(f"Candidate clusters: {len(cluster_features)} "
          f"(real={  (cluster_features['is_real_ring']==1).sum()}, "
          f"noise={(cluster_features['is_real_ring']==0).sum()})")

    model, metrics, test_df = stage5_train_eval(cluster_features)
    print(f"\nHELD-OUT TEST SET ({metrics['test_clusters']} clusters):")
    print(f"  Precision: {metrics['precision']:.1%}")
    print(f"  Recall:    {metrics['recall']:.1%}")
    print(f"  F1:        {metrics['f1']:.1%}")
    print(f"  False-positive clusters: {metrics['fp_clusters']} ({metrics['fp_accounts']} accounts) <- your cost number")

    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nFeature importances:")
    print(importances.to_string())

    with open(f"{DATA_DIR}/account_graph.gpickle", "wb") as f:
        pickle.dump(account_graph, f)
    with open(f"{DATA_DIR}/gbm_model.pkl", "wb") as f:
        pickle.dump(model, f)
    test_df[["cluster_id", "size", "density", "chargeback_rate", "txn_velocity_per_day",
              "is_real_ring", "predicted"]].to_csv(f"{DATA_DIR}/gbm_test_predictions.csv", index=False)

    print("\n" + "=" * 60, "\nSTAGE 6: Explainability + audit trail\n" + "=" * 60)
    audit_records, scored_clusters = stage6_build_audit(model, cluster_features, devices, instruments, addresses)
    print(f"Clusters flagged for human review (final, deployment-scored): {len(audit_records)}")
    print(f"Total accounts across flagged clusters: {sum(r['cluster_size'] for r in audit_records)}")

    with open(f"{DATA_DIR}/audit_log.jsonl", "w") as f:
        for rec in audit_records:
            f.write(json.dumps(rec) + "\n")
    print(f"Saved -> {DATA_DIR}/audit_log.jsonl  (one JSON case-file per flagged cluster)")

    top_cases = sorted(audit_records, key=lambda r: r["risk_score"], reverse=True)[:5]
    report_lines = ["# RingSentinel — Top Flagged Cases (Audit Report)\n"]
    for rec in top_cases:
        report_lines.append(f"## {rec['cluster_id']}  —  risk score {rec['risk_score']:.2f}")
        report_lines.append(f"- **Accounts flagged ({rec['cluster_size']}):** {', '.join(rec['member_account_ids'][:6])}"
                             f"{' ...' if rec['cluster_size'] > 6 else ''}")
        report_lines.append(f"- **Action:** {rec['action']} (never auto-actioned)")
        if rec["shared_entity_evidence"]:
            ev = rec["shared_entity_evidence"][0]
            report_lines.append(f"- **Primary evidence:** {len(ev['accounts_sharing'])} accounts share the same "
                                 f"{ev['signal']} (`{ev['entity_id']}`)")
        if rec["anomalous_features"]:
            report_lines.append(f"- **Anomalous signals:** {'; '.join(rec['anomalous_features'][:3])}")
        report_lines.append("")
    with open(f"{DATA_DIR}/audit_report_top_cases.md", "w") as f:
        f.write("\n".join(report_lines))
    print(f"Saved -> {DATA_DIR}/audit_report_top_cases.md  (human-readable top-5 case writeup for your pitch)")

    baseline_stats = cluster_features[FEATURE_COLS].mean().to_dict()
    with open(f"{DATA_DIR}/baseline_stats.json", "w") as f:
        json.dump(baseline_stats, f, indent=2)
    print(f"Saved -> {DATA_DIR}/baseline_stats.json  (population feature means, for the serving API's explanations)")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE. All files in ./data/ are from this one run.")
    print("=" * 60)


main()
