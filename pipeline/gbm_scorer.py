"""RingSentinel — Stage 5: GBM Ring Scorer + Eval Harness.

TRAIN/TEST DISCIPLINE: split is by CLUSTER, not account — a cluster's
accounts are entirely in train or entirely in test. Labels are used
ONLY here (train + score); no upstream stage touches labels_HELD_OUT.csv.

CAVEAT: with ~60 injected rings you get ~90-100 candidate clusters
total after Louvain. Treat precision/recall as directionally real,
not statistically tight, at this scale.

Run standalone:  python -m pipeline.gbm_scorer
"""

import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import xgboost as xgb

from .config import DATA_DIR, FEATURE_COLS, TEST_FRACTION, POSITIVE_LABEL_THRESHOLD


def build_cluster_features(flags, accounts, txns):
    accounts_idx = accounts.set_index("account_id")
    rows = []
    for cluster_id, group in flags.groupby("cluster_id"):
        member_ids = group["account_id"].tolist()
        missing = [m for m in member_ids if m not in accounts_idx.index]
        if missing:
            raise RuntimeError(
                f"{len(missing)} account IDs from flags are not in accounts.csv "
                f"(e.g. {missing[:3]}). data/ likely has files from different "
                f"pipeline runs mixed together. Delete data/ and re-run the full "
                f"pipeline (run_pipeline.py) start to finish, in one go."
            )
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


def attach_labels(cluster_features, labels):
    ring_accounts = set(labels.loc[labels["is_ring_member"] == 1, "account_id"])
    fractions, binary = [], []
    for members in cluster_features["_member_ids"]:
        frac = sum(m in ring_accounts for m in members) / len(members)
        fractions.append(frac)
        binary.append(1 if frac >= POSITIVE_LABEL_THRESHOLD else 0)
    cf = cluster_features.copy()
    cf["_true_ring_fraction"] = fractions
    cf["is_real_ring"] = binary
    return cf


def train_and_eval(cluster_df, seed=42):
    train_df, test_df = train_test_split(
        cluster_df, test_size=TEST_FRACTION, random_state=seed,
        stratify=cluster_df["is_real_ring"] if cluster_df["is_real_ring"].nunique() > 1 else None)

    model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                               random_state=seed, eval_metric="logloss")
    model.fit(train_df[FEATURE_COLS], train_df["is_real_ring"])

    test_pred = model.predict(test_df[FEATURE_COLS])
    metrics = {
        "test_clusters": len(test_df),
        "precision": precision_score(test_df["is_real_ring"], test_pred, zero_division=0),
        "recall": recall_score(test_df["is_real_ring"], test_pred, zero_division=0),
        "f1": f1_score(test_df["is_real_ring"], test_pred, zero_division=0),
    }
    test_df = test_df.copy()
    test_df["predicted"] = test_pred
    fp_clusters = test_df[(test_df["predicted"] == 1) & (test_df["is_real_ring"] == 0)]
    metrics["fp_clusters"] = len(fp_clusters)
    metrics["fp_accounts"] = sum(len(m) for m in fp_clusters["_member_ids"])

    return model, metrics, test_df


def main(data_dir=DATA_DIR):
    flags = pd.read_csv(f"{data_dir}/louvain_flags.csv")
    accounts = pd.read_csv(f"{data_dir}/accounts.csv", parse_dates=["created_at"])
    txns = pd.read_csv(f"{data_dir}/transactions.csv", parse_dates=["timestamp"])
    labels = pd.read_csv(f"{data_dir}/labels_HELD_OUT.csv")  # used ONLY here

    cluster_features = build_cluster_features(flags, accounts, txns)
    cluster_features = attach_labels(cluster_features, labels)
    print(f"Candidate clusters: {len(cluster_features)} "
          f"(real={(cluster_features['is_real_ring']==1).sum()}, "
          f"noise={(cluster_features['is_real_ring']==0).sum()})")

    model, metrics, test_df = train_and_eval(cluster_features)
    print(f"HELD-OUT TEST SET ({metrics['test_clusters']} clusters): "
          f"precision={metrics['precision']:.1%} recall={metrics['recall']:.1%} f1={metrics['f1']:.1%} "
          f"| false-positive accounts={metrics['fp_accounts']}")

    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("Feature importances:\n" + importances.to_string())

    with open(f"{data_dir}/gbm_model.pkl", "wb") as f:
        pickle.dump(model, f)
    test_df[["cluster_id", "size", "density", "chargeback_rate", "txn_velocity_per_day",
              "is_real_ring", "predicted"]].to_csv(f"{data_dir}/gbm_test_predictions.csv", index=False)

    return model, cluster_features, metrics


if __name__ == "__main__":
    main()
