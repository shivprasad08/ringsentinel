"""
RingSentinel — Graph Embeddings (Module A: AI-driven detection upgrade)
==========================================================================

Adds structure-aware features learned FROM the graph itself, instead of
only hand-engineered ones (density, avg_edge_weight). This is a
lightweight alternative to a full GNN: spectral embedding (Laplacian
eigenmaps) of the account-account graph, deterministic and dependency-
light (scipy/sklearn only — no gensim/node2vec/torch needed).

Method: compute a low-dimensional embedding for every account that has
at least one graph edge (isolated accounts get no embedding — they're
never candidates anyway, since Louvain only flags connected clusters).
Two accounts close together in embedding space are structurally similar
in the graph — not just directly connected, but occupying a similar
role, which captures multi-hop patterns a raw edge-weight feature can't.

Per-cluster features derived from embeddings:
  - embedding_coherence: average pairwise cosine similarity among a
    cluster's member embeddings. A tight, coordinated ring should have
    HIGH coherence (members occupy a similar graph neighborhood); a
    coincidental cluster of otherwise-unrelated accounts should be lower.
  - embedding_centroid_deviation: how far this cluster's embedding
    centroid sits from the overall active-population centroid — an
    "unusualness in graph-structure-space" signal, distinct from the
    unsupervised anomaly layer (which uses BEHAVIORAL features, not
    structural ones).

This is an EXPERIMENT script: it reports cross-validated precision/
recall WITH vs WITHOUT these two features added to FEATURE_COLS, so you
get an honest answer to "does this actually help" rather than an
assumption. Wire it into the main pipeline only if the numbers justify it.

Run: python -m pipeline.graph_embeddings
"""

import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import networkx as nx
from scipy.linalg import eigh
from scipy.sparse import csgraph
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score
import xgboost as xgb

from .config import DATA_DIR, FEATURE_COLS
from .gbm_scorer import build_cluster_features, attach_labels

N_COMPONENTS = 8   # embedding dimensionality
SEED = 42
N_FOLDS = 5


def compute_spectral_embeddings(account_graph, n_components=N_COMPONENTS):
    """Laplacian eigenmap embedding of the graph's connected (active)
    nodes. Returns {account_id: np.array} for nodes with degree > 0
    only — isolated nodes have no structural position to embed."""
    active_nodes = [n for n in account_graph.nodes() if account_graph.degree(n) > 0]
    if len(active_nodes) < n_components + 1:
        return {}  # too few active nodes to embed meaningfully

    sub = account_graph.subgraph(active_nodes)
    node_list = list(sub.nodes())
    adj = nx.to_scipy_sparse_array(sub, nodelist=node_list, weight="weight", format="csr")

    # Normalized Laplacian. The graph is a union of many small disconnected
    # components (each ring/noise-cluster is its own component), which gives
    # the Laplacian many exact zero eigenvalues — that makes sparse
    # shift-invert solvers (ARPACK) numerically unstable (singular factor).
    # Dense eigh sidesteps this entirely and is still fast at this scale
    # (a few hundred to ~1500 active nodes).
    laplacian = csgraph.laplacian(adj, normed=True).toarray()
    eigvals, eigvecs = eigh(laplacian)
    # Skip the trivial/zero eigenvalues (one per connected component,
    # not just one) — take the first n_components AFTER those.
    n_trivial = int(np.sum(eigvals < 1e-8))
    start = max(n_trivial, 1)
    end = min(start + n_components, eigvecs.shape[1])
    embedding_matrix = eigvecs[:, start:end]

    return {node_list[i]: embedding_matrix[i] for i in range(len(node_list))}


def add_embedding_features(cluster_features, embeddings):
    """Given a cluster_features df (must have _member_ids) and an
    {account_id: vector} embedding dict, add embedding_coherence and
    embedding_centroid_deviation columns."""
    if not embeddings:
        cluster_features = cluster_features.copy()
        cluster_features["embedding_coherence"] = 0.0
        cluster_features["embedding_centroid_deviation"] = 0.0
        return cluster_features

    all_vectors = np.array(list(embeddings.values()))
    global_centroid = all_vectors.mean(axis=0)

    coherences, deviations = [], []
    for members in cluster_features["_member_ids"]:
        vecs = [embeddings[m] for m in members if m in embeddings]
        if len(vecs) < 2:
            coherences.append(0.0)
            deviations.append(0.0)
            continue
        vecs = np.array(vecs)
        norm_vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
        sim_matrix = norm_vecs @ norm_vecs.T
        n = len(vecs)
        avg_sim = (sim_matrix.sum() - n) / (n * (n - 1))  # exclude self-similarity
        coherences.append(avg_sim)

        cluster_centroid = vecs.mean(axis=0)
        deviations.append(float(np.linalg.norm(cluster_centroid - global_centroid)))

    cf = cluster_features.copy()
    cf["embedding_coherence"] = coherences
    cf["embedding_centroid_deviation"] = deviations
    return cf


def cross_validated_compare(cluster_features, baseline_cols, augmented_cols, seed=SEED):
    """Run the SAME cross-validation the depth experiment used, once on
    baseline features, once with embedding features added, so the
    comparison is apples-to-apples."""
    results = {}
    for label, cols in [("baseline", baseline_cols), ("with_embeddings", augmented_cols)]:
        X, y = cluster_features[cols], cluster_features["is_real_ring"]
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        precisions, recalls = [], []
        for train_idx, test_idx in skf.split(X, y):
            model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                       random_state=seed, eval_metric="logloss")
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict(X.iloc[test_idx])
            precisions.append(precision_score(y.iloc[test_idx], pred, zero_division=0))
            recalls.append(recall_score(y.iloc[test_idx], pred, zero_division=0))
        results[label] = {"precision_mean": np.mean(precisions), "precision_std": np.std(precisions),
                           "recall_mean": np.mean(recalls), "recall_std": np.std(recalls)}
    return results


def main(data_dir=DATA_DIR):
    with open(f"{data_dir}/account_graph.gpickle", "rb") as f:
        account_graph = pickle.load(f)

    flags = pd.read_csv(f"{data_dir}/louvain_flags.csv")
    accounts = pd.read_csv(f"{data_dir}/accounts.csv", parse_dates=["created_at"])
    txns = pd.read_csv(f"{data_dir}/transactions.csv", parse_dates=["timestamp"])
    labels = pd.read_csv(f"{data_dir}/labels_HELD_OUT.csv")  # eval-only

    cf = attach_labels(build_cluster_features(flags, accounts, txns), labels)

    print("Computing spectral embeddings on the active (non-isolated) subgraph...")
    embeddings = compute_spectral_embeddings(account_graph)
    print(f"Embedded {len(embeddings)} accounts into {N_COMPONENTS}-dimensional space.\n")

    cf = add_embedding_features(cf, embeddings)

    augmented_cols = FEATURE_COLS + ["embedding_coherence", "embedding_centroid_deviation"]
    results = cross_validated_compare(cf, FEATURE_COLS, augmented_cols)

    print("5-fold cross-validated comparison (mean ± std across folds):")
    print(f"{'':<20}{'precision':>18}{'recall':>18}")
    for label in ["baseline", "with_embeddings"]:
        r = results[label]
        print(f"{label:<20}{r['precision_mean']:>10.1%}±{r['precision_std']:.0%}"
              f"{'':>3}{r['recall_mean']:>10.1%}±{r['recall_std']:.0%}")

    p_delta = results["with_embeddings"]["precision_mean"] - results["baseline"]["precision_mean"]
    r_delta = results["with_embeddings"]["recall_mean"] - results["baseline"]["recall_mean"]
    print(f"\nDelta: precision {p_delta:+.1%}, recall {r_delta:+.1%}")
    if abs(p_delta) < 0.02 and abs(r_delta) < 0.02:
        print("Verdict: embedding features made no meaningful difference on this dataset.")
        print("Honest reason why: this dataset's classes are already near-perfectly separable")
        print("by chargeback_rate alone (see the earlier depth-vs-generalization experiment) —")
        print("there's little headroom left for ANY additional feature to improve on, structural")
        print("or otherwise. This is a property of the synthetic data's current difficulty, not")
        print("evidence that graph embeddings are a bad idea in general.")
    else:
        print("Verdict: embedding features changed the result — inspect the delta above")
        print("to decide if it's worth wiring into the main pipeline permanently.")

    cf.to_csv(f"{data_dir}/cluster_features_with_embeddings.csv", index=False)
    print(f"\nSaved -> {data_dir}/cluster_features_with_embeddings.csv")


if __name__ == "__main__":
    main()