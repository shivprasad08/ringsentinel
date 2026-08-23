"""RingSentinel — Stage 4: Louvain Soft-Link Detection.

Run standalone:  python -m pipeline.louvain_detection
"""

import pickle

import networkx as nx
import pandas as pd

from .config import DATA_DIR, MIN_CLUSTER_SIZE, LOUVAIN_RESOLUTION


def detect(G, resolution=LOUVAIN_RESOLUTION, seed=42, min_size=MIN_CLUSTER_SIZE):
    connected_nodes = [n for n in G.nodes() if G.degree(n) > 0]
    G_active = G.subgraph(connected_nodes)
    communities = nx.algorithms.community.louvain_communities(
        G_active, weight="weight", resolution=resolution, seed=seed)

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
    return pd.DataFrame(flags)


def main(data_dir=DATA_DIR):
    with open(f"{data_dir}/account_graph.gpickle", "rb") as f:
        G = pickle.load(f)

    flags_df = detect(G)
    print(f"Flagged {flags_df['cluster_id'].nunique() if not flags_df.empty else 0} clusters, "
          f"{len(flags_df)} accounts")

    try:
        labels = pd.read_csv(f"{data_dir}/labels_HELD_OUT.csv")
        flagged = set(flags_df["account_id"]) if not flags_df.empty else set()
        ring_accts = set(labels.loc[labels["is_ring_member"] == 1, "account_id"])
        tp, fp, fn = len(flagged & ring_accts), len(flagged - ring_accts), len(ring_accts - flagged)
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        print(f"[sanity check only] precision={prec:.1%} recall={rec:.1%}")
    except FileNotFoundError:
        pass

    flags_df.to_csv(f"{data_dir}/louvain_flags.csv", index=False)
    return flags_df


if __name__ == "__main__":
    main()
