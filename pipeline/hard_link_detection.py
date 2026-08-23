"""RingSentinel — Stage 3: Hard-Link Detection (connected components).

Run standalone:  python -m pipeline.hard_link_detection
"""

import pickle

import networkx as nx
import pandas as pd

from .config import DATA_DIR, MIN_HARD_LINK_WEIGHT, MIN_CLUSTER_SIZE


def detect(G, min_weight=MIN_HARD_LINK_WEIGHT, min_cluster_size=MIN_CLUSTER_SIZE):
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

    flags_df.to_csv(f"{data_dir}/hard_link_flags.csv", index=False)
    return flags_df


if __name__ == "__main__":
    main()
