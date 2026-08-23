"""RingSentinel — Stage 2: Graph Construction.

Run standalone:  python -m pipeline.build_graph
"""

import pickle
from collections import defaultdict
from itertools import combinations

import networkx as nx
import pandas as pd

from .config import DATA_DIR


def build_account_graph(devices, instruments, addresses):
    """Account-account graph: edge if two accounts share any entity,
    weighted by how many distinct signal TYPES they share."""
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


def main(data_dir=DATA_DIR):
    devices = pd.read_csv(f"{data_dir}/devices.csv")
    instruments = pd.read_csv(f"{data_dir}/payment_instruments.csv")
    addresses = pd.read_csv(f"{data_dir}/addresses.csv")
    # NOTE: labels_HELD_OUT.csv intentionally NOT loaded here.

    G = build_account_graph(devices, instruments, addresses)
    n_hard = sum(1 for _, _, d in G.edges(data=True) if d["weight"] >= 2)
    n_soft = sum(1 for _, _, d in G.edges(data=True) if d["weight"] == 1)
    print(f"Account graph: {G.number_of_nodes()} accounts, {G.number_of_edges()} edges "
          f"({n_hard} hard-link, {n_soft} soft-link)")

    with open(f"{data_dir}/account_graph.gpickle", "wb") as f:
        pickle.dump(G, f)
    return G


if __name__ == "__main__":
    main()
