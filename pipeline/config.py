"""Single source of truth for pipeline configuration. Every stage module
imports from here — change a parameter once, it applies everywhere."""

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
RISK_THRESHOLD = 0.5
ANOMALY_CONTAMINATION = 0.1  # IsolationForest: expected fraction of anomalies

FEATURE_COLS = ["size", "density", "avg_edge_weight", "account_age_std_days",
                 "txn_velocity_per_day", "chargeback_rate", "avg_txn_amount"]
