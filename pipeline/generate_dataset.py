"""RingSentinel — Stage 1: Synthetic Abuse-Ring Dataset Generator.

Run standalone:  python -m pipeline.generate_dataset
Or import generate(cfg) from run_pipeline.py.
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

from .config import CONFIG
from .utils import new_id, hash_instrument


def generate(cfg=CONFIG):
    import random
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
        acc_id = new_id("acc")
        accounts.append({"account_id": acc_id, "created_at": created_at.isoformat(),
                          "name": fake.name(), "email": fake.free_email(),
                          "phone": fake.phone_number(),
                          "kyc_status": random.choices(["verified", "pending", "unverified"],
                                                        weights=[0.75, 0.15, 0.10])[0]})
        return acc_id

    def new_device(acc_id, ts, shared=None):
        dev_id = shared or new_id("dev")
        devices.append({"account_id": acc_id, "device_id": dev_id,
                         "ip_address": fake.ipv4_public(), "first_seen": ts.isoformat()})
        return dev_id

    def new_instrument(acc_id, ts, shared=None):
        instr_id = shared or hash_instrument(fake.credit_card_number())
        instruments.append({"account_id": acc_id, "instrument_id": instr_id,
                             "instrument_type": random.choice(["card", "upi", "netbanking"]),
                             "first_seen": ts.isoformat()})
        return instr_id

    def new_address(acc_id, ts, shared=None):
        addr_id = shared or new_id("addr")
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
            transactions.append({"transaction_id": new_id("txn"), "account_id": acc_id,
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
        shared_val = new_id("dev") if shared_kind == "device" else new_id("addr")
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
        shared_device_id = new_id("dev") if shares_device else None
        shared_instr_id = hash_instrument(new_id("raw")) if shares_instr else None
        shared_addr_id = new_id("addr") if shares_addr else None

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


def main(cfg=CONFIG):
    accounts, devices, instruments, addresses, txns, labels = generate(cfg)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    accounts.to_csv(f"{cfg['output_dir']}/accounts.csv", index=False)
    devices.to_csv(f"{cfg['output_dir']}/devices.csv", index=False)
    instruments.to_csv(f"{cfg['output_dir']}/payment_instruments.csv", index=False)
    addresses.to_csv(f"{cfg['output_dir']}/addresses.csv", index=False)
    txns.to_csv(f"{cfg['output_dir']}/transactions.csv", index=False)
    labels.to_csv(f"{cfg['output_dir']}/labels_HELD_OUT.csv", index=False)

    n_ring = labels["is_ring_member"].sum()
    print(f"Accounts: {len(accounts)} | Ring accounts: {n_ring} ({n_ring/len(accounts):.1%}) "
          f"| Rings: {cfg['n_rings']} | Transactions: {len(txns)}")
    return accounts, devices, instruments, addresses, txns, labels


if __name__ == "__main__":
    main()
