# RingSentinel Synthetic Dataset — Specification

Use this as: (a) documentation for your repo/pitch, or (b) a prompt if you
ask an LLM to extend `generate_dataset.py`.

## Task

Generate a synthetic dataset of payment-platform accounts, with a subset
belonging to injected, labeled **abuse rings** — coordinated groups of
accounts that appear independent individually but share hidden
infrastructure. The dataset must support training/evaluating a graph-based
ring detector, with ground truth kept separate from the feature data.

## Entity schema

**accounts** — account_id, created_at, name, email, phone, kyc_status
**devices** — account_id, device_id, ip_address, first_seen (many-to-one: several accounts can point at the same device_id)
**payment_instruments** — account_id, instrument_id (hashed, never raw), instrument_type, first_seen
**addresses** — account_id, address_id, city, pincode
**transactions** — transaction_id, account_id, timestamp, amount_inr, device_id, instrument_id, address_id, status (success / failed / chargeback)
**labels_HELD_OUT** (ground truth, eval-only) — account_id, ring_id (null if not a ring member), is_ring_member

## Ring-injection logic (the difficulty knob)

For each of `n_rings` rings:
1. Pick a ring size in `ring_size_range`.
2. Pick a short activity window (`ring_activity_window_days`) — rings transact in a compressed burst, not spread evenly like normal accounts.
3. Independently decide, per ring, whether it shares a **device** (hard link), a payment **instrument** (hard link), and/or an **address** (soft link) — each with its own probability. A ring need not share all three; realistic rings often share only one or two.
4. `ring_signal_coverage` (< 1.0) controls what fraction of ring members actually carry each shared signal. This is the core subtlety mechanism: a ring where every member shares every signal is trivial to catch with exact-match clustering; a ring where only 60-70% of members carry any given shared attribute forces a detector to rely on partial/soft evidence and community structure, not just exact joins.

## False-positive stress (required — do not skip)

Inject `legit_coincidence_rate` of the normal population as small clusters
(2-4 accounts) that share exactly **one weak signal** (e.g. same IP — think
shared office/hostel wifi — or same address — think a family or PG) and
nothing else. These are legitimate, independent accounts. A detector that
flags these as a ring is producing exactly the false positives the track
brief penalizes. Report your false-positive rate specifically against this
injected noise set, not just against the general population.

## Behavioral differences (ring vs. normal)

- Ring accounts transact more per account on average (`ring_txns_per_account` > `normal_txns_per_account`) — velocity signal.
- Ring accounts have a much higher chargeback rate (`chargeback_rate_ring` ≈ 20x `chargeback_rate_normal`) — this is the actual "loss" the track cares about.
- Ring transaction amounts skew toward a narrower, higher band (`ring_amount_range`) — mimics reward/coupon-abuse or cash-out patterns.

## Difficulty presets to generate (for your eval writeup)

Run the generator three times with different `ring_signal_coverage` /
`ring_shares_*_prob` values and report precision/recall at each level —
this is what turns "we built a detector" into "we built a detector and
show exactly where it starts to break," which is the evaluation rigor
angle from your pitch:

- **Easy**: coverage 0.95, all three share-probabilities ≥ 0.7
- **Medium** (the CONFIG default): coverage 0.7, mixed probabilities
- **Hard**: coverage 0.4, only `ring_shares_address_prob` > 0, device/instrument sharing off — rings must be caught almost entirely through soft-link + behavioral/velocity signals

## Non-negotiables

- Never store raw card numbers or UPI VPAs — hash them (the script does this via SHA-256 on synthetic values, mirroring real PCI hygiene).
- Keep `labels_HELD_OUT.csv` out of any feature-engineering or model-training code path. Only the evaluation harness opens it.
