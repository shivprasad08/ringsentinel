"""Shared helpers. _id() deliberately does NOT use uuid.uuid4() — uuid4
ignores random.seed(), which breaks reproducibility across runs (this
was a real bug caught and fixed during development — see repo history /
build notes). Using the seeded `random` module keeps a given seed fully
deterministic end to end, including generated IDs."""

import hashlib
import random


def new_id(prefix: str) -> str:
    return f"{prefix}_" + "".join(random.choices("0123456789abcdef", k=10))


def hash_instrument(raw: str) -> str:
    # Never store raw card/UPI values, even synthetic ones — hash them,
    # matching real PCI hygiene.
    return "instr_" + hashlib.sha256(raw.encode()).hexdigest()[:16]
