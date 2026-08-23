# RingSentinel Dashboard

Single-file, no build step, no framework — just `index.html`.

## Running it

The API must be running first (see `../api/README.md`). Then either:

**Option A — open directly:**
Double-click `index.html`. It defaults to `http://127.0.0.1:8000`. If your
API is on a different port, add it as a URL hash, e.g.:
`file:///path/to/index.html#http://127.0.0.1:8001`

**Option B — serve it (recommended, avoids occasional file:// quirks):**
```bash
cd frontend
python -m http.server 5500
```
Then open `http://localhost:5500`.

## What it shows

- **Case queue** (left): every flagged cluster from `/rings`, highest risk first.
- **Case detail** (right): click a case to load `/audit/{cluster_id}` — feature
  snapshot (anomalous features highlighted), evidence list, and a hand-drawn
  evidence graph (diamond = shared device/instrument/address, dots = accounts,
  teal = accounts actually tied to that evidence).
- **Score a candidate group** (bottom of the queue panel): paste any account
  IDs and it calls `/score-ring` live — this is real-time inference, not a
  cached lookup.

## What it deliberately does NOT do

No "Approve" / "Dismiss" buttons. The API has no decision-recording endpoint,
and building UI that implies an action it can't actually take would be
misleading. This is a read-only case viewer, matching what's actually built.

Tested with Playwright (headless Chromium) against a live local API before
being handed over — screenshots of that test are in the build notes, not
committed here.
