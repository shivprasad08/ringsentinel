"""
RingSentinel — LLM Case Narrative (Module B)
================================================

Turns a flagged case's structured evidence (feature snapshot, shared
entities, anomalous features) into a 2-3 sentence plain-English summary
an ops reviewer can read in one glance instead of parsing a JSON blob.

DESIGN DECISION — generated ON DEMAND, not in the batch pipeline:
Calling an LLM for all ~50 flagged clusters on every `run_pipeline.py`
run would be slow and cost money on every regeneration, most of which
you don't need a narrative for. Instead this is called lazily, when a
reviewer actually opens a case (wire into GET /audit/{cluster_id} in
the API, or a new GET /audit/{cluster_id}/narrative endpoint — see
bottom of this file for the exact FastAPI addition).

This module NEVER makes the flag decision — it only explains a
decision the GBM/anomaly layer already made. If the API call fails or
no key is set, callers get a clear fallback message, not a crash.

Setup:
    pip install groq python-dotenv
    Create a .env file in the repo root: GROQ_API_KEY=your_key_here

NOTE: this file was written and code-reviewed but the live API call has
NOT been tested end-to-end in the environment this was built in (no
API key available there). Test it yourself with a real key before
trusting it in the demo — same as the Docker build earlier in this
project.
"""

import os

from dotenv import load_dotenv

load_dotenv()  # reads .env in the current working directory, if present

from .case_retrieval import load_case_pool, find_similar_cases

NARRATIVE_MODEL = "openai/gpt-oss-20b"  # Groq deprecated the llama-3.x chat models; this is their current
                                          # recommended general-purpose model. Swap to "openai/gpt-oss-120b"
                                          # if you want higher quality at the cost of speed.


def build_prompt(audit_record: dict, similar_cases: list = None) -> str:
    fs = audit_record["feature_snapshot"]
    evidence = audit_record["shared_entity_evidence"]
    anomalous = audit_record["anomalous_features"]

    evidence_lines = "\n".join(
        f"- {e['signal']}: {len(e['accounts_sharing'])} accounts share the same {e['signal']} ({e['entity_id']})"
        for e in evidence
    ) or "- No shared device/instrument/address found (flagged on behavioral pattern alone)"

    anomalous_lines = "\n".join(f"- {a}" for a in anomalous) or "- None flagged as statistically unusual"

    if similar_cases:
        precedent_lines = "\n".join(
            f"- {c['cluster_id']} ({c['similarity']:.0%} structurally similar): reviewed as '{c['decision']}'"
            + (f" — reviewer note: \"{c['reviewer_note']}\"" if c["reviewer_note"] else "")
            for c in similar_cases
        )
        precedent_section = f"""
Similar past reviewed cases (use as light context only — do NOT treat 2-3 examples as statistically conclusive, and do not claim this case's outcome is "likely" based on precedent alone):
{precedent_lines}"""
    else:
        precedent_section = "\nNo similar past-reviewed cases exist yet in the system — this pattern has no reviewed precedent."

    return f"""You are writing a one-paragraph case summary for a payment-platform fraud analyst reviewing a flagged cluster of accounts. Be factual and concrete — state only what the data shows, don't speculate about motive or make legal claims (never say "this is fraud", say "this pattern is consistent with coordinated account creation" or similar). If precedent is mentioned, refer to it briefly as context, not as proof. 2-4 sentences maximum.

Cluster: {audit_record['cluster_id']}
Size: {audit_record['cluster_size']} accounts
Risk score: {audit_record['risk_score']}
Detection method: {audit_record['detection_method']}

Shared evidence:
{evidence_lines}

Feature snapshot:
{', '.join(f'{k}={v}' for k, v in fs.items())}

Anomalous vs. population baseline:
{anomalous_lines}
{precedent_section}

Write the summary now, plain text, no markdown formatting, no preamble."""


def generate_narrative(audit_record: dict, data_dir=None) -> str:
    """Returns a narrative string, or a clear fallback message if the
    API key is missing or the call fails — never raises, so a broken
    LLM call can't take down the case-viewing endpoint.

    Retrieves similar past-reviewed cases first (cheap, local, no
    network call) and grounds the narrative in them if any exist."""
    from .config import DATA_DIR as _DEFAULT_DATA_DIR
    data_dir = data_dir or _DEFAULT_DATA_DIR

    try:
        pool = load_case_pool(data_dir)
        similar_cases = find_similar_cases(audit_record["feature_snapshot"], pool,
                                            exclude_cluster_id=audit_record["cluster_id"])
    except Exception:
        similar_cases = []  # retrieval is a nice-to-have; never let it block narrative generation

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return ("[Narrative unavailable: GROQ_API_KEY not set. "
                "The structured evidence above is still complete and reliable on its own.]")

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=NARRATIVE_MODEL,
            max_tokens=600,  # gpt-oss models spend tokens on internal reasoning before
                              # answering — 200 was too tight and returned empty content
            reasoning_effort="low",  # this is a short factual summary, not a task that
                                      # needs deep chain-of-thought; keeps more of the
                                      # token budget available for the actual answer
            messages=[{"role": "user", "content": build_prompt(audit_record, similar_cases)}],
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            finish_reason = response.choices[0].finish_reason
            return (f"[Narrative generation returned empty content (finish_reason={finish_reason}). "
                    f"Try increasing max_tokens further if this persists. "
                    f"The structured evidence above is still complete and reliable on its own.]")
        return content.strip()
    except Exception as e:
        return f"[Narrative generation failed: {e}. The structured evidence above is still complete and reliable on its own.]"


if __name__ == "__main__":
    # Standalone smoke test — run against a real case from your audit log.
    # python -m pipeline.llm_narrative
    import json
    from .config import DATA_DIR

    with open(f"{DATA_DIR}/audit_log.jsonl") as f:
        first_record = json.loads(f.readline())

    print(f"Testing narrative generation for {first_record['cluster_id']}...")
    print("-" * 60)
    print(generate_narrative(first_record))