"""RingSentinel — full pipeline orchestrator.

Run from the repo root:
    python run_pipeline.py

Chains stages 1-6 in one process, wiping data/ first — same guarantee
the Colab version had (no stale-file drift between stages), but now
with proper module separation for real development in VS Code.
"""

import os
import shutil

from pipeline.config import DATA_DIR
from pipeline import generate_dataset, build_graph, hard_link_detection, louvain_detection, gbm_scorer, audit_layer


def main():
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
        print(f"Cleared stale {DATA_DIR}/ from any previous run.\n")

    print("=" * 60, "\nSTAGE 1: Generating synthetic dataset\n" + "=" * 60)
    generate_dataset.main()

    print("\n" + "=" * 60, "\nSTAGE 2: Building graph\n" + "=" * 60)
    build_graph.main()

    print("\n" + "=" * 60, "\nSTAGE 3: Hard-link detection\n" + "=" * 60)
    hard_link_detection.main()

    print("\n" + "=" * 60, "\nSTAGE 4: Louvain soft-link detection\n" + "=" * 60)
    louvain_detection.main()

    print("\n" + "=" * 60, "\nSTAGE 5: GBM ring scorer + held-out eval\n" + "=" * 60)
    model, cluster_features, metrics = gbm_scorer.main()

    print("\n" + "=" * 60, "\nSTAGE 6: Explainability + audit trail\n" + "=" * 60)
    audit_layer.main(model=model, cluster_features=cluster_features)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE.")
    print("=" * 60)


if __name__ == "__main__":
    main()
