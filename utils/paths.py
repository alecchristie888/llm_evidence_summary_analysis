"""Input-data locations for the generation stages (00, 01, 02, 02b).

The original code read Conservation Evidence action files from `action_data/` and the
cached dense embeddings from `summary_gen_data/all_actions_dense_embeddings_cache/`, relative
to the repository root. In this repository they live under `data/actions/` (unpacked from
the Zenodo archive). Run every generation script from the repository root, e.g.

    python -m 02_summary_generation.summary_gen

Set IGF_ACTION_DATA_DIR to point somewhere else if needed.
"""
import os

ACTION_DATA_DIR = os.environ.get("IGF_ACTION_DATA_DIR", os.path.join("data", "actions"))
DENSE_EMBEDDINGS_DIR = os.path.join(ACTION_DATA_DIR, "dense_embeddings_cache")
