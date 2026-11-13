"""Rebuild the per-subject-area concatenated action files used as input to question
generation (01_question_generation) from the archived action files.

Run from the repository root after synopsis_splitter.py:

    python -m 00_action_data_prep.synopsis_splitter
    python -m 00_action_data_prep.build_synopsis_concat

Writes data/actions/background_key_messages/bg_km_synopsis_concat/bg_km_<Synopsis>_concat.txt.
Each file is the action files of one synopsis concatenated in filename order with no
separator, which reproduces the files used in the study byte-for-byte (checked against the
originals for all 23 synopses; the archived action files use CRLF line endings, so compare
after normalising line endings).
"""
import os

from utils.paths import ACTION_DATA_DIR

SRC = os.path.join(ACTION_DATA_DIR, "background_key_messages", "bg_km_synopsis")
DST = os.path.join(ACTION_DATA_DIR, "background_key_messages", "bg_km_synopsis_concat")


def main():
    os.makedirs(DST, exist_ok=True)
    for synopsis in sorted(os.listdir(SRC)):
        files = sorted(os.listdir(os.path.join(SRC, synopsis)))
        out = os.path.join(DST, f"bg_km_{''.join(synopsis.split())}_concat.txt")
        with open(out, "wb") as fo:
            for f in files:
                with open(os.path.join(SRC, synopsis, f), "rb") as fi:
                    fo.write(fi.read())
        print(f"{synopsis}: {len(files)} actions -> {out}")


if __name__ == "__main__":
    main()
