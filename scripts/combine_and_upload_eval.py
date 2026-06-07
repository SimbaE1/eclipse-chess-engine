#!/usr/bin/env python3
"""Combine all monthly eval extracts, gzip, and upload to Kaggle eclipse-partial.

    python3 scripts/combine_and_upload_eval.py

What it does:
  1. Cats all non-empty eval_*.txt files into data/eval_combined.txt
  2. Gzips to data/eval_training.txt.gz
  3. Uploads as a new version of simbae11/eclipse-partial

After this, go to:
  https://www.kaggle.com/code/simbae11/tcec-chess-engine
  Settings -> Accelerator: GPU T4 -> Save Version -> Save & Run All
"""

import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO    = Path(__file__).parent.parent
DATA    = REPO / "data"
TOKEN   = "KGAT_eda6abf5bb3456fbfdb4dd55810c2d08"
DATASET = "simbae11/eclipse-partial"

COMBINED = DATA / "eval_combined.txt"
GZ_OUT   = DATA / "eval_training.txt.gz"


def main():
    sources = sorted(DATA.glob("eval_20*.txt"))
    sources = [p for p in sources if p.stat().st_size > 0]
    
    # Also include the special feb file if it exists
    feb = DATA / "eval_feb.txt"
    if feb.exists() and feb not in sources:
        sources.append(feb)

    if not sources:
        print("error: no non-empty eval_*.txt files found"); sys.exit(1)

    total_lines = 0
    for src in sources:
        # Fast line count using wc -l
        res = subprocess.run(["wc", "-l", str(src)], capture_output=True, text=True)
        n = int(res.stdout.split()[0])
        mb = src.stat().st_size / 1e6
        print(f"  {src.name}: {n:,} lines  ({mb:.1f} MB)")
        total_lines += n
    print(f"\nTotal: {total_lines:,} positions\n")

    print(f"Concatenating -> {COMBINED.name} ...")
    with open(COMBINED, "wb") as out:
        for src in sources:
            with open(src, "rb") as f:
                shutil.copyfileobj(f, out, length=1 << 24)
    print(f"  combined: {COMBINED.stat().st_size / 1e6:.1f} MB")

    print(f"Compressing -> {GZ_OUT.name} ...")
    with open(COMBINED, "rb") as src, gzip.open(GZ_OUT, "wb", compresslevel=6) as dst:
        shutil.copyfileobj(src, dst, length=1 << 24)
    gz_mb = GZ_OUT.stat().st_size / 1e6
    print(f"  compressed: {gz_mb:.1f} MB")

    print(f"\nUploading to {DATASET} ...")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copy2(GZ_OUT, tmp / "eval_training.txt.gz")
        meta = {"title": "eclipse-partial", "id": DATASET,
                "licenses": [{"name": "other"}]}
        (tmp / "dataset-metadata.json").write_text(json.dumps(meta))

        msg = (f"eval {total_lines:,} positions, 1800+Elo 180s+TC, "
               f"Lichess SF depth-22 labels, {len(sources)} months")
        env = {**os.environ, "KAGGLE_API_TOKEN": TOKEN}
        result = subprocess.run(
            f"kaggle datasets version -p {tmp} -m {json.dumps(msg)} --dir-mode tar",
            shell=True, env=env, capture_output=True, text=True, timeout=7200,
        )
        if result.returncode == 0:
            print(f"  Uploaded: {msg}")
            print(f"\nDataset live at: https://www.kaggle.com/datasets/{DATASET}")
        else:
            print(f"  Upload failed:\n{result.stderr[:800]}")
            sys.exit(1)

    print("\nDone. Now open Kaggle and run the notebook:")
    print("  https://www.kaggle.com/code/simbae11/tcec-chess-engine")
    print("  Settings -> Accelerator: GPU T4 -> Save Version -> Save & Run All")


if __name__ == "__main__":
    main()
