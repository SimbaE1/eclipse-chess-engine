#!/usr/bin/env python3
"""Combine all monthly WDL extracts, gzip, and upload to Kaggle eclipse-partial.

Run after all extract_*.log pipelines show EOF:
    python3 scripts/combine_and_upload_wdl.py

What it does:
  1. Cats all monthly files (2024 + 2025 + 2026) into data/wdl_combined.txt
  2. Gzips to data/wdl_training.txt.gz (replaces the old single-month gz)
  3. Uploads as a new version of simbae11/eclipse-partial

After this script, go to:
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

REPO     = Path(__file__).parent.parent
DATA     = REPO / "data"
TOKEN    = "KGAT_eda6abf5bb3456fbfdb4dd55810c2d08"
DATASET  = "simbae11/eclipse-partial"

SOURCES = [
    # 2024
    DATA / "wdl_2024_01.txt",
    DATA / "wdl_2024_02.txt",
    DATA / "wdl_2024_03.txt",
    DATA / "wdl_2024_04.txt",
    DATA / "wdl_2024_05.txt",
    DATA / "wdl_2024_06.txt",
    DATA / "wdl_2024_07.txt",
    DATA / "wdl_2024_08.txt",
    DATA / "wdl_2024_09.txt",
    DATA / "wdl_2024_10.txt",
    DATA / "wdl_2024_11.txt",
    DATA / "wdl_2024_12.txt",
    # 2025
    DATA / "wdl_training.txt",       # Jan 2025 (2.36M)
    DATA / "wdl_feb2025.txt",        # Feb 2025
    DATA / "wdl_2025_03.txt",
    DATA / "wdl_2025_04.txt",
    DATA / "wdl_2025_05.txt",
    DATA / "wdl_2025_06.txt",
    DATA / "wdl_2025_07.txt",
    DATA / "wdl_2025_08.txt",
    DATA / "wdl_2025_09.txt",
    DATA / "wdl_2025_10.txt",
    DATA / "wdl_2025_11.txt",
    DATA / "wdl_2025_12.txt",
    # 2026
    DATA / "wdl_2026_01.txt",
    DATA / "wdl_2026_02.txt",
    DATA / "wdl_2026_03.txt",
    DATA / "wdl_2026_04.txt",
    DATA / "wdl_2026_05.txt",
]

COMBINED = DATA / "wdl_combined.txt"
GZ_OUT   = DATA / "wdl_training.txt.gz"


def count_lines(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def main():
    # --- 1. Check sources ---
    total_lines = 0
    for src in SOURCES:
        if not src.exists():
            print(f"WARNING: {src.name} not found — skipping")
            continue
        n = count_lines(src)
        mb = src.stat().st_size / 1e6
        print(f"  {src.name}: {n:,} lines  ({mb:.1f} MB)")
        total_lines += n
    print(f"Total: {total_lines:,} lines\n")

    if total_lines == 0:
        print("error: no source files found"); sys.exit(1)

    # --- 2. Concatenate ---
    print(f"Concatenating -> {COMBINED.name} ...")
    with open(COMBINED, "wb") as out:
        for src in SOURCES:
            if src.exists():
                with open(src, "rb") as f:
                    shutil.copyfileobj(f, out, length=1 << 24)
    print(f"  combined: {COMBINED.stat().st_size / 1e6:.1f} MB")

    # --- 3. Gzip ---
    print(f"Compressing -> {GZ_OUT.name} ...")
    with open(COMBINED, "rb") as src, gzip.open(GZ_OUT, "wb", compresslevel=6) as dst:
        shutil.copyfileobj(src, dst, length=1 << 24)
    print(f"  compressed: {GZ_OUT.stat().st_size / 1e6:.1f} MB")

    # --- 4. Upload to Kaggle ---
    print(f"\nUploading to {DATASET} ...")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copy2(GZ_OUT, tmp / "wdl_training.txt.gz")
        meta = {"title": "eclipse-partial", "id": DATASET,
                "licenses": [{"name": "other"}]}
        (tmp / "dataset-metadata.json").write_text(json.dumps(meta))

        msg = (f"combined WDL: {total_lines:,} positions from 2024-2026, "
               f"2500+ ELO, 300s+ TC, mirror aug")
        env = {**os.environ, "KAGGLE_API_TOKEN": TOKEN}
        result = subprocess.run(
            f"kaggle datasets version -p {tmp} -m {json.dumps(msg)} --dir-mode tar",
            shell=True, env=env, capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            print(f"  Uploaded: {msg}")
            print(f"\nDataset live at: https://www.kaggle.com/datasets/{DATASET}")
        else:
            print(f"  Upload failed: {result.stderr[:400]}")
            sys.exit(1)

    print("\nDone. Now open Kaggle and run the notebook:")
    print("  https://www.kaggle.com/code/simbae11/tcec-chess-engine")
    print("  Settings -> Accelerator: GPU T4 -> Save Version -> Save & Run All")


if __name__ == "__main__":
    main()
