"""Download the LongEval labelled splits and unlabelled corpus from HuggingFace."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

from config import DATA_DIR, HF_DATA_REPO


def main() -> None:
    print(f"Downloading {HF_DATA_REPO} ...")
    cache_root = Path(snapshot_download(repo_id=HF_DATA_REPO, repo_type="dataset"))
    print(f"Cache: {cache_root}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Mirror the two subdirectories the downstream scripts expect
    for sub in ("train_eval", "test"):
        src = cache_root / sub
        if not src.exists():
            print(f"  {src} not present in the dataset, skipping.")
            continue
        dst = DATA_DIR / sub
        if dst.exists() and not dst.is_symlink():
            print(f"  {dst} already exists (kept as-is).")
            continue
        if dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
        print(f"  linked {dst}  ->  {src}")

    # The unlabelled corpus (~2.8 GB extracted) is only needed by the
    # semantic-drift step. It ships zipped and extracts to the filename the
    # rest of the pipeline expects
    inner_name = "sample-10k-monthly-120k-yearly.jl"
    dst = DATA_DIR / inner_name
    if dst.exists() and dst.stat().st_size > 1_000_000_000:
        print(f"  {dst} already present ({dst.stat().st_size / 1e9:.1f} GB), skipping extract.")
        return
    zip_path = cache_root / "unlabeled_corpus.jl.zip"
    if not zip_path.exists():
        print("  Unlabelled corpus not found in the HF dataset (semantic drift step will skip).")
        return
    print(f"  Extracting {zip_path.name}  ->  {dst}  (this can take a minute) ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract(inner_name, path=DATA_DIR)
    print(f"  Extracted {dst.stat().st_size / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
