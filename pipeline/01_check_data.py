"""Check that the LongEval data is present under data/ before running the rest.

The data is not redistributed with this repository. Download the LongEval 2023
Task 2 splits from https://clef-longeval-2023.github.io/data/ and place them
under data/ (see the README). This step fails early with a clear message if
anything is missing."""

from __future__ import annotations

from config import DATA_DIR, EXPECTED_DATA, UNLABELED_CORPUS


def main() -> None:
    # Check the labelled splits required by every pipeline stage
    missing = [f for f in EXPECTED_DATA if not (DATA_DIR / f).exists()]
    if missing:
        print("Missing data files under data/:")
        for f in missing:
            print(f"  - {f}")
        print("\nDownload the LongEval 2023 Task 2 data from")
        print("  https://clef-longeval-2023.github.io/data/")
        print("and place the splits under data/ (see the README).")
        raise SystemExit(1)

    print(f"All required splits present under {DATA_DIR}:")
    for f in EXPECTED_DATA:
        print(f"  - {f}")

    # The unlabelled corpus is optional and only affects semantic drift
    if UNLABELED_CORPUS.exists():
        print(f"\nUnlabelled corpus present ({UNLABELED_CORPUS.name}).")
    else:
        print(f"\nUnlabelled corpus not found ({UNLABELED_CORPUS.name}). "
              "The semantic-drift step will be skipped.")


if __name__ == "__main__":
    main()
