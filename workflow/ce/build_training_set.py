"""Build the 6x6x6 training-set JSON from collated DFT outputs.

Reads ``dft_calculations/collated_outputs.json`` (the canonical store of
DFT results across all cell sizes), filters to 6x6x6 entries, renames the
energy field to match the convention used by downstream fitting scripts,
and writes the result to a target JSON.

Methodology DFT calculations (e.g. validation runs on alternative-model
phantoms) that should not enter the training set can be excluded by
listing their paths in ``--exclude``.

Usage:
    python build_training_set.py
    python build_training_set.py --exclude ./6x6x6/config_99/
    python build_training_set.py --out training_set_refitC.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COLLATED = REPO_ROOT / "dft_calculations" / "collated_outputs.json"
DEFAULT_OUT = Path(__file__).parent / "training_set_refitB.json"


def build(
    collated_path: Path,
    out_path: Path,
    cell_label: str,
    exclude: set[str],
    dry_run: bool,
) -> list[dict]:
    """Filter collated DFT outputs to a single cell size and write a training JSON.

    Args:
        collated_path: Path to the canonical ``collated_outputs.json``.
        out_path: Path to write the training JSON.
        cell_label: Cell-size segment to filter on (e.g. ``"6x6x6"``).
        exclude: Set of canonical paths to omit from the training set.
        dry_run: If ``True``, log what would be written without writing.

    Returns:
        The list of dicts written to ``out_path`` (or that would have been
        written, in ``dry_run`` mode), each with keys ``path``,
        ``energy_per_atom``, ``relaxed geometry``.
    """
    with collated_path.open() as f:
        collated = json.load(f)

    cell_segment = f"/{cell_label}/"
    selected = []
    skipped_excluded = []
    for entry in collated:
        path = entry["path"]
        if cell_segment not in path:
            continue
        if path in exclude:
            skipped_excluded.append(path)
            continue
        selected.append(
            {
                "path": path,
                "energy_per_atom": entry["energy per atom"],
                "relaxed geometry": entry["relaxed geometry"],
            }
        )

    selected.sort(key=lambda e: e["path"])

    print(f"Collated entries: {len(collated)}")
    print(f"Filtered to {cell_label}: {sum(1 for e in collated if cell_segment in e['path'])}")
    print(f"Excluded by --exclude: {len(skipped_excluded)}")
    for p in skipped_excluded:
        print(f"  {p}")
    print(f"Training-set size: {len(selected)}")

    if not dry_run:
        with out_path.open("w") as f:
            json.dump(selected, f)
        print(f"Wrote {out_path}")
    else:
        print("(dry run; nothing written)")

    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collated", type=Path, default=COLLATED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cell", type=str, default="6x6x6")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Path to omit from the training set (e.g. './6x6x6/config_99/'). "
        "Repeat to exclude multiple structures.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    build(
        collated_path=args.collated,
        out_path=args.out,
        cell_label=args.cell,
        exclude=set(args.exclude),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
