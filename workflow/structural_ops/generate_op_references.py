"""Generate the structural-OP reference-value table for the paper SI.

Thin runner over ``nbo2f_analysis.structural_op_references``: the closed-form
random limits, the Monte-Carlo random reference evaluated through the
production observer, and the exact P3_121 ground-state anchors for the
manuscript order parameters, written as a single provenance-headed CSV used
as supporting data.

The computation and its tests live in the package (``nbo2f_analysis``); this
script only pins the published run parameters and the output location, so the
table regenerates byte-for-byte from a tagged package version.

    python generate_op_references.py
"""
from __future__ import annotations

from pathlib import Path

from nbo2f_analysis.structural_op_references import (
    provenance_lines,
    reference_table,
    write_csv,
)

# Published SI run parameters -- pinned here rather than inherited from the
# package defaults, so the supporting-data table is reproducible from this
# repo even if the package defaults later change.
SIZES = (6, 9, 12, 15, 18)
N_SAMPLES = 500
SEED = 0
OUT = (Path(__file__).resolve().parents[2] / "build" / "structural_ops"
       / "op_reference_values.csv")


def main() -> None:
    rows = reference_table(SIZES, N_SAMPLES, seed=SEED)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as handle:
        write_csv(rows, handle, provenance=provenance_lines(SIZES, N_SAMPLES, SEED))
    print(f"wrote {OUT}  ({len(rows)} rows: {len(SIZES)} sizes x "
          f"{len(rows) // len(SIZES)} OPs)")


if __name__ == "__main__":
    main()
