"""TaO2F compared with NbO2F on paired 3x3x3 configurations.

Each configuration in ``data/dft/TaO2F_3x3x3`` has the O/F arrangement of the
configuration of the same name in ``data/dft/3x3x3``, with Ta in place of Nb and
the same k-point mesh and numerical settings. The set holds the twelve cis + OOF
orbit representatives (orbit 11 in two settings), and eight configurations each
from the period-three random-phase and random tiers.

Writes tao2f_comparison.csv: for each configuration, its tier, its orbit (-1
outside the orbit tier), and its energy above the lowest configuration of the set
in each compound (meV/atom). Prints the median energy of each tier in each
compound (each orbit counted once, at its lower setting), the orbit energies relative to orbit 11, the Spearman rank correlation
of the orbit energies between the two compounds, and the energy difference
between the two orbit-11 settings.

    python tao2f_comparison.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from ase.io import read

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DFT = REPO / "data" / "dft"
BUILD = REPO / "build" / "dft"
DFT_OPS = BUILD / "dft_ops.csv"
OUT = BUILD / "tao2f_comparison.csv"
TIERS = ["random", "OOF-random-phase", "orbit"]
COMPOUNDS = ("tao2f", "nbo2f")


def energy_per_atom(outcar: Path) -> float:
    """Energy per atom, read as in dft_ops.py."""
    atoms = read(outcar)
    return atoms.get_potential_energy() / len(atoms)


def main() -> None:
    ops = pd.read_csv(DFT_OPS, dtype={"config": str}).set_index("config")

    rows = []
    for d in sorted((DFT / "TaO2F_3x3x3").glob("config_*")):
        config = d.name.split("_")[1]
        rows.append({
            "config": config,
            "tier": ops.loc[config, "cls"],
            "orbit": int(ops.loc[config, "orbit"]),
            "tao2f": energy_per_atom(d / "OUTCAR"),
            "nbo2f": energy_per_atom(DFT / "3x3x3" / d.name / "OUTCAR"),
        })
    df = pd.DataFrame(rows)
    for compound in COMPOUNDS:
        df[f"{compound}_meV_per_atom"] = 1000.0 * (df[compound] - df[compound].min())
    df["tier"] = pd.Categorical(df.tier, categories=TIERS, ordered=True)
    df = df.sort_values(["tier", "nbo2f_meV_per_atom"])[
        ["config", "tier", "orbit", "tao2f_meV_per_atom", "nbo2f_meV_per_atom"]]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, float_format="%.6f")

    print("median energy of each tier (meV/atom above the lowest configuration; each orbit once):")
    medians = {}
    for col in ("tao2f_meV_per_atom", "nbo2f_meV_per_atom"):
        orbit_tier = df[df.tier == "orbit"].sort_values(col).drop_duplicates("orbit")   # lower of two settings
        tiers = pd.concat([df[df.tier != "orbit"], orbit_tier])
        medians[col] = tiers.groupby("tier", observed=True)[col].median()
    print(pd.DataFrame(medians).to_string(float_format=lambda x: f"{x:.2f}"))

    orbits = df[df.tier == "orbit"].groupby("orbit")[["tao2f_meV_per_atom", "nbo2f_meV_per_atom"]].min()
    orbits = (orbits - orbits.loc[11]).sort_values("nbo2f_meV_per_atom")
    print("\norbit energies (meV/atom above orbit 11):")
    print(orbits.to_string(float_format=lambda x: f"{x:.3f}"))
    others = orbits.drop(index=11)
    print(f"\norbit 11 lowest: TaO2F {bool((others.tao2f_meV_per_atom > 0).all())}, "
          f"NbO2F {bool((others.nbo2f_meV_per_atom > 0).all())}")
    print("Spearman rank correlation of the twelve orbit energies: "
          f"{orbits.tao2f_meV_per_atom.rank().corr(orbits.nbo2f_meV_per_atom.rank()):.2f}")
    pair = df.set_index("config").loc[["063", "072"], "tao2f_meV_per_atom"]
    print(f"orbit-11 settings 063 and 072 differ by {abs(pair.iloc[0] - pair.iloc[1]):.4f} meV/atom in TaO2F")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
