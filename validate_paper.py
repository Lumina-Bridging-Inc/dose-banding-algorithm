#!/usr/bin/env python3
"""
Regenerate every table reported in the base algorithm manuscript.

    python validate_paper.py

Writes CSVs to output/paper/ and prints the figures quoted in the text so
that each one can be checked against the manuscript. This script is the
reproducibility artefact cited by the paper: a reviewer should be able to
clone the repository, run this command, and obtain every published number.
"""

import csv
from pathlib import Path

from dose_banding import (
    NHS_20_REF,
    VARIANCE,
    build_bands,
    verify_bands,
    write_cerner_csv,
    write_band_csv,
)

OUT = Path("output/paper")

CONC = 20.0
DTYPE = "traditional"
VAR = VARIANCE[DTYPE]
MIN_DOSE = 5.0
PAPER_MAX = 380.0     # range reported in manuscript v1.0
FULL_MAX = 1058.30    # full extent of the embedded NHS v7 reference table

REGIONS = [("< 20 mg", 0.0, 20.0),
           ("20–99 mg", 20.0, 100.0),
           ("≥ 100 mg", 100.0, float("inf"))]


def cfg(max_dose: float) -> dict:
    return {"drug_name": "Validation", "concentration_mg_per_ml": CONC,
            "drug_type": DTYPE, "min_dose_mg": MIN_DOSE, "max_dose_mg": max_dose}


def region_of(dose: float) -> str:
    for label, lo, hi in REGIONS:
        if lo <= dose < hi:
            return label
    return REGIONS[-1][0]


def true_variance(band: float, frm: float, to_a: float) -> tuple[float, float]:
    """Variance computed from published boundaries, as defined in §2.6."""
    return (band - frm) / frm * 100.0, (band - to_a) / to_a * 100.0


def compare(rows: list[dict], max_dose: float) -> list[dict]:
    """Band-by-band NHS vs algorithm comparison (Supplementary Table 1)."""
    # NHS bands whose band dose lies inside the validated span (48 for 380 mg)
    ref = [r for r in NHS_20_REF if MIN_DOSE <= r[2] <= max_dose]
    out = []
    for ref_from, ref_to, ref_dose in ref:
        match = next((r for r in rows
                      if float(r["from_mg"]) <= ref_dose < float(r["to_a_mg"])), None)
        vb, va = true_variance(ref_dose, ref_from, ref_to)
        agree = match is not None and abs(float(match["band_dose_mg"]) - ref_dose) <= 0.05
        out.append({
            "region": region_of(ref_dose),
            "nhs_band_dose_mg": ref_dose,
            "nhs_from_mg": ref_from,
            "nhs_to_a_mg": ref_to,
            "nhs_var_below_pct": round(vb, 2),
            "nhs_var_above_pct": round(va, 2),
            "nhs_exceeds_6_00": vb > 6.0 or abs(va) > 6.0,
            "nhs_exceeds_6_50": vb > 6.5 or abs(va) > 6.5,
            "algo_band_dose_mg": float(match["band_dose_mg"]) if match else None,
            "algo_from_mg": float(match["from_mg"]) if match else None,
            "algo_to_a_mg": float(match["to_a_mg"]) if match else None,
            "algo_var_below_pct": match["variance_below_pct"] if match else None,
            "algo_var_above_pct": match["variance_above_pct"] if match else None,
            "outcome": "Agreement" if agree else "Divergence",
        })
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def report(max_dose: float, tag: str) -> list[dict]:
    rows = build_bands(cfg(max_dose))          # strict=True: raises on any failure
    comp = compare(rows, max_dose)
    ref_n = len(comp)

    print(f"\n{'=' * 74}\n  {tag}: {MIN_DOSE:g}–{max_dose:g} mg @ {CONC:g} mg/mL, "
          f"±{VAR * 100:g}%\n{'=' * 74}")

    # §3.1 / §3.3 — algorithm output and tolerance compliance
    worst = max(max(abs(r["variance_below_pct"]), abs(r["variance_above_pct"]))
                for r in rows)
    assert not verify_bands(rows, VAR, CONC)
    print(f"  Algorithm bands ............ {len(rows)}")
    print(f"  Max boundary variance ...... {worst:.2f}%   (limit ±{VAR * 100:g}%)")
    print(f"  Bands exceeding limit ...... 0")
    print(f"  Coverage gaps .............. 0")
    print(f"  All volumes on-graduation .. yes")

    # §3.2 / Table 2 — agreement by region
    print(f"\n  TABLE 2 — comparison by dose region ({ref_n} NHS bands)")
    print(f"  {'Region':<10} {'NHS':>5} {'Algo':>5} {'Agree':>6} {'Rate':>7} "
          f"{'NHS>6.0':>8} {'Algo>6.0':>9}")
    tot_nhs = tot_agree = tot_viol = 0
    for label, lo, hi in REGIONS:
        c = [x for x in comp if x["region"] == label]
        a = [x for x in rows if lo <= float(x["band_dose_mg"]) < hi]
        agree = sum(1 for x in c if x["outcome"] == "Agreement")
        viol = sum(1 for x in c if x["nhs_exceeds_6_00"])
        tot_nhs += len(c); tot_agree += agree; tot_viol += viol
        rate = f"{agree / len(c) * 100:.0f}%" if c else "—"
        print(f"  {label:<10} {len(c):>5} {len(a):>5} {agree:>6} {rate:>7} "
              f"{viol:>8} {0:>9}")
    print(f"  {'Total':<10} {tot_nhs:>5} {len(rows):>5} {tot_agree:>6} "
          f"{tot_agree / tot_nhs * 100:>6.1f}% {tot_viol:>8} {0:>9}")

    # Table 3 — NHS bands exceeding 6.00%
    exc = [x for x in comp if x["nhs_exceeds_6_00"]]
    print(f"\n  TABLE 3 — NHS bands with true boundary variance > 6.00%: {len(exc)}")
    for x in exc:
        print(f"    {x['nhs_band_dose_mg']:>7g} mg  from {x['nhs_from_mg']:>8.2f}  "
              f"to {x['nhs_to_a_mg']:>8.2f}  below {x['nhs_var_below_pct']:+.2f}%  "
              f"above {x['nhs_var_above_pct']:+.2f}%  "
              f"exceeds 6.5%: {'Yes' if x['nhs_exceeds_6_50'] else 'No'}")

    # Divergences
    div = [x for x in comp if x["outcome"] == "Divergence"]
    print(f"\n  Divergences: {len(div)}")
    for label, _, _ in REGIONS:
        d = [x for x in div if x["region"] == label]
        if d:
            pairs = ", ".join(
                f"{x['nhs_band_dose_mg']:g}→"
                f"{x['algo_band_dose_mg']:g}" if x['algo_band_dose_mg'] is not None
                else f"{x['nhs_band_dose_mg']:g}→(uncovered)" for x in d)
            print(f"    {label}: {pairs}")

    OUT.mkdir(parents=True, exist_ok=True)
    write_band_csv(rows, OUT / f"supp_table2_algorithm_bands_{tag}.csv")
    write_cerner_csv(rows, OUT / f"table4_cerner_format_{tag}.csv")
    write_csv(OUT / f"supp_table1_nhs_comparison_{tag}.csv", comp)
    return rows


def table4_excerpt(rows: list[dict]) -> None:
    """The 10 representative rows printed as Table 4 in the manuscript."""
    wanted = [5.2, 5.8, 6.4, 7.2, 20.0, 96.0, 108.0, 200.0, 300.0, 320.0]
    print(f"\n{'=' * 74}\n  TABLE 4 — Cerner CPOE excerpt (regenerated at "
          f"4 dp)\n{'=' * 74}")
    print(f"  {'Range Op':<9} {'Lower':>12} {'Upper':>12} {'Unit':>5} "
          f"{'Std Dose':>12} {'Unit':>5}")
    by_dose = {float(r["band_dose_mg"]): r for r in rows}
    for d in wanted:
        r = by_dose.get(d)
        if r is None:
            print(f"  (band {d:g} mg not present)")
            continue
        print(f"  {'BETWEEN':<9} {float(r['from_mg']):>12.4f} "
              f"{float(r['to_a_mg']):>12.4f} {'mg':>5} "
              f"{float(r['band_dose_mg']):>12.4f} {'mg':>5}")


def main() -> None:
    rows_380 = report(PAPER_MAX, "5-380mg")
    table4_excerpt(rows_380)
    report(FULL_MAX, "5-1058mg")
    print(f"\n  CSVs written to {OUT.resolve()}\n")


if __name__ == "__main__":
    main()
