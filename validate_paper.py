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
from contextlib import contextmanager
from pathlib import Path

import dose_banding as db
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
FULL_MAX = 1058.30    # first 58 reference bands. No longer cited by the
                      # manuscript (Supplementary Table 3 became the full
                      # table at v1.7); kept so the figures reported in
                      # earlier drafts stay reproducible.
PUBLISHED_MAX = 6368.68   # extent of the published NHS v7 table. All 73 rows
                          # are now transcribed into NHS_20_REF, so the
                          # band-by-band comparison runs over the whole table.

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


def out_of_sample() -> None:
    """
    Out-of-sample validation against the NHS England 6 mg/mL Version 2 table
    (paclitaxel, single container), which was not consulted during development
    of the algorithm or of the volume-precision framework.
    """
    ref_path = Path("nhs_6mgml_ref.csv")
    if not ref_path.exists():
        print(f"\n  (skipping out-of-sample test — {ref_path} not found)")
        return

    ref = [(float(r["from_mg"]), float(r["to_a_mg"]), float(r["band_dose_mg"]))
           for r in csv.DictReader(open(ref_path, encoding="utf-8"))]
    conc, lo, hi = 6.0, ref[0][0], ref[-1][1]
    rows = build_bands({"drug_name": "Paclitaxel",
                        "concentration_mg_per_ml": conc, "drug_type": DTYPE,
                        "min_dose_mg": lo, "max_dose_mg": hi})

    print(f"\n{'=' * 74}\n  OUT-OF-SAMPLE: NHS England 6 mg/mL v2 (paclitaxel), "
          f"{lo:g}–{hi:g} mg\n{'=' * 74}")

    exact, deltas = 0, []
    for _, _, dose in ref:
        m = next((r for r in rows if float(r["from_mg"]) <= dose < float(r["to_a_mg"])), None)
        if m is None:
            continue
        algo = float(m["band_dose_mg"])
        deltas.append(abs(algo - dose) / dose * 100.0)
        exact += abs(algo - dose) <= 0.05

    worst = max(max(abs(r["variance_below_pct"]), abs(r["variance_above_pct"]))
                for r in rows)
    assert not verify_bands(rows, VAR, conc)
    nhs_exc = [d for f, t, d in ref if (d - f) / f * 100 > 6.0 or abs((d - t) / t * 100) > 6.0]

    print(f"  NHS bands / algorithm bands ....... {len(ref)} / {len(rows)}")
    print(f"  Exact band dose agreement ......... {exact}/{len(ref)} "
          f"= {exact / len(ref) * 100:.1f}%")
    print(f"  NHS dose within ±{VAR * 100:g}% of algo band .. "
          f"{sum(1 for x in deltas if x <= VAR * 100)}/{len(deltas)}")
    print(f"  Mean / max divergence ............. {sum(deltas) / len(deltas):.2f}% "
          f"/ {max(deltas):.2f}%")
    print(f"  Algorithm max boundary variance ... {worst:.2f}%  (0 exceedances)")
    print(f"  NHS bands over 6.00% .............. {len(nhs_exc)}  "
          f"(doses {', '.join(f'{d:g}' for d in nhs_exc)})")

    OUT.mkdir(parents=True, exist_ok=True)
    write_band_csv(rows, OUT / "out_of_sample_6mgml_algorithm_bands.csv")


def published_extent() -> None:
    """
    Algorithm-only summary across the full published extent of the NHS v7
    table (5.0–6368.68 mg): the three guaranteed properties, independent of
    any reference table. The band-by-band comparison over the same range is
    produced by report(PUBLISHED_MAX, ...) — possible since all 73 reference
    rows were transcribed on 2026-08-01.
    """
    rows = build_bands(cfg(PUBLISHED_MAX))
    worst_below = max(r["variance_below_pct"] for r in rows)
    worst_above = max(abs(r["variance_above_pct"]) for r in rows)
    gaps = [i for i in range(len(rows) - 1)
            if abs(float(rows[i]["to_a_mg"]) - float(rows[i + 1]["from_mg"])) > 1e-6]
    assert not verify_bands(rows, VAR, CONC)

    above = [r for r in rows if float(r["band_dose_mg"]) > FULL_MAX]
    cyclo = [r for r in rows if 960.0 <= float(r["band_dose_mg"]) <= 2500.0]

    print(f"\n{'=' * 74}\n  PUBLISHED EXTENT: algorithm only, "
          f"{MIN_DOSE:g}–{PUBLISHED_MAX:g} mg at {CONC:g} mg/mL\n{'=' * 74}")
    print(f"  Algorithm bands ................... {len(rows)}")
    print(f"  Max boundary variance below/above . {worst_below:.2f}% / "
          f"{worst_above:.2f}%  (0 exceedances)")
    print(f"  Coverage gaps ..................... {len(gaps)}")
    print(f"  Bands above {FULL_MAX:g} mg ............ {len(above)}  "
          f"(first {float(above[0]['band_dose_mg']):g} mg)")
    print(f"  Bands over the 960–2500 mg window . {len(cyclo)}  "
          f"({float(cyclo[0]['band_dose_mg']):g}–"
          f"{float(cyclo[-1]['band_dose_mg']):g} mg)")
    print(f"  Largest band volume ............... "
          f"{float(rows[-1]['volume_mL']):g} mL")
    print("  NHS comparison .................... see the "
          f"{MIN_DOSE:g}–{PUBLISHED_MAX:g} mg section below")

    OUT.mkdir(parents=True, exist_ok=True)
    write_band_csv(rows, OUT / "algorithm_bands_5-6368mg.csv")


# ─────────────────────────────────────────────────────────────────────────────
# §4.5 — SYRINGE FILL-POLICY SENSITIVITY
#
# The published tiers assume a dose is drawn in a single syringe of the
# smallest calibrated capacity sufficient to contain it. Where a site caps the
# fill of a hazardous-drug syringe at a fraction of calibrated capacity — BC
# Cancer applies 75%, to prevent plunger–barrel separation — the syringe
# actually used is one size larger over part of the range, so the tier
# BOUNDARIES move to the usable volumes. The graduation values do not move:
# they are physical properties of the same syringes.
#
# Nothing below changes the published output. The tier table is swapped for
# the duration of a call and restored, which is the point being made in §4.5 —
# the tiers are declared configuration, not band construction logic.
# ─────────────────────────────────────────────────────────────────────────────

FILL_FRACTION = 0.75

# (upper_volume_mL_exclusive, volume_step_mL), boundaries at 75% of the
# calibrated capacity of the syringe that carries each graduation:
#   1 mL syringe  -> 0.75 mL usable, 0.01 mL graduation
#   3 mL syringe  -> 2.25 mL usable, 0.10 mL
#   5 and 10 mL   -> 3.75 / 7.50 mL usable, 0.20 mL for both, so 7.50 binds
#   20 mL upward  -> 1.00 mL, unchanged; volumes past one syringe are aliquoted
REBASED_TIERS: list[tuple[float, float]] = [
    (0.75,         0.01),
    (2.25,         0.10),
    (7.50,         0.20),
    (float('inf'), 1.00),
]

# Sweep grid. Stated in §4.5 as eighteen concentrations, 400 minimum doses
# spanning 0.05–20 mL, and a range of forty times the minimum dose.
SWEEP_CONCS = [0.5, 1, 2, 2.5, 3, 4, 5, 6, 10,
               12.5, 20, 25, 30, 40, 50, 60, 80, 100]
SWEEP_VOLS = [x / 100 for x in range(5, 2001, 5)]
SWEEP_RANGE_MULTIPLE = 40


@contextmanager
def tiers(table: list[tuple[float, float]]):
    """Run a block against a different graduation tier table.

    Mutated in place rather than rebound: get_vol_step_mL closes over the
    module global by name, and an in-place swap cannot be defeated by anything
    else holding a reference to the same list.
    """
    saved = list(db.VOLUME_PRECISION_TIERS)
    db.VOLUME_PRECISION_TIERS[:] = table
    try:
        yield
    finally:
        db.VOLUME_PRECISION_TIERS[:] = saved


def min_feasible_tolerance(table: list[tuple[float, float]]) -> float:
    """Worst-case δ over the tier transitions, per condition (ii) of Lemma #1.

    The condition is s(M) ≤ D·2δ/(1−δ) with M = D(1+δ)/(1−δ). A transition at
    volume Vt to graduation g is therefore reached with a window of Vt·2δ/(1+δ),
    and g ≤ Vt·2δ/(1+δ) rearranges to δ ≥ g/(2·Vt − g).
    """
    return max(g / (2 * vt - g)
               for (vt, _), (_, g) in zip(table, table[1:]))


def feasibility_sweep() -> tuple[int, int, int, int]:
    """(lost, gained, ok_both, bad_both) across the sweep grid at ±6%."""
    lost = gained = ok_both = bad_both = 0
    for conc in SWEEP_CONCS:
        for vol in SWEEP_VOLS:
            lo = round(vol * conc, 4)
            drug = {"drug_name": "sweep", "concentration_mg_per_ml": conc,
                    "drug_type": DTYPE, "min_dose_mg": lo,
                    "max_dose_mg": lo * SWEEP_RANGE_MULTIPLE}
            ok = []
            for table in (db.VOLUME_PRECISION_TIERS[:], REBASED_TIERS):
                with tiers(table):
                    try:
                        build_bands(drug)
                        ok.append(True)
                    except ValueError:
                        ok.append(False)
            published, rebased = ok
            if published and not rebased:
                lost += 1
            elif rebased and not published:
                gained += 1
            elif published:
                ok_both += 1
            else:
                bad_both += 1
    return lost, gained, ok_both, bad_both


def fill_policy_sensitivity() -> None:
    """Regenerate every figure quoted in §4.5 on the fill-policy limitation.

    No CSV: none of this is a published table. It is the arithmetic behind a
    limitations paragraph, and a reviewer should be able to obtain it by
    running this script.
    """
    print(f"\n{'=' * 74}\n  §4.5 FILL-POLICY SENSITIVITY: tier boundaries "
          f"rebased on {FILL_FRACTION:.0%} of calibrated capacity\n{'=' * 74}")

    print("  Tier boundaries (mL), published -> rebased")
    for (pub_v, pub_g), (reb_v, reb_g) in zip(db.VOLUME_PRECISION_TIERS,
                                              REBASED_TIERS):
        assert pub_g == reb_g, "graduation values must not move"
        lo = "inf" if pub_v == float("inf") else f"{pub_v:g}"
        hi = "inf" if reb_v == float("inf") else f"{reb_v:g}"
        print(f"    < {lo:>4} -> < {hi:>4}   graduation {pub_g:g} mL")

    print("\n  Coarsened intervals (the volumes where the tiers disagree)")
    for lo, hi, was, now in ((0.75, 1.00, 0.01, 0.10),
                             (2.25, 3.00, 0.10, 0.20),
                             (7.50, 10.00, 0.20, 1.00)):
        print(f"    {lo:>5.2f}–{hi:<5.2f} mL   {was:g} mL -> {now:g} mL "
              f"({now / was:g}x coarser)")

    # ── effect on the three tables reported in the paper ────────────────────
    def agreement(rows, max_dose):
        comp = compare(rows, max_dose)
        n = sum(1 for r in comp if r["outcome"] == "Agreement")
        return n, len(comp)

    print("\n  Effect on the tables reported in this paper")
    print(f"  {'Comparison':<28} {'bands':>13} {'agreement':>21}")
    for max_dose, label in ((PAPER_MAX, "20 mg/mL, 5–380 mg"),
                            (PUBLISHED_MAX, "20 mg/mL, full 73-band")):
        pub = build_bands(cfg(max_dose))
        with tiers(REBASED_TIERS):
            reb = build_bands(cfg(max_dose))
        pa, pn = agreement(pub, max_dose)
        ra, _ = agreement(reb, max_dose)
        print(f"  {label:<28} {len(pub):>5} -> {len(reb):<5} "
              f"{pa}/{pn} = {pa / pn * 100:>4.1f}% -> {ra / pn * 100:>4.1f}%")
        if max_dose == PAPER_MAX:
            moved = _moved_doses(pub, reb)

    print(f"\n  Band doses that move over 5–380 mg: "
          f"{', '.join(f'{a:g} -> {b:g} mg' for a, b in moved)}")

    ref_path = Path("nhs_6mgml_ref.csv")
    if ref_path.exists():
        ref = [(float(r["from_mg"]), float(r["to_a_mg"]), float(r["band_dose_mg"]))
               for r in csv.DictReader(open(ref_path, encoding="utf-8"))]
        oos = {"drug_name": "Paclitaxel", "concentration_mg_per_ml": 6.0,
               "drug_type": DTYPE, "min_dose_mg": ref[0][0],
               "max_dose_mg": ref[-1][1]}
        pub = build_bands(oos)
        with tiers(REBASED_TIERS):
            reb = build_bands(oos)
        moved_oos = _moved_doses(pub, reb)
        print(f"  Out-of-sample 6 mg/mL:       {len(pub):>5} -> {len(reb):<5} "
              f"bands; {len(moved_oos)} band dose(s) move "
              f"({', '.join(f'{a:g} -> {b:g} mg' for a, b in moved_oos)})")

    # ── effect on feasibility ───────────────────────────────────────────────
    pub_delta = min_feasible_tolerance(db.VOLUME_PRECISION_TIERS)
    reb_delta = min_feasible_tolerance(REBASED_TIERS)
    print(f"\n  Lemma #1 condition (ii), minimum tolerance over all transitions")
    print(f"    published tiers ... {pub_delta * 100:.2f}%")
    print(f"    rebased tiers ..... {reb_delta * 100:.2f}%   "
          f"(above both the ±6.0% NHS principle and the ±6.5% NCCP ceiling)")

    lost, gained, ok_both, bad_both = feasibility_sweep()
    total = lost + gained + ok_both + bad_both
    print(f"\n  Feasibility sweep, {total} configurations at ±{VAR * 100:g}% "
          f"({len(SWEEP_CONCS)} concentrations x {len(SWEEP_VOLS)} minimum "
          f"doses)")
    print(f"    feasible on both ................. {ok_both}")
    print(f"    infeasible on both ............... {bad_both}")
    print(f"    lost (published -> rebased) ...... {lost}  "
          f"= {lost / total * 100:.1f}% of the grid")
    print(f"    gained (rebased -> published) .... {gained}")
    print("    every loss raised ValueError; no table was emitted "
          "non-compliant")


def _moved_doses(pub: list[dict], reb: list[dict]) -> list[tuple[float, float]]:
    """Published band doses absent from the rebased table, and what replaced them."""
    reb_doses = [float(r["band_dose_mg"]) for r in reb]
    out = []
    for r in pub:
        d = float(r["band_dose_mg"])
        if not any(abs(d - x) <= 1e-6 for x in reb_doses):
            out.append((d, min(reb_doses, key=lambda x: abs(x - d))))
    return out


def main() -> None:
    rows_380 = report(PAPER_MAX, "5-380mg")
    table4_excerpt(rows_380)
    report(FULL_MAX, "5-1058mg")
    published_extent()
    report(PUBLISHED_MAX, "5-6368mg")
    out_of_sample()
    fill_policy_sensitivity()
    print(f"\n  CSVs written to {OUT.resolve()}\n")


if __name__ == "__main__":
    main()
