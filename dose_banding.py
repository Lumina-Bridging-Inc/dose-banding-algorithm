#!/usr/bin/env python3
"""
Dose Banding Table Generator
=============================
Generates dose banding CSV tables for chemotherapy drugs following the NHS
dose banding principles (originally published by NHS England).

VARIANCE LIMITS
---------------
  Traditional chemotherapy : ±6%  of the precise calculated dose
  Monoclonal antibodies     : ±10% of the precise calculated dose

ALGORITHM
---------
Band doses are dispensable volumes (multiples of the volume precision step
for the current dose range). The algorithm greedily selects the WIDEST valid
band at each step, guaranteeing:

  1. Every precisely calculated dose maps to exactly one band.
  2. The band dose assigned is ALWAYS within ±var% of the prescribed dose.
  3. There are NO coverage gaps between adjacent bands.

NOTE ON NHS ENGLAND REFERENCE TABLES
--------------------------------------
The NHS England published tables (e.g. the 20 mg/mL Version 7 table) allow
certain band boundaries to slightly exceed ±6% (up to ~6.6%) at tier
transitions — e.g. the 6.8 mg band whose lower boundary gives 6.58% variance.
This script applies a strict ±6% guarantee, so it may produce slightly more
bands in the sub-1 mL range than the NHS table, while being fully within
tolerance at every point.

USAGE
-----
  python dose_banding.py drugs_input.csv              # generate tables
  python dose_banding.py drugs_input.csv --validate   # + NHS cross-check
  python dose_banding.py --template                   # write input template
"""

import csv
import math
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# VOLUME PRECISION TIERS
# Based on analysis of NHS England 20 mg/mL Version 7 reference table.
# Tuple: (upper_volume_mL_exclusive, volume_step_mL)
# ─────────────────────────────────────────────────────────────────────────────
VOLUME_PRECISION_TIERS: list[tuple[float, float]] = [
    # Small syringes: BD minor graduations per Jordan et al,
    # Hospital Pharmacy 2019 (PMC8114303), Table 1.
    # Larger syringes (25, 50, 60 mL) use standard clinical graduations.
    # Note: 20 mL and 25 mL share the same 1.00 mL minor graduation,
    # so they are combined into one tier.
    (1.0,          0.01),   # 1 mL syringe          minor grad: 0.01 mL
    (3.0,          0.10),   # 3 mL syringe          minor grad: 0.10 mL
    (10.0,         0.20),   # 5 mL + 10 mL syringe  minor grad: 0.20 mL
    (25.0,         1.00),   # 20 mL + 25 mL syringe minor grad: 1.00 mL
    (60.0,         2.00),   # 50 mL + 60 mL syringe minor grad: 2.00 mL
    (float('inf'), 5.00),   # > 60 mL (bag)
]

# Variance limits by drug type
VARIANCE: dict[str, float] = {
    "traditional": 0.06,
    "mab":         0.10,
}

OUTPUT_DIR = Path("output")


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def get_vol_step_mL(dose_mg: float, conc: float) -> float:
    """Return volume precision step (mL) for the given dose."""
    vol = dose_mg / conc
    for upper_vol, vstep in VOLUME_PRECISION_TIERS:
        if vol < upper_vol:
            return vstep
    return VOLUME_PRECISION_TIERS[-1][1]


def get_dose_step_mg(dose_mg: float, conc: float) -> float:
    """Return dose precision step (mg) = concentration × volume step."""
    return conc * get_vol_step_mL(dose_mg, conc)


def floor_to_step(value: float, step: float) -> float:
    """Floor to nearest multiple of step (with fp-noise guard)."""
    return math.floor(round(value / step, 9)) * step


def ceil_to_step(value: float, step: float) -> float:
    """Ceiling to nearest multiple of step."""
    return math.ceil(round(value / step, 9)) * step


def geo_mean(a: float, b: float) -> float:
    return math.sqrt(a * b)


# ─────────────────────────────────────────────────────────────────────────────
# BAND DOSE SELECTION — GREEDY MAXIMUM-WIDTH ALGORITHM
# ─────────────────────────────────────────────────────────────────────────────

def _max_next_dose(D_prev: float, var: float, conc: float) -> float:
    """
    Return the largest dispensable dose D_next such that bands D_prev and
    D_next cover ALL prescribed doses without a gap:

        D_next / 1.06  ≤  D_prev / 0.94
        D_next         ≤  D_prev × (1+var) / (1-var)

    D_next must be a multiple of the dose step at its own dose level.
    """
    D_next_max = D_prev * (1.0 + var) / (1.0 - var)
    step        = get_dose_step_mg(D_next_max, conc)
    return floor_to_step(D_next_max, step)


def generate_band_doses(
    min_dose: float,
    max_dose: float,
    conc:     float,
    var:      float,
) -> list[float]:
    """
    Return ordered list of band doses covering [min_dose, max_dose].
    """
    # ── First band dose ───────────────────────────────────────────────────────
    # The lowest boundary of the first band = min_dose, so we need:
    #   D₀ ≤ min_dose × (1+var)       [band dose is at most var% above min_dose]
    # Pick the largest dispensable dose satisfying this.
    step0 = get_dose_step_mg(min_dose, conc)
    D0    = floor_to_step(min_dose * (1.0 + var), step0)

    # Safety: if D0 < min_dose push up one step so coverage starts ≤ min_dose
    if D0 < min_dose - 1e-9:
        D0 += step0
    band_doses: list[float] = [round(D0, 10)]

    # ── Subsequent band doses ─────────────────────────────────────────────────
    while True:
        D_prev  = band_doses[-1]
        D_next  = _max_next_dose(D_prev, var, conc)

        # Must strictly advance by at least one dose step
        if D_next <= D_prev + 1e-9:
            D_next = D_prev + get_dose_step_mg(D_prev, conc)

        band_doses.append(round(D_next, 10))

        # Stop when the upper coverage of D_next reaches max_dose.
        # Upper coverage limit of D_next = D_next * (1+var) / (1-var) × (1-var)
        # Simplified: the band covers up to D_next / (1-var).
        if D_next / (1.0 - var) >= max_dose:
            break

        if len(band_doses) > 100_000:
            print("WARNING: exceeded 100 000 band limit. Stopping.", file=sys.stderr)
            break

    return band_doses


# ─────────────────────────────────────────────────────────────────────────────
# BUILD FULL BAND TABLE
# ─────────────────────────────────────────────────────────────────────────────

def build_bands(drug: dict) -> list[dict]:
    """Return band-row dicts for one drug config entry."""
    name      = drug["drug_name"].strip()
    conc      = float(drug["concentration_mg_per_ml"])
    dtype     = drug["drug_type"].strip().lower()
    min_dose  = float(drug["min_dose_mg"])
    max_dose  = float(drug["max_dose_mg"])

    if dtype not in VARIANCE:
        raise ValueError(
            f"drug_type must be 'traditional' or 'mab' — got '{dtype}' for {name!r}"
        )

    var        = VARIANCE[dtype]
    band_doses = generate_band_doses(min_dose, max_dose, conc, var)
    n          = len(band_doses)
    rows: list[dict] = []

    for i, D in enumerate(band_doses):

        # ── Boundaries ───────────────────────────────────────────────────────
        # Lower boundary = D_prev / (1-var)  [first band: min_dose]
        # Upper boundary = D_curr × (1+var) / (1-var)  [last band: max_dose]
        # We floor/ceil to 2 dp to match NHS presentation convention.

        if i == 0:
            from_mg = min_dose
        else:
            # Precise lower boundary: any dose below this is covered by D_prev
            from_mg = band_doses[i - 1] / (1.0 - var)
            # Floor to 2 dp (matches NHS "From ≥" convention)
            from_mg = math.floor(round(from_mg, 8) * 100) / 100

        if i == n - 1:
            to_a_mg = max_dose
        else:
            # Precise upper boundary: doses up to but NOT including this value
            to_a_mg = D / (1.0 - var)
            # Floor to 2 dp
            to_a_mg = math.floor(round(to_a_mg, 8) * 100) / 100

        to_b_mg = round(to_a_mg - 0.01, 2)   # Round-DOWN system boundary

        # ── Volume ───────────────────────────────────────────────────────────
        volume_mL = round(D / conc, 4)
        vol_step  = get_vol_step_mL(D, conc)

        # ── Actual variance at boundaries ────────────────────────────────────
        # Below: patient at from_mg receives D  →  positive = over-dose
        # Above: patient at to_a_mg receives D  →  negative = under-dose
        var_below_pct = (D - from_mg) / from_mg * 100.0 if from_mg > 0 else 0.0
        var_above_pct = (D - to_a_mg) / to_a_mg * 100.0 if to_a_mg > 0 else 0.0

        max_pct = var * 100.0
        within  = (
            abs(var_below_pct) <= max_pct + 0.11 and   # 0.11% rounding allowance
            abs(var_above_pct) <= max_pct + 0.11
        )

        rows.append({
            "drug_name":               name,
            "concentration_mg_per_ml": conc,
            "drug_type":               dtype,
            "band_dose_mg":            round(D, 2),
            "volume_mL":               volume_mL,
            "volume_step_mL":          vol_step,
            "from_mg":                 from_mg,
            "to_a_mg":                 to_a_mg,
            "to_b_mg":                 to_b_mg,
            "variance_below_pct":      round(var_below_pct, 1),
            "variance_above_pct":      round(var_above_pct, 1),
            "within_tolerance":        within,
        })

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION: TOLERANCE & COVERAGE CHECK
# ─────────────────────────────────────────────────────────────────────────────

def validate_tolerance_and_coverage(rows: list[dict], var: float) -> dict:
    """
    Verify that:
      1. Every band is within ±var% at both boundaries.
      2. Bands are contiguous — no coverage gaps.
    Returns a summary dict.
    """
    max_var       = 0.0
    max_var_band  = None
    gaps          = []
    violations    = []
    prev_to       = None

    for row in rows:
        D    = float(row["band_dose_mg"])
        frm  = float(row["from_mg"])
        to_a = float(row["to_a_mg"])

        # Variance at boundaries
        v_low  = abs(D - frm) / frm  if frm  > 0 else 0.0
        v_high = abs(D - to_a) / to_a if to_a > 0 else 0.0
        v_max  = max(v_low, v_high)

        if v_max > max_var:
            max_var      = v_max
            max_var_band = D

        if v_max > var + 0.001:   # 0.1% tolerance for fp rounding
            violations.append((D, v_max * 100))

        # Coverage gap check
        if prev_to is not None and frm > prev_to + 0.02:
            gaps.append((prev_to, frm))
        prev_to = to_a

    return {
        "max_variance_pct": round(max_var * 100, 2),
        "max_variance_band": max_var_band,
        "violations":        violations,
        "gaps":              gaps,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NHS 20 mg/mL REFERENCE COMPARISON (informational only)
# ─────────────────────────────────────────────────────────────────────────────

NHS_20_REF: list[tuple[float, float, float]] = [
    # (from_mg, to_a_mg, band_dose_mg)
    (5.00,    5.40,    5.2),   (5.40,    5.80,    5.6),
    (5.80,    6.38,    6.0),   (6.38,    7.18,    6.8),
    (7.18,    7.98,    7.6),   (7.98,    8.80,    8.4),
    (8.80,    9.78,    9.2),   (9.78,   10.98,   10.4),
    (10.98,  12.18,   11.6),  (12.18,  13.58,   12.8),
    (13.58,  15.18,   14.4),  (15.18,  16.98,   16.0),
    (16.98,  18.98,   18.0),  (18.98,  20.98,   20.0),
    (20.98,  22.98,   22.0),  (22.98,  24.98,   24.0),
    (24.98,  26.98,   26.0),  (26.98,  28.98,   28.0),
    (28.98,  30.98,   30.0),  (30.98,  32.98,   32.0),
    (32.98,  34.98,   34.0),  (34.98,  37.94,   36.0),
    (37.94,  41.96,   40.0),  (41.96,  45.96,   44.0),
    (45.96,  49.96,   48.0),  (49.96,  53.96,   52.0),
    (53.96,  57.96,   56.0),  (57.96,  61.96,   60.0),
    (61.96,  65.96,   64.0),  (65.96,  69.98,   68.0),
    (69.98,  75.90,   72.0),  (75.90,  83.90,   80.0),
    (83.90,  91.92,   88.0),  (91.92, 101.82,   96.0),
    (101.82, 113.84,  108.0), (113.84, 125.86,  120.0),
    (125.86, 139.78,  132.0), (139.78, 155.80,  148.0),
    (155.80, 171.82,  164.0), (171.82, 189.74,  180.0),
    (189.74, 209.76,  200.0), (209.76, 229.78,  220.0),
    (229.78, 249.80,  240.0), (249.80, 269.82,  260.0),
    (269.82, 289.82,  280.0), (289.82, 309.84,  300.0),
    (309.84, 339.42,  320.0), (339.42, 379.48,  360.0),
    (379.48, 419.52,  400.0), (419.52, 459.56,  440.0),
    (459.56, 509.12,  480.0), (509.12, 569.20,  540.0),
    (569.20, 629.28,  600.0), (629.28, 689.34,  660.0),
    (689.34, 758.94,  720.0), (758.94, 848.52,  800.0),
    (848.52, 948.68,  900.0), (948.68, 1058.30, 1000.0),
]


def nhs_comparison(rows: list[dict]) -> None:
    """
    Informational cross-check against NHS England 20 mg/mL v7 table.
    Differences are EXPECTED because the NHS allows up to ~6.6% variance
    at tier transitions (our algorithm maintains strict ±6.0%).
    """
    print()
    print("  ┌─────────────────────────────────────────────────────────────────┐")
    print("  │   NHS England 20 mg/mL v7 Comparison (informational)            │")
    print("  │   Note: NHS allows up to ~6.6% at transitions; we use ±6.0%     │")
    print("  └─────────────────────────────────────────────────────────────────┘")

    gen_by_dose = {float(r["band_dose_mg"]): r for r in rows}
    ref_doses   = {d for _, _, d in NHS_20_REF}

    exact_dose_matches = sum(
        1 for _, _, d in NHS_20_REF if d in gen_by_dose
    )

    # Check how many NHS bands fall within our generated ranges
    nhs_covered = 0
    for ref_from, ref_to, ref_dose in NHS_20_REF:
        # Find which generated band covers ref_from
        mid = (ref_from + ref_to) / 2
        for row in rows:
            if float(row["from_mg"]) <= mid < float(row["to_a_mg"]):
                nhs_covered += 1
                break

    print(f"\n  NHS reference bands : {len(NHS_20_REF)}")
    print(f"  Our generated bands : {len(rows)}")
    print(f"  Exact dose matches  : {exact_dose_matches}/{len(NHS_20_REF)}")
    print(f"  NHS mid-dose covered by our bands: {nhs_covered}/{len(NHS_20_REF)}")
    print()
    print("  The NHS table allows brief tolerance exceedances at tier")
    print("  transitions to keep a lower band count; our tables maintain strict")
    print("  ≤6.0% guarantee at every prescribed dose — a conservative choice")
    print("  appropriate for a clinical Cerner CPOE system.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

BAND_FIELDS = [
    "drug_name", "concentration_mg_per_ml", "drug_type",
    "band_dose_mg", "volume_mL", "volume_step_mL",
    "from_mg", "to_a_mg", "to_b_mg",
    "variance_below_pct", "variance_above_pct", "within_tolerance",
]

# ── Cerner Standardized Dose Range output format ────────────────────────────
CERNER_FIELDS = [
    "Range Operator",
    "Range 1",
    "Range 2",
    "Dose Unit",
    "Standardized Dose",
    "Dose Unit2",       # second Dose Unit column; renamed to avoid duplicate key
]


def write_cerner_csv(rows: list[dict], filepath: Path,
                   dose_unit: str = "mg") -> None:
    """
    Write a CPOE-formatted CSV matching the Cerner Standardized Dose Range layout:
      Range Operator | Range 1 | Range 2 | Dose Unit | Standardized Dose | Dose Unit
    Values are written to 4 decimal places as the system expects.
    """
    with open(filepath, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        # Header — write two separate "Dose Unit" columns (matching screen)
        writer.writerow([
            "Range Operator", "Range 1", "Range 2",
            "Dose Unit", "Standardized Dose", "Dose Unit",
        ])
        for row in rows:
            writer.writerow([
                "BETWEEN",
                f"{float(row['from_mg']):.4f}",
                f"{float(row['to_a_mg']):.4f}",
                dose_unit,
                f"{float(row['band_dose_mg']):.4f}",
                dose_unit,
            ])


TEMPLATE_ROWS = [
    {"drug_name": "ExampleDrug_A", "concentration_mg_per_ml": 20,
     "drug_type": "traditional", "min_dose_mg": 5, "max_dose_mg": 1060},
    {"drug_name": "ExampleDrug_B", "concentration_mg_per_ml": 10,
     "drug_type": "mab",         "min_dose_mg": 100, "max_dose_mg": 2000},
]


def write_template() -> None:
    path = Path("drugs_input_template.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=[
                "drug_name", "concentration_mg_per_ml",
                "drug_type", "min_dose_mg", "max_dose_mg"
            ]
        )
        writer.writeheader()
        writer.writerows(TEMPLATE_ROWS)
    print(f"Template written → {path}")


def read_drug_csv(filepath: str) -> list[dict]:
    with open(filepath, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def write_band_csv(rows: list[dict], filepath: Path) -> None:
    with open(filepath, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=BAND_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def safe_filename(name: str, conc: float, suffix: str = "bands") -> str:
    safe = (name.replace(" ", "_").replace("/", "-")
                .replace("(", "").replace(")", ""))
    c    = f"{conc:g}".replace(".", "p")
    return f"{safe}_{c}mgmL_{suffix}.csv"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2 or "--help" in sys.argv:
        print(__doc__)
        sys.exit(0)

    if "--template" in sys.argv:
        write_template()
        sys.exit(0)

    input_path   = sys.argv[1]
    run_validate = "--validate" in sys.argv

    try:
        drugs = read_drug_csv(input_path)
    except FileNotFoundError:
        print(f"ERROR: file not found — {input_path!r}", file=sys.stderr)
        sys.exit(1)

    if not drugs:
        print("ERROR: input CSV contains no data rows.", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)
    total_bands    = 0
    total_warnings = 0

    SEP = "═" * 72
    print(f"\n{SEP}")
    print("  Dose Banding Table Generator")
    print(SEP)

    for drug in drugs:
        name  = drug.get("drug_name", "").strip()
        conc  = drug.get("concentration_mg_per_ml", "").strip()
        dtype = drug.get("drug_type", "").strip().lower()
        print(f"\n  ► {name}  ({conc} mg/mL, {dtype})")

        try:
            bands = build_bands(drug)
        except (ValueError, KeyError) as exc:
            print(f"    ERROR — skipping: {exc}", file=sys.stderr)
            continue

        var   = VARIANCE.get(dtype, 0.06)
        check = validate_tolerance_and_coverage(bands, var)

        if check["violations"]:
            n_v = len(check["violations"])
            print(f"    ⚠  {n_v} band(s) exceed the ±{var*100:.0f}% limit — review required")
            total_warnings += n_v
        else:
            print(f"    ✓  {len(bands)} bands — max variance {check['max_variance_pct']:.2f}%"
                  f" (limit ±{var*100:.0f}%)  |  no coverage gaps")

        if check["gaps"]:
            for gap_from, gap_to in check["gaps"]:
                print(f"    ⚠  Coverage gap: {gap_from:.2f} – {gap_to:.2f} mg")

        conc_f     = float(conc)
        out_detail = OUTPUT_DIR / safe_filename(name, conc_f, "bands")
        out_cpoe   = OUTPUT_DIR / safe_filename(name, conc_f, "Cerner")
        write_band_csv(bands, out_detail)
        write_cerner_csv(bands, out_cpoe)
        print(f"    → {out_detail}  (full detail)")
        print(f"    → {out_cpoe}  (Cerner import format)")
        total_bands += len(bands)

        if run_validate and abs(float(conc) - 20.0) < 1e-6 and dtype == "traditional":
            nhs_comparison(bands)

    print(f"\n{SEP}")
    print(f"  Done  │  {len(drugs)} drug(s)  │  {total_bands} total bands"
          f"  │  {total_warnings} warning(s)")
    print(f"  Output → {OUTPUT_DIR.resolve()}")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
