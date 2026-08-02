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

# Algorithm version, surfaced in the web interface so that output can always be
# traced to the code that produced it.
#   v1.0        — base algorithm as described in the JOPP manuscript (tagged)
#   2.0.0       — adds Phase 2 vial optimisation, corrected volume tiers above
#                 25 mL, and the strict tolerance guarantee (tagged)
#   2.0.1       — no change to band generation. First release of the
#                 algorithm as an installable package, with the property-based
#                 verification suite. This is the earliest version a dependant
#                 can pin, since 2.0.0 predates the packaging metadata.
#   2.1.0       — adds opt-in vial-aware band placement
#                 (`build_bands(..., vial_aware=True)`). Default is off, so
#                 2.0.1 output is reproduced byte for byte unless the caller
#                 asks for the new behaviour.
#   2.1.1       — current: vial-aware placement now requires a band dose to sit
#                 inside its own band. 2.1.0 could seat one beneath its range,
#                 producing a 2 mg-wide band next to a near-identical one.
#                 Changes vial_aware=True output only; the default is untouched.
# Bump this in the same commit as the release tag.
__version__ = "2.1.1"
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# VOLUME PRECISION TIERS
# The tier structure was originally derived by analysis of the NHS England
# 20 mg/mL Version 7 reference table; the graduation values themselves are
# physical properties of the syringes in routine clinical use.
# Tuple: (upper_volume_mL_exclusive, volume_step_mL)
#
# Revised 2026-07-31 after out-of-sample validation against the NHS England
# 6 mg/mL Version 2 table (paclitaxel, single container). Earlier revisions
# assumed 2.00 mL graduations for 50/60 mL syringes and a 5.00 mL increment
# above 60 mL. Both were wrong: 50 and 60 mL syringes are graduated at 1 mL,
# and volumes beyond a single syringe are drawn in successive aliquots, so
# the achievable resolution stays at 1 mL rather than coarsening. The 6 mg/mL
# table confirms this directly — all 26 of its band doses at or above 10 mL
# land on whole millilitres, the largest at 128 mL. (The remaining 7 sit below
# 10 mL, on the finer 0.2 and 0.1 mL graduations.) Those two tiers had never
# been exercised by any validation: the 20 mg/mL comparison stops at 380 mg,
# i.e. 19 mL.
# ─────────────────────────────────────────────────────────────────────────────
VOLUME_PRECISION_TIERS: list[tuple[float, float]] = [
    # Small syringes: BD minor graduations per Jordan et al,
    # Hospital Pharmacy 2019 (PMC8114303), Table 1.
    # 20 mL and larger syringes use standard clinical graduations of 1 mL.
    (1.0,          0.01),   # 1 mL syringe          minor grad: 0.01 mL
    (3.0,          0.10),   # 3 mL syringe          minor grad: 0.10 mL
    (10.0,         0.20),   # 5 mL + 10 mL syringe  minor grad: 0.20 mL
    (float('inf'), 1.00),   # 20 mL and larger      minor grad: 1.00 mL
]

# Variance limits by drug type
VARIANCE: dict[str, float] = {
    "traditional": 0.06,
    "mab":         0.10,
}

# Decimal places used when presenting band boundaries.
# NOTE: 2 dp is NOT safe. At small absolute doses a 0.01 mg floor is a large
# relative perturbation, and flooring `from_mg` downward inflates the below-band
# variance computed from the *published* boundaries. Sweeping 306 concentration/
# range configurations (12 413 bands) gives a worst case of 6.13% at 2 dp
# (10 bands over 6.00%) versus exactly 6.00% and zero exceedances at 3 or 4 dp.
# 4 dp also matches the precision displayed by the Cerner dose range screen.
BOUNDARY_DP: int = 4

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


def floor_to_dp(value: float, dp: int = BOUNDARY_DP) -> float:
    """Floor to `dp` decimal places (matches the NHS 'From ≥' convention)."""
    q = 10 ** dp
    return math.floor(round(value, dp + 6) * q) / q

def _count_vials(label: str) -> int:
    """Count total vials in a combination label, e.g. '2x30mg + 1x60mg' → 3."""
    total = 0
    for part in label.split('+'):
        part = part.strip()
        if 'x' in part:
            try:
                total += int(part.split('x')[0])
            except ValueError:
                pass
    return total


# ─────────────────────────────────────────────────────────────────────────────
# VIAL OPTIMISATION
# ─────────────────────────────────────────────────────────────────────────────

def parse_vial_sizes(raw: str) -> list[float]:
    """
    Parse the vial_sizes_mg CSV field into a sorted list of floats.
    Returns [] if the field is absent, empty, or non-numeric.

    Examples:
      '100'       → [100.0]
      '100,160'   → [100.0, 160.0]
      ''          → []
    """
    if not raw or not str(raw).strip():
        return []
    sizes = []
    for part in str(raw).split(','):
        part = part.strip()
        try:
            v = float(part)
            if v > 0:
                sizes.append(v)
        except ValueError:
            pass  # ignore non-numeric entries gracefully
    return sorted(sizes)


def parse_vials_shared(raw: str) -> bool:
    """
    Parse the vials_shared CSV field.
    Accepts 'yes' / 'no' (case-insensitive). Defaults to False.
    """
    return str(raw).strip().lower() in ("yes", "true", "1")


def vial_aware_applies(
    vial_sizes:   Optional[list],
    vial_aware:   bool,
    vials_shared: bool,
) -> bool:
    """
    Whether vial-aware band placement will actually engage.

    It needs vial sizes to place bands on, and it is suppressed when vials are
    shared. Under pooling the residual volume is recovered for another patient
    on the same day, so the drug that zero-waste placement "saves" would mostly
    have been used anyway — the saving is notional, while the extra bands it
    costs are real. A site that pools is better served by the minimum-band
    greedy table.

    Exposed so that a front-end can tell the user the flag was overridden
    rather than re-deriving the rule and drifting out of step with it.
    """
    return bool(vial_sizes) and vial_aware and not vials_shared


def enumerate_vial_combinations(
    vial_sizes:   list[float],
    max_dose:     float,
    max_per_size: int = 8,
) -> list[tuple[float, str]]:
    """
    Return all achievable (dose_mg, label) pairs from combinations of vial_sizes.
    Uses recursive enumeration across ALL vial sizes jointly — not independently.

    This is essential: the combination 2x100mg + 1x160mg = 360mg is only
    discoverable by iterating (n, m) pairs together, not in separate loops.

    Args:
        vial_sizes:   sorted list of vial sizes in mg, e.g. [100.0, 160.0]
        max_dose:     upper ceiling — combinations above this are discarded
        max_per_size: maximum number of vials of any single size (default 8)

    Returns:
        Sorted list of (total_dose_mg, label_string) tuples.
        Label examples: '2x100mg', '1x160mg + 2x100mg'
    """
    results: list[tuple[float, str]] = []

    def recurse(idx: int, current_dose: float, parts: list[str]) -> None:
        if idx == len(vial_sizes):
            if current_dose > 0:  # must use at least one vial
                label = ' + '.join(parts) if parts else ''
                results.append((current_dose, label))
            return
        size = vial_sizes[idx]
        for n in range(0, max_per_size + 1):
            new_dose = current_dose + n * size
            if new_dose > max_dose + 1e-9:
                break  # pruning: further n only increases dose
            new_parts = parts.copy()
            if n > 0:
                new_parts.append(f'{n}x{size:g}mg')
            recurse(idx + 1, new_dose, new_parts)

    recurse(0, 0.0, [])

    # Sort by dose, then by label (deterministic ordering)
    results.sort(key=lambda x: (x[0], x[1]))
    return results


def best_vial_dose_in_window(
    window_low:  float,
    window_high: float,
    vial_combos: list,
) -> Optional[tuple]:
    """
    Find the best vial combination within [window_low, window_high].

    Selection priority:
      1. Zero-waste combinations — prefer the LARGEST (maximises band width).
      2. If no zero-waste option: minimum-waste combination.
      3. If no combination in window at all: return None (caller uses fallback).

    Returns:
        (dose_mg, label, waste_mg, waste_pct) or None
    """
    candidates = [
        (dose, label)
        for dose, label in vial_combos
        if window_low - 1e-9 <= dose <= window_high + 1e-9
    ]
    if not candidates:
        return None

    # All candidates are exact vial combinations → waste is 0 for all
    # Prefer the largest dose in window (maximises band width / coverage).
    # Break ties by fewest total vials (simplest preparation).
    best_dose = max(c[0] for c in candidates)
    tied = [(dose, label) for dose, label in candidates if abs(dose - best_dose) < 1e-9]
    best_dose, best_label = min(tied, key=lambda x: _count_vials(x[1]))
    waste_mg  = 0.0
    waste_pct = 0.0

    return (best_dose, best_label, waste_mg, waste_pct)


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
    # The step must be evaluated at the CANDIDATE dose, not at min_dose: the two
    # can sit in different volume tiers (e.g. min_dose 19.4 mg @ 20 mg/mL puts
    # min_dose at 0.97 mL but the candidate at 1.02 mL, a coarser tier), which
    # would otherwise yield an off-graduation first band. Mirrors _max_next_dose.
    D0_max = min_dose * (1.0 + var)
    step0  = get_dose_step_mg(D0_max, conc)
    D0     = floor_to_step(D0_max, step0)

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


def generate_band_doses_vial_aware(
    min_dose:   float,
    max_dose:   float,
    conc:       float,
    var:        float,
    vial_doses: list[float],
) -> list[float]:
    """
    As `generate_band_doses`, but PLACES bands on whole-vial totals wherever
    one is admissible, instead of maximising each band's width.

    Why this exists
    ---------------
    Phase 2 (`best_vial_dose_in_window`) substitutes a vial total into a window
    that `generate_band_doses` has already fixed. Those two steps interact
    badly. Because `to_a_mg = D/(1-var)`, the window's lower bound
    `to_a_mg*(1-var)` collapses to D itself, so the substitution can only ever
    move a band dose UP, into a sliver whose width is the leftover from
    `floor_to_step` — often under 1 mg. A vial total sitting just BELOW the
    greedy dose is unreachable no matter how attractive it is: with 10/50/200 mg
    vials the 202 mg band cannot take the single 200 mg vial, because 200 < 202.

    Placing the bands first removes that ordering problem: the width sacrificed
    by choosing a vial total over the widest admissible dose is exactly what
    buys the zero-waste preparation.

    The cost is band count. A vial total below the maximum advance narrows the
    band, so more bands are needed to cover the range. Callers who want the
    minimum-band table (P4) should keep using `generate_band_doses`.

    Dispensability
    --------------
    A vial-exact band dose is exempt from the graduation grid, on the same
    rationale `verify_bands` already applies to substituted bands: a zero-waste
    total is prepared by drawing each vial entire, so no partial volume is
    measured. Fallback doses — steps where no vial total is admissible — are
    graduation-aligned as usual.

    Args:
        vial_doses: sorted, deduplicated achievable vial-combination totals.
    """
    def largest_vial_dose_in(low: float, high: float) -> Optional[float]:
        """Largest vial total in [low, high], or None."""
        found = [d for d in vial_doses if low - 1e-9 <= d <= high + 1e-9]
        return max(found) if found else None

    # ── First band dose ───────────────────────────────────────────────────────
    # Same admissible interval as the greedy opening: [min_dose, min_dose*(1+var)].
    # It is only var% wide — half the room later steps get — so a vial total
    # lands here far less often than it does mid-table.
    D0_max = min_dose * (1.0 + var)
    D0     = largest_vial_dose_in(min_dose, D0_max)

    if D0 is None:
        step0 = get_dose_step_mg(D0_max, conc)
        D0    = floor_to_step(D0_max, step0)
        if D0 < min_dose - 1e-9:
            D0 += step0

    band_doses: list[float] = [round(D0, 10)]

    # ── Subsequent band doses ─────────────────────────────────────────────────
    while True:
        D_prev = band_doses[-1]

        # Gap-free coverage bounds the advance exactly as in the greedy case:
        # D_next/(1+var) <= D_prev/(1-var), so D_next <= cap.
        cap = D_prev * (1.0 + var) / (1.0 - var)

        # The next band's own lower boundary. A vial total below this is
        # admissible on tolerance alone — it still serves every dose in the
        # band to within var% — but it produces a band whose dose sits beneath
        # its own range, and the band is then only as wide as the gap between
        # the vial total and the boundary. With 10/50/200 mg vials that gave a
        # 2 mg-wide band at 80 mg immediately after one at 78 mg: two bands a
        # pharmacist cannot tell apart, which is the opposite of what banding
        # is for. Requiring the dose to sit inside its own band costs the odd
        # zero-waste opportunity and buys a table that reads sensibly.
        band_low = D_prev / (1.0 - var)

        D_next = largest_vial_dose_in(band_low, cap)

        if D_next is None:
            D_next = _max_next_dose(D_prev, var, conc)

        # Must strictly advance by at least one dose step
        if D_next <= D_prev + 1e-9:
            D_next = D_prev + get_dose_step_mg(D_prev, conc)

        band_doses.append(round(D_next, 10))

        if D_next / (1.0 - var) >= max_dose:
            break

        if len(band_doses) > 100_000:
            print("WARNING: exceeded 100 000 band limit. Stopping.", file=sys.stderr)
            break

    return band_doses


# ─────────────────────────────────────────────────────────────────────────────
# BUILD FULL BAND TABLE
# ─────────────────────────────────────────────────────────────────────────────

def build_bands(
    drug:         dict,
    vial_sizes:   Optional[list] = None,
    max_per_vial: int = 8,
    vials_shared: bool = False,
    cost_per_mg:  Optional[float] = None,
    strict:       bool = True,
    vial_aware:   bool = False,
) -> list[dict]:
    """
    Return band-row dicts for one drug config entry.

    With strict=True (default) the resulting table is verified against the
    three properties the algorithm claims to guarantee — dispensability,
    tolerance at the published boundaries, and gap-free coverage — and a
    ValueError is raised if any of them fails. See `verify_bands`.

    With vial_aware=True (and vial_sizes supplied) band doses are PLACED on
    whole-vial totals rather than substituted into windows fixed by the
    width-maximising greedy — far more zero-waste bands, at the cost of a
    larger table. See `generate_band_doses_vial_aware`. Default is off: the
    flag changes published output, so it is opt-in per call.

    vials_shared=True suppresses vial-aware placement — see
    `vial_aware_applies` for why, and call it to report the override.
    """
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

    # ── Vial optimisation setup ──────────────────────────────────────────────
    use_vials = bool(vial_sizes)
    vial_combos: list[tuple[float, str]] = []
    if use_vials:
        # Enumerate all combinations up to 10% above max_dose as ceiling.
        ceiling = max_dose * 1.10
        vial_combos = enumerate_vial_combinations(
            vial_sizes, ceiling, max_per_vial
        )

    placing_on_vials = vial_aware_applies(vial_sizes, vial_aware, vials_shared)

    if placing_on_vials:
        band_doses = generate_band_doses_vial_aware(
            min_dose, max_dose, conc, var,
            sorted({dose for dose, _ in vial_combos}),
        )
    else:
        band_doses = generate_band_doses(min_dose, max_dose, conc, var)
    n          = len(band_doses)
    rows: list[dict] = []

    def _waste_fields(v_waste: float, v_waste_pct: float):
        """
        Format the waste triple for one band.

        When vials are pooled the residual is recovered for another patient, so
        a non-zero figure is indicative rather than actual and is marked with a
        leading '~'. The marker covers the cost as well as the mass: quoting an
        exact currency figure for drug that will in fact be reused overstates
        the saving, which is the number a business case leans on.
        """
        cost = round(v_waste * cost_per_mg, 2) if cost_per_mg is not None else ""
        if vials_shared and v_waste > 1e-9:
            return (
                f"~{v_waste:.1f}",
                f"~{v_waste_pct:.1f}",
                f"~{cost:.2f}" if cost != "" else "",
            )
        return round(v_waste, 1), round(v_waste_pct, 1), cost

    for i, D in enumerate(band_doses):

        # ── Boundaries ───────────────────────────────────────────────────────
        # Lower boundary = D_prev / (1-var)  [first band: min_dose]
        # Upper boundary = D_curr × (1+var) / (1-var)  [last band: max_dose]
        # We floor/ceil to 2 dp to match NHS presentation convention.

        if i == 0:
            from_mg = min_dose
        else:
            # Precise lower boundary: any dose below this is covered by D_prev
            from_mg = floor_to_dp(band_doses[i - 1] / (1.0 - var))

        if i == n - 1:
            to_a_mg = max_dose
        else:
            # Precise upper boundary: doses up to but NOT including this value
            to_a_mg = floor_to_dp(D / (1.0 - var))

        # Round-DOWN system boundary: the largest value still inside this band
        # for a CPOE system whose range operator is inclusive at the upper end.
        to_b_mg = round(to_a_mg - 10 ** (-BOUNDARY_DP), BOUNDARY_DP)

        # ── Vial optimisation ────────────────────────────────────────────────
        # Attempt to substitute D with a zero-waste vial combination that
        # falls within the tolerance window already established by from_mg
        # and to_a_mg. The boundaries do NOT change — they define the window.
        vial_combo_label = ""
        waste_mg         = ""
        waste_pct_val    = ""
        waste_cost_val   = ""
        vial_optimized   = ""

        if use_vials:
            # The band dose D must be within var% of every prescribed dose in
            # [from_mg, to_a_mg]. The safe range for D is therefore:
            #   D >= to_a_mg * (1 - var)   (avoid underdose at upper boundary)
            #   D <= from_mg * (1 + var)   (avoid overdose at lower boundary)
            tol_low  = to_a_mg * (1.0 - var)
            tol_high = from_mg * (1.0 + var)

            # Substitution runs after placement, so without this it can undo
            # placement's guarantee: tol_low sits below from_mg, so a vial
            # total beneath the band's own lower boundary is admissible on
            # tolerance and would be substituted in, reseating the dose under
            # its range. Only applied when placing — raising tol_low in the
            # greedy path would change published v2.0.1 output for every vial
            # user, which is a decision and not a fix.
            if placing_on_vials:
                tol_low = max(tol_low, from_mg)

            result = best_vial_dose_in_window(tol_low, tol_high, vial_combos)
            if result is not None:
                v_dose, v_label, v_waste, v_waste_pct = result
                vial_optimized = True
                if abs(v_dose - D) > 1e-6:
                    D = v_dose
                vial_combo_label = v_label
                waste_mg, waste_pct_val, waste_cost_val = _waste_fields(
                    v_waste, v_waste_pct
                )
            else:
                # No zero-waste combination fits the tolerance window.
                # Find the nearest vial combination >= band dose (minimum waste).
                candidates_ge = [
                    (dose, label) for dose, label in vial_combos if dose >= D - 1e-9
                ]
                if candidates_ge:
                    v_dose, v_label = min(candidates_ge, key=lambda x: (x[0], _count_vials(x[1])))
                    v_waste = round(v_dose - D, 4)
                    v_waste_pct = round(v_waste / D * 100, 1) if D > 0 else 0.0
                    vial_combo_label = v_label
                    waste_mg, waste_pct_val, waste_cost_val = _waste_fields(
                        v_waste, v_waste_pct
                    )
                vial_optimized = False

        # ── Volume ───────────────────────────────────────────────────────────
        volume_mL = round(D / conc, 4)
        vol_step  = get_vol_step_mL(D, conc)

        # ── Actual variance at boundaries ────────────────────────────────────
        # Below: patient at from_mg receives D  →  positive = over-dose
        # Above: patient at to_a_mg receives D  →  negative = under-dose
        var_below_pct = (D - from_mg) / from_mg * 100.0 if from_mg > 0 else 0.0
        var_above_pct = (D - to_a_mg) / to_a_mg * 100.0 if to_a_mg > 0 else 0.0

        # No rounding allowance: with BOUNDARY_DP = 4 the published boundaries
        # satisfy the tolerance exactly, so only a floating-point epsilon is
        # needed. (The former +0.11% allowance existed to absorb 2 dp flooring.)
        max_pct = var * 100.0
        within  = (
            abs(var_below_pct) <= max_pct + 1e-9 and
            abs(var_above_pct) <= max_pct + 1e-9
        )

        rows.append({
            "drug_name":               name,
            "concentration_mg_per_ml": conc,
            "drug_type":               dtype,
            "band_dose_mg":            round(D, BOUNDARY_DP),
            "volume_mL":               volume_mL,
            "volume_step_mL":          vol_step,
            "from_mg":                 from_mg,
            "to_a_mg":                 to_a_mg,
            "to_b_mg":                 to_b_mg,
            "variance_below_pct":      round(var_below_pct, 1),
            "variance_above_pct":      round(var_above_pct, 1),
            "within_tolerance":        within,
            # vial optimisation fields — empty strings for drugs without vial sizes
            "vial_combination":        vial_combo_label,
            "waste_mg":                waste_mg,
            "waste_pct":               waste_pct_val,
            "waste_cost":              waste_cost_val,
            "vial_optimized":          vial_optimized,
            # Provenance: a downloaded table is often detached from the tool
            # that produced it, so it has to carry its own version.
            "algorithm_version":       __version__,
        })

    if strict:
        failures = verify_bands(rows, var, conc)
        if failures:
            shown = "\n  - ".join(failures[:5])
            more  = (f"\n  … and {len(failures) - 5} further failure(s)"
                     if len(failures) > 5 else "")
            raise ValueError(
                f"{name}: the requested configuration cannot be banded within "
                f"±{var * 100:.1f}% using dispensable volumes.\n"
                f"  - {shown}{more}\n"
                f"  This happens when the syringe graduation step is large "
                f"relative to the dose — i.e. the relative step exceeds "
                f"2τ/(1−τ) = {2 * var / (1 - var) * 100:.2f}%. Raise the "
                f"minimum dose, use a lower concentration, or widen the "
                f"tolerance."
            )

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# STRICT VERIFICATION OF THE THREE GUARANTEED PROPERTIES
# ─────────────────────────────────────────────────────────────────────────────

def verify_bands(rows: list[dict], var: float, conc: float) -> list[str]:
    """
    Check the properties the algorithm guarantees, from the PUBLISHED values
    (i.e. the boundaries as rounded for presentation, not the exact reals):

      1. Dispensability — every band volume is an exact multiple of the
         graduation step declared for its own volume tier.
      2. Tolerance — variance at both published boundaries is ≤ var.
      3. Coverage — no gap between adjacent bands.

    Returns a list of human-readable failure descriptions; empty means the
    table satisfies all three. This is an exact check rather than an analytic
    precondition test, so it neither rejects configurations that are in fact
    valid nor accepts ones that are not.
    """
    failures: list[str] = []
    prev_to = None

    for row in rows:
        D    = float(row["band_dose_mg"])
        frm  = float(row["from_mg"])
        to_a = float(row["to_a_mg"])
        vol  = float(row["volume_mL"])
        vstep = float(row["volume_step_mL"])

        # 1. Dispensability
        # Vial-substituted band doses are exempt: a zero-waste whole-vial
        # combination is prepared by drawing the entire contents of each vial,
        # so no partial volume is measured and the syringe graduation is not
        # the binding constraint. Bands where vial optimisation found no
        # zero-waste fit keep the graduation-aligned dose and are still checked.
        if row.get("vial_optimized") is not True:
            if abs(vol / vstep - round(vol / vstep)) > 1e-6:
                failures.append(
                    f"band {D:g} mg = {vol:g} mL is not a multiple of the "
                    f"{vstep:g} mL graduation step"
                )

        # 2. Tolerance at the published boundaries
        v_low  = abs(D - frm) / frm  * 100.0 if frm  > 0 else 0.0
        v_high = abs(D - to_a) / to_a * 100.0 if to_a > 0 else 0.0
        limit  = var * 100.0
        if v_low > limit + 1e-9:
            failures.append(
                f"band {D:g} mg: below-boundary variance {v_low:.2f}% "
                f"exceeds ±{limit:.1f}% (from {frm:g} mg)"
            )
        if v_high > limit + 1e-9:
            failures.append(
                f"band {D:g} mg: above-boundary variance {v_high:.2f}% "
                f"exceeds ±{limit:.1f}% (to {to_a:g} mg)"
            )

        # 3. Coverage
        if prev_to is not None and frm > prev_to + 1e-9:
            failures.append(
                f"coverage gap between {prev_to:g} mg and {frm:g} mg"
            )
        prev_to = to_a

    return failures


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

        if v_max > var + 1e-9:    # fp epsilon only — see BOUNDARY_DP note
            violations.append((D, v_max * 100))

        # Coverage gap check
        if prev_to is not None and frm > prev_to + 1e-9:
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
    # Complete transcription of National-Dose-Banding-Table-20mgmL-v7.pdf
    # (NHS England, Version 7, August 2022): all 73 published rows,
    # 5.00-6368.68 mg. To(B) is omitted; it is always To(A) - 0.01.
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
    # Rows above 1058.30 mg, transcribed 2026-08-01 from the published PDF.
    # These complete the table: 73 rows, ending at To(A) = 6368.68 mg.
    (1058.30, 1187.94, 1120.0), (1187.94, 1328.16, 1260.0),
    (1328.16, 1487.28, 1400.0), (1487.28, 1677.02, 1580.0),
    (1677.02, 1886.80, 1780.0), (1886.80, 2126.02, 2000.0),
    (2126.02, 2405.32, 2260.0), (2405.32, 2724.70, 2560.0),
    (2724.70, 3084.16, 2900.0), (3084.16, 3483.68, 3280.0),
    (3483.68, 3932.68, 3700.0), (3932.68, 4441.80, 4180.0),
    (4441.80, 5020.44, 4720.0), (5020.44, 5660.38, 5340.0),
    (5660.38, 6368.68, 6000.0),
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
    # vial optimisation — empty strings for drugs without vial_sizes_mg
    "vial_combination", "waste_mg", "waste_pct", "waste_cost", "vial_optimized",
    # provenance — deliberately NOT added to the Cerner export, which must
    # match the standardised dose range screen field for field
    "algorithm_version",
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
     "drug_type": "traditional", "min_dose_mg": 200, "max_dose_mg": 1060},
    {"drug_name": "ExampleDrug_B", "concentration_mg_per_ml": 10,
     "drug_type": "mab",         "min_dose_mg": 400, "max_dose_mg": 2000},
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
            vial_sizes   = parse_vial_sizes(drug.get("vial_sizes_mg", ""))
            max_per_vial = int(drug.get("max_vials_per_size", 8) or 8)
            vials_shared = parse_vials_shared(drug.get("vials_shared", "no"))
            raw_cost     = drug.get("cost_per_mg", "")
            cost_per_mg  = float(raw_cost) if raw_cost not in ("", None) else None
            bands = build_bands(drug, vial_sizes, max_per_vial, vials_shared, cost_per_mg)
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
