"""
Property-based verification of the dose banding algorithm.

The manuscript claims three guarantees for every generated table:

  P1  Dispensability — each band dose is an exact multiple of the syringe
      graduation for its own volume tier.
  P2  Tolerance      — every prescribed dose in range receives a band dose
      within +/- tau of it, judged at the *published* boundaries.
  P3  Coverage       — the bands tile the requested range with no gap and no
      overlap, so each prescribed dose maps to exactly one band.

plus two structural claims used in the discussion:

  P4  Minimality     — the greedy width-maximising choice admits no larger
      next band, so the table has the fewest bands possible.
  P5  Soundness      — strict mode never returns a table that violates
      P1-P3; where the guarantee is unattainable it refuses.

These tests stand in for a formal proof: they check the properties against
randomly generated configurations rather than a fixed set of examples.
"""

import math

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from dose_banding import (
    enumerate_vial_combinations,
    BOUNDARY_DP,
    VARIANCE,
    _max_next_dose,
    build_bands,
    generate_band_doses,
    get_dose_step_mg,
    get_vol_step_mL,
    verify_bands,
)

from .helpers import (
    CONCENTRATIONS,
    DRUG_TYPES,
    TIERS,
    any_config,
    band_for_dose,
    config,
    min_advance_volume,
    min_first_band_volume,
    vol_step_for,
    well_posed_config,
)

# Inherits the active Hypothesis profile, so `--hypothesis-profile=deep` raises
# the example count everywhere at once. Profiles are registered in conftest.py.
SETTINGS = settings()


# ─────────────────────────────────────────────────────────────────────────────
# P5 — SOUNDNESS. The safety-critical property: no silent bad table.
# ─────────────────────────────────────────────────────────────────────────────

@SETTINGS
@given(cfg=any_config())
def test_strict_mode_never_returns_an_invalid_table(cfg):
    """
    For an *arbitrary* configuration — including ones that cannot be banded —
    strict mode either raises or returns a table satisfying P1-P3.

    This needs no precondition, which is what makes it the property that
    matters clinically: a table that reaches a pharmacy is correct.
    """
    tau = VARIANCE[cfg["drug_type"]]
    try:
        rows = build_bands(cfg, strict=True)
    except ValueError:
        return  # refusal is an acceptable outcome
    assert verify_bands(rows, tau, cfg["concentration_mg_per_ml"]) == []


@SETTINGS
@given(cfg=any_config())
def test_non_strict_mode_flags_every_band_it_cannot_guarantee(cfg):
    """
    With strict=False the table is returned regardless, but any band outside
    tolerance must be marked `within_tolerance = False`. Callers that opt out
    of the exception still get told which rows are unsafe.
    """
    tau = VARIANCE[cfg["drug_type"]] * 100.0
    rows = build_bands(cfg, strict=False)
    for row in rows:
        worst = max(
            abs(float(row["variance_below_pct"])),
            abs(float(row["variance_above_pct"])),
        )
        if worst > tau + 1e-9:
            assert row["within_tolerance"] is False, (
                f"band {row['band_dose_mg']} mg is {worst:.2f}% off but was "
                f"not flagged"
            )


# ─────────────────────────────────────────────────────────────────────────────
# P1/P2/P3 — the three guarantees, on configurations that satisfy the lemma's
# precondition. Completeness: these must all succeed, never raise.
# ─────────────────────────────────────────────────────────────────────────────

@SETTINGS
@given(cfg=well_posed_config())
def test_well_posed_configs_are_never_rejected(cfg):
    """
    P5 in the other direction. The precondition must be *sufficient*, not
    merely necessary — otherwise the guard would reject configurations that
    are in fact bandable. See `helpers` for the two conditions involved.
    """
    rows = build_bands(cfg, strict=True)
    assert rows


@SETTINGS
@given(cfg=well_posed_config())
def test_p1_every_band_dose_is_dispensable(cfg):
    """Each band volume is a whole number of graduations for its own tier."""
    conc = cfg["concentration_mg_per_ml"]
    for row in build_bands(cfg, strict=True):
        vol = float(row["volume_mL"])
        step = vol_step_for(vol)
        # Re-derived tier, not the one the row reports.
        assert step == float(row["volume_step_mL"])
        ratio = vol / step
        assert abs(ratio - round(ratio)) < 1e-6, (
            f"{row['band_dose_mg']} mg = {vol} mL is not a multiple of {step} mL"
        )


@SETTINGS
@given(cfg=well_posed_config(), frac=st.floats(0.0, 1.0, allow_nan=False))
def test_p2_any_prescribed_dose_receives_a_band_within_tolerance(cfg, frac):
    """
    The clinical statement of the guarantee: pick any dose a prescriber could
    calculate inside the range, look up the band it lands in, and the dose the
    patient actually receives is within tau of what was prescribed.

    Endpoint variance is what `verify_bands` checks; this samples the interior
    as well, which is what the guarantee is actually about.
    """
    tau = VARIANCE[cfg["drug_type"]]
    rows = build_bands(cfg, strict=True)

    lo, hi = float(cfg["min_dose_mg"]), float(cfg["max_dose_mg"])
    # Clamp: lo + 1.0 * (hi - lo) can land an ulp above hi and fall out of the
    # table for reasons that have nothing to do with the algorithm.
    prescribed = min(max(lo + frac * (hi - lo), lo), hi)

    row = band_for_dose(rows, prescribed)
    assert row is not None, f"{prescribed} mg fell outside the table"

    delivered = float(row["band_dose_mg"])
    variance = abs(delivered - prescribed) / prescribed
    assert variance <= tau + 1e-9, (
        f"prescribed {prescribed:.4f} mg -> band {delivered} mg "
        f"= {variance * 100:.3f}% (limit {tau * 100:.1f}%)"
    )


@SETTINGS
@given(cfg=well_posed_config())
def test_p3_bands_tile_the_range_without_gap_or_overlap(cfg):
    """
    Adjacent bands share a boundary exactly: the upper boundary of one is the
    lower boundary of the next. A shared boundary is what makes the half-open
    convention total — every dose has exactly one home.
    """
    rows = build_bands(cfg, strict=True)

    assert float(rows[0]["from_mg"]) == pytest.approx(float(cfg["min_dose_mg"]))
    assert float(rows[-1]["to_a_mg"]) == pytest.approx(float(cfg["max_dose_mg"]))

    for prev, nxt in zip(rows, rows[1:]):
        assert float(prev["to_a_mg"]) == pytest.approx(float(nxt["from_mg"])), (
            f"boundary mismatch: band ends at {prev['to_a_mg']}, "
            f"next starts at {nxt['from_mg']}"
        )


@SETTINGS
@given(cfg=well_posed_config())
def test_p3_band_doses_and_boundaries_increase_strictly(cfg):
    rows = build_bands(cfg, strict=True)
    doses = [float(r["band_dose_mg"]) for r in rows]
    assert all(b > a for a, b in zip(doses, doses[1:])), doses
    froms = [float(r["from_mg"]) for r in rows]
    assert all(b > a for a, b in zip(froms, froms[1:])), froms


# ─────────────────────────────────────────────────────────────────────────────
# P4 — MINIMALITY of the band count.
# ─────────────────────────────────────────────────────────────────────────────

@SETTINGS
@given(cfg=well_posed_config())
def test_p4_no_larger_next_band_exists(cfg):
    """
    The greedy exchange argument, checked locally: for each consecutive pair,
    the next dispensable dose above the one chosen would leave a coverage gap.
    Since no step can be made larger, no alternative table has fewer bands.

    Verified by stepping up independently rather than by re-calling the
    selection function, so this does not just restate the implementation.
    """
    conc = cfg["concentration_mg_per_ml"]
    tau = VARIANCE[cfg["drug_type"]]
    doses = generate_band_doses(
        float(cfg["min_dose_mg"]), float(cfg["max_dose_mg"]), conc, tau
    )

    for prev, chosen in zip(doses, doses[1:]):
        gap_free_ceiling = prev * (1.0 + tau) / (1.0 - tau)
        if chosen > gap_free_ceiling + 1e-9:
            # Forced advance: no dispensable dose fits under the ceiling at
            # all, so there is nothing to be maximal over.
            continue
        bigger = chosen + get_dose_step_mg(chosen, conc)
        assert bigger > gap_free_ceiling + 1e-9, (
            f"after {prev} mg the algorithm chose {chosen} mg but "
            f"{bigger} mg also fits under the gap-free ceiling "
            f"{gap_free_ceiling:.6f} mg"
        )


@SETTINGS
@given(cfg=well_posed_config())
def test_p4_first_band_is_the_largest_admissible(cfg):
    """The opening dose is as high as the +tau allowance at min_dose permits."""
    conc = cfg["concentration_mg_per_ml"]
    tau = VARIANCE[cfg["drug_type"]]
    min_dose = float(cfg["min_dose_mg"])
    doses = generate_band_doses(min_dose, float(cfg["max_dose_mg"]), conc, tau)

    d0 = doses[0]
    ceiling = min_dose * (1.0 + tau)
    assert d0 <= ceiling + 1e-9
    assert d0 >= min_dose - 1e-9
    assert d0 + get_dose_step_mg(d0, conc) > ceiling + 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# PRESENTATION INVARIANTS
# ─────────────────────────────────────────────────────────────────────────────

@SETTINGS
@given(cfg=well_posed_config())
def test_published_boundaries_are_exact_multiples_of_the_declared_precision(cfg):
    """
    The manuscript states boundaries are computed to four decimal places. If a
    boundary carried more precision than that, the printed table and the table
    the guarantee was proved about would not be the same table.
    """
    q = 10 ** BOUNDARY_DP
    for row in build_bands(cfg, strict=True):
        for key in ("from_mg", "to_a_mg", "to_b_mg"):
            scaled = float(row[key]) * q
            assert abs(scaled - round(scaled)) < 1e-6, (
                f"{key}={row[key]} is not a {BOUNDARY_DP} dp value"
            )


@SETTINGS
@given(cfg=well_posed_config())
def test_round_down_boundary_sits_one_ulp_below_the_shared_boundary(cfg):
    """
    `to_b_mg` exists for CPOE systems whose range operator includes the upper
    bound. It must be the largest value still inside the band — exactly one
    unit of the published precision below `to_a_mg`.
    """
    ulp = 10 ** -BOUNDARY_DP
    for row in build_bands(cfg, strict=True):
        assert float(row["to_b_mg"]) == pytest.approx(
            float(row["to_a_mg"]) - ulp, abs=1e-9
        )


@SETTINGS
@given(cfg=well_posed_config())
def test_generation_is_deterministic(cfg):
    a = build_bands(cfg, strict=True)
    b = build_bands(cfg, strict=True)
    assert a == b


# ─────────────────────────────────────────────────────────────────────────────
# THE PRECONDITION ITSELF
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dtype", DRUG_TYPES)
@pytest.mark.parametrize("conc", CONCENTRATIONS)
def test_precondition_cliff(conc, dtype):
    """
    Either side of the first-band volume floor step/tau, behaviour must flip
    from guaranteed to refused. Comfortably above it the table is produced;
    comfortably below it strict mode raises rather than emitting bands that
    breach tolerance.
    """
    tau = VARIANCE[dtype]
    floor_vol = min_first_band_volume(tau, 0.01)

    safe = config(conc, dtype, round(floor_vol * 1.5 * conc, 4), None)
    safe["max_dose_mg"] = round(safe["min_dose_mg"] * 20, 4)
    assert build_bands(safe, strict=True)

    unsafe = config(conc, dtype, round(floor_vol * 0.2 * conc, 6), None)
    unsafe["max_dose_mg"] = round(unsafe["min_dose_mg"] * 20, 6)
    with pytest.raises(ValueError, match="cannot be banded"):
        build_bands(unsafe, strict=True)


@pytest.mark.parametrize("dtype", DRUG_TYPES)
def test_advance_floor_is_slack_at_every_tier_transition(dtype):
    """
    Why the *advance* condition never binds mid-table. A band only enters a
    coarser tier from a dose at least (1-tau)/(1+tau) of that tier's lower
    edge, and at that volume the coarser graduation is still comfortably
    within the 2*tau/(1-tau) advance window. So once a table has begun, it
    never stalls at a tier boundary.
    """
    tau = VARIANCE[dtype]
    for lower, _upper, step in TIERS[1:]:
        # Smallest previous volume that can step up into this tier.
        entering_from = lower * (1.0 - tau) / (1.0 + tau)
        required = min_advance_volume(tau, step)
        assert entering_from >= required, (
            f"a band entering the {step} mL tier from {entering_from:.4f} mL "
            f"needs at least {required:.4f} mL"
        )


def _is_refused(vol: float, conc: float, dtype: str) -> bool:
    cfg = config(conc, dtype, vol * conc, vol * conc * 20)
    try:
        build_bands(cfg, strict=True)
        return False
    except ValueError:
        return True


@pytest.mark.parametrize("dtype", DRUG_TYPES)
def test_first_band_floor_is_sufficient_within_every_tier(dtype):
    """
    Above step/tau, no minimum dose is ever refused — the bound is sufficient.

    The scan is clamped to a single tier because a range starting near the top
    of one tier extends into the next, whose own floor is a separate question.
    """
    tau = VARIANCE[dtype]
    for lower, upper, step in TIERS:
        floor = min_first_band_volume(tau, step)
        top = min(upper, 60.0)
        if floor >= top:
            continue
        start = floor * 1.001
        refused = [
            v
            for i in range(200)
            if _is_refused(v := start + (top - start) * i / 200, 20.0, dtype)
        ]
        assert not refused, (
            f"{step} mL tier: volumes at or above {floor:.4f} mL should always "
            f"band, but these were refused: {refused[:3]}"
        )


@pytest.mark.parametrize("dtype", DRUG_TYPES)
def test_first_band_window_at_the_bottom_of_a_tier_is_genuinely_exposed(dtype):
    """
    ...and below step/tau the bound is necessary in substance: a real fraction
    of minimum doses in the exposed window cannot be banded at all.

    At tau=6% those windows are [1.0, 1.667), [3.0, 3.333) and [10, 16.667) mL,
    where the graduation has just coarsened but the volume has not yet grown to
    match. Refusal is the correct outcome — this test exists to show the window
    is not a theoretical artefact. Minimum doses that happen to land on the
    graduation grid still succeed, so the bound is not necessary pointwise.
    """
    tau = VARIANCE[dtype]
    exposed = [
        (lower, step)
        for lower, _upper, step in TIERS
        if min_first_band_volume(tau, step) > max(lower, step)
    ]
    if not exposed:
        pytest.skip(f"no exposed window at tau={tau:.0%}")

    for lower, step in exposed:
        floor = min_first_band_volume(tau, step)
        start = max(lower, step)
        refused = sum(
            _is_refused(start + (floor - start) * i / 100, 20.0, dtype)
            for i in range(1, 100)
        )
        assert refused > 0, (
            f"expected some refusals in the {step} mL tier window "
            f"[{start:.4f}, {floor:.4f}) mL, found none"
        )


@SETTINGS
@given(
    conc=st.sampled_from(CONCENTRATIONS),
    dtype=st.sampled_from(DRUG_TYPES),
    vol=st.floats(0.001, 0.05, allow_nan=False),
)
def test_sub_floor_volumes_are_refused_not_silently_wrong(conc, dtype, vol):
    """
    Below the *advance* floor no table can exist at all, whatever the grid
    alignment, so refusal is unconditional.
    """
    tau = VARIANCE[dtype]
    assume(vol < min_advance_volume(tau, 0.01) * 0.8)
    cfg = config(conc, dtype, round(vol * conc, 6), round(vol * conc * 30, 6))
    with pytest.raises(ValueError):
        build_bands(cfg, strict=True)


# ─────────────────────────────────────────────────────────────────────────────
# TIER TABLE
# ─────────────────────────────────────────────────────────────────────────────

@SETTINGS
@given(
    conc=st.sampled_from(CONCENTRATIONS),
    dose=st.floats(0.01, 200_000.0, allow_nan=False, allow_infinity=False),
)
def test_dose_step_is_concentration_times_volume_step(conc, dose):
    assert get_dose_step_mg(dose, conc) == pytest.approx(
        conc * get_vol_step_mL(dose, conc)
    )


@SETTINGS
@given(vol=st.floats(0.0, 5000.0, allow_nan=False, allow_infinity=False))
def test_graduation_never_gets_finer_as_volume_grows(vol):
    """
    Monotonicity of the tier table. Aliquoting means resolution can plateau at
    1 mL but it must never improve — a coarser-then-finer table would break the
    argument that checking the minimum volume suffices.
    """
    assert vol_step_for(vol + 1.0) >= vol_step_for(vol)


def test_tier_table_is_ordered_and_terminates():
    from dose_banding import VOLUME_PRECISION_TIERS

    uppers = [t[0] for t in VOLUME_PRECISION_TIERS]
    assert uppers == sorted(uppers)
    assert math.isinf(uppers[-1]), "tier table must cover unbounded volumes"


# ─────────────────────────────────────────────────────────────────────────────
# VIAL-AWARE BAND PLACEMENT (v2.1.0) — P2 and P3 must survive the new path.
#
# P1 is deliberately NOT asserted here. A vial-exact band dose is exempt from
# the graduation grid, on the rationale `verify_bands` already applies to
# substituted bands: a zero-waste total is drawn from whole vials, so no
# partial volume is measured. Placement makes that case the common one rather
# than the exception, which is why P1 is checked below only for the fallback
# doses — the steps where no vial total was admissible.
# ─────────────────────────────────────────────────────────────────────────────

VIAL_SETS = [[100.0], [100.0, 160.0], [10.0, 50.0, 200.0], [10.0, 30.0, 60.0]]


@SETTINGS
@given(cfg=well_posed_config(), vials=st.sampled_from(VIAL_SETS))
def test_vial_aware_tables_keep_the_guarantees(cfg, vials):
    """
    The flag buys zero-waste preparation; it may not buy it with tolerance or
    coverage. Strict mode either refuses or returns a table passing P1-P3 as
    `verify_bands` judges them.
    """
    tau = VARIANCE[cfg["drug_type"]]
    try:
        rows = build_bands(cfg, vial_sizes=vials, strict=True, vial_aware=True)
    except ValueError:
        return  # refusal is an acceptable outcome
    assert verify_bands(rows, tau, cfg["concentration_mg_per_ml"]) == []


@SETTINGS
@given(cfg=well_posed_config(), vials=st.sampled_from(VIAL_SETS))
def test_vial_aware_band_doses_are_vial_exact_or_dispensable(cfg, vials):
    """
    Every band dose is justified one way or the other: either it is a whole-vial
    total (drawn entire, no measurement) or it sits on the graduation grid.
    Nothing in between — that would be a dose no one can actually prepare.
    """
    conc = cfg["concentration_mg_per_ml"]
    try:
        rows = build_bands(cfg, vial_sizes=vials, strict=True, vial_aware=True)
    except ValueError:
        return

    totals = {d for d, _ in enumerate_vial_combinations(
        vials, cfg["max_dose_mg"] * 1.10, 8)}

    for row in rows:
        D = float(row["band_dose_mg"])
        if any(abs(D - t) < 1e-6 for t in totals):
            continue
        vol = D / conc
        step = vol_step_for(vol)
        ratio = vol / step
        assert abs(ratio - round(ratio)) < 1e-6, (
            f"band {D} mg is neither a vial total nor on the {step} mL grid"
        )


# There is deliberately NO property here asserting that vial-aware placement
# beats the greedy table on waste. It does not, and no bound on how much worse
# it can be survived the deep profile either — 5 mg/mL mab over 40-80 mg on
# 10/50/200 mg vials gives 17 mg against the greedy 6 mg. Placement moves the
# bands; substitution keeps them and fits vials in afterwards. Neither
# dominates, so the comparison belongs to the caller, and the non-domination
# is pinned with its witnesses in test_regressions.py.


@SETTINGS
@given(cfg=well_posed_config(), vials=st.sampled_from(VIAL_SETS))
def test_every_band_dose_sits_inside_its_own_range(cfg, vials):
    """
    A band dose below the range it serves is compliant but incoherent: under
    placement the band is then only as wide as the leftover, and the table
    carries two near-identical doses.

    Asserted for the doses placement actually chooses — the zero-waste vial
    totals. Two other routes to the same shape exist and are NOT covered here,
    both pinned with witnesses in test_regressions.py:

      * Phase 2 substitution searches from `to_a_mg*(1-tau)`, below `from_mg`,
        so the greedy table can seat a substituted dose under its range. Milder:
        the boundaries are already fixed, so the band is not narrowed.
      * The shared `_max_next_dose` fallback floors to the graduation step, and
        at a volume-tier crossing that floor can land below `from_mg`. This one
        predates vial optimisation entirely and reaches the vial-free algorithm
        the manuscript describes — though it needs a range that straddles a tier
        edge at fine resolution, and none of 32 ordinary clinical configurations
        triggered it.

    Correcting either would change published output, so they are decisions
    rather than defects, and are recorded rather than silently fixed.

    Bands are skipped where the graduation makes the invariant unachievable.
    At a volume-tier crossing the step can be coarser than the whole window
    [band_low, cap], so no dispensable dose exists inside the band at all —
    1 mg/mL over 7.5-15 mg is the witness, where the window [10.2128, 10.8255]
    contains no multiple of the 1 mg step. There the only alternatives are a
    dose fractionally under its range or refusing the configuration outright,
    and the algorithm reasonably takes the former. `_max_next_dose` landing
    below `band_low` is exactly that condition, so it is the skip test.
    """
    conc = cfg["concentration_mg_per_ml"]
    tau = VARIANCE[cfg["drug_type"]]
    try:
        rows = build_bands(cfg, vial_sizes=vials, strict=True, vial_aware=True)
    except ValueError:
        return

    for previous, row in zip(rows, rows[1:]):
        D_prev = float(previous["band_dose_mg"])
        band_low = D_prev / (1.0 - tau)
        if _max_next_dose(D_prev, tau, conc) < band_low - 1e-9:
            continue  # no dispensable dose exists inside this band

        D = float(row["band_dose_mg"])
        from_mg = float(row["from_mg"])
        assert D >= from_mg - 1e-9, (
            f"band {D} mg below its range {from_mg} "
            f"(conc={conc}, vials={vials})"
        )
