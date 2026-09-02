"""
Verification of split-aware band placement (multi-syringe doses).

Where a dose is too large for one syringe it is split across several, and each
aliquot has to sit on the graduation of the barrel it is drawn in. For an equal
split into k syringes the total volume must therefore be a multiple of
k x graduation — a lattice k times coarser than the one the default algorithm
uses — so a band dose chosen for its total volume alone usually does not
survive division.

The properties claimed for a split-active table are the three the algorithm
already guarantees, plus:

  S1  Divisibility   — an aligned band divides into k identical fills, each an
      exact multiple of the graduation of the barrel it is drawn in.
  S2  Minimality     — a split never uses more syringes than the volume forces.
  S3  Capacity       — no fill exceeds its barrel's fill limit or the route cap.
  S4  Conservation   — the fills sum to the band volume.
  S5  Non-regression — without a route_profile, output is exactly as before.

As in `helpers`, nothing here asks `dose_banding` what the right answer is.
Barrel selection and graduation are re-derived from `SYRINGE_INVENTORY`, which
is declared data, so a defect in the module's own helpers cannot hide.
"""

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dose_banding import (
    HAZARDOUS_FILL_FRACTION,
    ROUTE_PROFILES,
    SYRINGE_INVENTORY,
    VARIANCE,
    VOLUME_PRECISION_TIERS,
    build_bands,
    build_syringe_set,
    enumerate_split_totals,
    generate_band_doses,
    parse_route_profile,
    validate_tolerance_and_coverage,
)

from .helpers import config

# Drugs whose volumes force a split at a clinically real concentration. The
# anthracyclines at 2 mg/mL are the motivating case; the dilute agents exercise
# the fills that land near the 7.5 mL graduation cliff.
SPLIT_CASES = [
    (2.0, 60.0, 220.0, "traditional"),    # epirubicin
    (2.0, 40.0, 160.0, "traditional"),    # doxorubicin
    (2.0, 30.0, 120.0, "traditional"),    # liposomal doxorubicin
    (2.0, 100.0, 300.0, "traditional"),   # high-dose epirubicin
    (1.0, 20.0, 90.0, "traditional"),     # dilute
    (0.5, 10.0, 60.0, "traditional"),     # very dilute
]


# ─────────────────────────────────────────────────────────────────────────────
# INDEPENDENT RE-DERIVATION FROM THE DECLARED INVENTORY
# ─────────────────────────────────────────────────────────────────────────────

def barrels(route: str = "iv_push", fraction: float = HAZARDOUS_FILL_FRACTION):
    """
    Re-derive the usable barrels from SYRINGE_INVENTORY.

    Deliberately does not call `build_syringe_set` — that is the code under
    test. Returns (capacity, graduation, largest usable fill), smallest first,
    with barrels that no volume could ever select removed.
    """
    route_max = ROUTE_PROFILES[route]
    out, reachable = [], 0.0
    for capacity, graduation, override in sorted(SYRINGE_INVENTORY):
        fill = override if override is not None else fraction * capacity
        if route_max is not None:
            fill = min(fill, route_max)
        if fill <= reachable + 1e-9:
            continue
        out.append((capacity, graduation, fill))
        reachable = fill
    return out


def barrel_for(fill_mL: float, route: str = "iv_push"):
    """Smallest barrel that can hold `fill_mL`, or None."""
    for capacity, graduation, usable in barrels(route):
        if fill_mL <= usable + 1e-9:
            return capacity, graduation, usable
    return None


def on_graduation(volume_mL: float, graduation: float) -> bool:
    return abs(volume_mL / graduation - round(volume_mL / graduation)) < 1e-6


def split_rows(conc, lo, hi, dtype="traditional", route="iv_push", **kw):
    return build_bands(config(conc, dtype, lo, hi), route_profile=route, **kw)


def fills_of(row):
    """Per-syringe fills, re-parsed from the published `syringe_split` string."""
    body = row["syringe_split"].split(" (")[0]
    return [float(part.strip().removesuffix(" mL")) for part in body.split(" + ")]


# ─────────────────────────────────────────────────────────────────────────────
# S5 — NON-REGRESSION
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("conc,lo,hi,dtype", SPLIT_CASES + [
    (20.0, 5.0, 380.0, "traditional"),    # the NHS v7 comparison range
    (6.0, 5.0, 300.0, "traditional"),     # the NHS v2 comparison range
])
def test_default_placement_is_untouched(conc, lo, hi, dtype):
    """Without a route_profile the band doses are exactly the greedy ones."""
    rows = build_bands(config(conc, dtype, lo, hi))
    expected = generate_band_doses(lo, hi, conc, VARIANCE[dtype])
    assert [r["band_dose_mg"] for r in rows] == [round(d, 4) for d in expected]


@pytest.mark.parametrize("conc,lo,hi,dtype", SPLIT_CASES)
def test_split_columns_are_empty_without_a_route_profile(conc, lo, hi, dtype):
    for row in build_bands(config(conc, dtype, lo, hi)):
        assert row["route_profile"] == ""
        assert row["n_syringes"] == ""
        assert row["syringe_split"] == ""
        assert row["split_aligned"] == ""
        assert row["exceeds_max_syringes"] == ""


# ─────────────────────────────────────────────────────────────────────────────
# S1-S4 — THE SPLIT ITSELF
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("conc,lo,hi,dtype", SPLIT_CASES)
def test_aligned_bands_divide_onto_the_barrel_graduation(conc, lo, hi, dtype):
    """S1: k identical fills, each on the graduation of the barrel used."""
    aligned = [r for r in split_rows(conc, lo, hi, dtype) if r["split_aligned"]]
    assert aligned, "expected at least one cleanly split band"
    for row in aligned:
        fills = fills_of(row)
        assert len(set(fills)) == 1, f"{row['band_dose_mg']} mg: fills differ"
        assert len(fills) == row["n_syringes"]
        found = barrel_for(fills[0])
        assert found is not None
        _capacity, graduation, _usable = found
        assert on_graduation(fills[0], graduation), (
            f"{row['band_dose_mg']} mg: {fills[0]} mL is not on the "
            f"{graduation} mL graduation"
        )


@pytest.mark.parametrize("conc,lo,hi,dtype", SPLIT_CASES)
def test_a_split_never_uses_more_syringes_than_the_volume_forces(conc, lo, hi, dtype):
    """
    S2. Without this guard the enumeration returns 46.4 mg as 4 x 5.8 mL when
    23.2 mL fits a single barrel — arithmetically valid, clinically absurd.
    """
    largest = max(usable for _c, _g, usable in barrels())
    for row in split_rows(conc, lo, hi, dtype):
        fewest = max(1, math.ceil(row["volume_mL"] / largest - 1e-9))
        assert row["n_syringes"] == fewest, (
            f"{row['band_dose_mg']} mg = {row['volume_mL']} mL used "
            f"{row['n_syringes']} syringes where {fewest} suffice"
        )


@pytest.mark.parametrize("conc,lo,hi,dtype", SPLIT_CASES)
def test_no_fill_exceeds_its_barrel_or_the_route_cap(conc, lo, hi, dtype):
    """S3: the fill limit and the ergonomic push cap both hold."""
    route_cap = ROUTE_PROFILES["iv_push"]
    for row in split_rows(conc, lo, hi, dtype):
        for fill in fills_of(row):
            found = barrel_for(fill)
            assert found is not None, f"{fill} mL fits no stocked barrel"
            capacity, _graduation, usable = found
            assert fill <= usable + 1e-9
            assert fill <= HAZARDOUS_FILL_FRACTION * capacity + 1e-9
            assert fill <= route_cap + 1e-9


@pytest.mark.parametrize("conc,lo,hi,dtype", SPLIT_CASES)
def test_fills_sum_to_the_band_volume(conc, lo, hi, dtype):
    """S4."""
    for row in split_rows(conc, lo, hi, dtype):
        assert abs(sum(fills_of(row)) - row["volume_mL"]) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# TOLERANCE AND COVERAGE STILL HOLD
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("conc,lo,hi,dtype", SPLIT_CASES)
def test_tolerance_and_coverage_hold_with_splitting(conc, lo, hi, dtype):
    result = validate_tolerance_and_coverage(split_rows(conc, lo, hi, dtype),
                                             VARIANCE[dtype])
    assert result["violations"] == []
    assert result["gaps"] == []


@given(
    conc=st.sampled_from([0.5, 1.0, 2.0, 3.0, 5.0]),
    dtype=st.sampled_from(sorted(VARIANCE)),
    min_vol=st.floats(min_value=8.0, max_value=60.0),
    ratio=st.floats(min_value=1.5, max_value=8.0),
)
@settings(max_examples=150)
def test_splitting_is_sound_over_a_sweep(conc, dtype, min_vol, ratio):
    """
    Whatever it is handed, strict mode either refuses or returns a table whose
    tolerance and coverage hold — the P5 soundness property, with splitting on.
    """
    lo = round(min_vol * conc, 4)
    cfg = config(conc, dtype, lo, round(lo * ratio, 4))
    try:
        rows = build_bands(cfg, route_profile="iv_push")
    except ValueError:
        return  # refusing is a correct outcome
    result = validate_tolerance_and_coverage(rows, VARIANCE[dtype])
    assert result["violations"] == []
    assert result["gaps"] == []


@pytest.mark.parametrize("conc,lo,hi,dtype", SPLIT_CASES)
def test_a_band_dose_sits_inside_its_own_band(conc, lo, hi, dtype):
    """
    The 2.1.1 guard, carried over: no band seated beneath its own range.

    Only the lower side is a guarantee. The final band's `to_a_mg` is clamped to
    the requested maximum, so its dose may sit above that while still being
    within tolerance of every dose it serves — the case the front end already
    reports as the top band exceeding the requested maximum.
    """
    rows = split_rows(conc, lo, hi, dtype)
    for i, row in enumerate(rows):
        assert row["from_mg"] - 1e-9 <= row["band_dose_mg"], (
            f"{row['band_dose_mg']} mg sits below its own band ({row['from_mg']} mg)"
        )
        if i < len(rows) - 1:
            assert row["band_dose_mg"] <= row["to_a_mg"] + 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# THE INVENTORY, THE CLIFF, AND THE FALLBACK
# ─────────────────────────────────────────────────────────────────────────────

def test_inventory_agrees_with_the_volume_precision_tiers():
    """
    The barrel graduations and the tier table describe the same syringes, so
    they must not drift. A barrel of capacity C is marked at the tier step that
    applies just below C.
    """
    for capacity, graduation, _override in SYRINGE_INVENTORY:
        expected = next(step for upper, step in VOLUME_PRECISION_TIERS
                        if capacity - 1e-9 < upper)
        assert graduation == expected, (
            f"{capacity} mL barrel declared at {graduation} mL but the tier "
            f"table implies {expected} mL"
        )


def test_the_graduation_cliff_sits_at_7_5_mL():
    """
    Crossing 7.5 mL moves from a 10 mL barrel at 0.2 mL marks to a 20 mL barrel
    at 1 mL marks — a fivefold coarsening, and the only real cliff on this route
    now that every larger barrel shares a 1 mL graduation.
    """
    assert barrel_for(7.4)[:2] == (10.0, 0.2)
    assert barrel_for(7.6)[:2] == (20.0, 1.0)
    assert on_graduation(7.4, barrel_for(7.4)[1])
    assert not on_graduation(7.6, barrel_for(7.6)[1])
    assert on_graduation(8.0, barrel_for(8.0)[1])
    # every barrel above the cliff shares one graduation
    assert {g for _c, g, usable in barrels() if usable > 7.5} == {1.0}


def test_shadowed_barrels_are_dropped():
    """
    Under the 30 mL push cap the 50 and 60 mL barrels both cap out at 30 mL, so
    the 60 is never the smallest that fits and must not be offered.
    """
    push = [c for c, _g, _l, _h in build_syringe_set("iv_push")]
    assert 60.0 not in push and 50.0 in push
    # without a route cap the two are distinguishable again
    plain = [c for c, _g, _l, _h in build_syringe_set("syringe")]
    assert 60.0 in plain


def test_a_band_with_no_clean_split_is_flagged_not_dropped():
    """
    Where no admissible dose divides cleanly the band still exists, prepared as
    successive aliquots, and says so.
    """
    rows = split_rows(5.0, 200.0, 900.0, "mab")
    fallback = [r for r in rows if not r["split_aligned"]]
    assert fallback, "expected this configuration to force a fallback"
    for row in fallback:
        assert row["syringe_split"]
        assert abs(sum(fills_of(row)) - row["volume_mL"]) < 1e-6
    # coverage is unaffected by the fallback
    result = validate_tolerance_and_coverage(rows, VARIANCE["mab"])
    assert result["gaps"] == []


def test_bands_over_the_syringe_limit_are_flagged_not_dropped():
    rows = split_rows(2.0, 100.0, 300.0, "traditional", max_syringes=4)
    over = [r for r in rows if r["exceeds_max_syringes"]]
    assert over, "expected this configuration to exceed 4 syringes"
    for row in over:
        assert row["n_syringes"] > 4


# ─────────────────────────────────────────────────────────────────────────────
# ENUMERATION AND CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

def test_enumeration_yields_only_minimum_syringe_candidates():
    syringes = build_syringe_set("iv_push")
    largest = max(usable for _c, _g, usable in barrels())
    for dose, k in enumerate_split_totals(2.0, 400.0, syringes, max_syringes=4):
        volume = dose / 2.0
        assert k == max(1, math.ceil(volume / largest - 1e-9))
        assert on_graduation(volume / k, barrel_for(volume / k)[1])


def test_enumeration_respects_the_syringe_limit():
    syringes = build_syringe_set("iv_push")
    for _dose, k in enumerate_split_totals(2.0, 400.0, syringes, max_syringes=2):
        assert k <= 2


def test_vial_sizes_and_route_profile_cannot_be_combined():
    with pytest.raises(ValueError, match="cannot be combined"):
        build_bands(config(2.0, "traditional", 40.0, 160.0),
                    vial_sizes=[50.0], route_profile="iv_push")


def test_unknown_route_profile_is_rejected():
    with pytest.raises(ValueError, match="unknown route_profile"):
        build_bands(config(2.0, "traditional", 40.0, 160.0), route_profile="nonsense")


@pytest.mark.parametrize("raw,expected", [
    ("", None), (None, None), ("  ", None),
    ("iv_push", "iv_push"), ("IV_PUSH", "iv_push"), (" iv_push ", "iv_push"),
])
def test_route_profile_parsing(raw, expected):
    assert parse_route_profile(raw) == expected


def test_unknown_split_strategy_is_rejected():
    with pytest.raises(ValueError, match="unknown split_strategy"):
        build_bands(config(2.0, "traditional", 40.0, 160.0),
                    route_profile="iv_push", split_strategy="proportional")


# ─────────────────────────────────────────────────────────────────────────────
# THE `balanced` STRATEGY
#
# Fills may differ by one graduation, so the total need only be a multiple of
# the graduation rather than of k x graduation. The lattice therefore does not
# coarsen with k, which is where the saving comes from.
# ─────────────────────────────────────────────────────────────────────────────

def balanced_rows(conc, lo, hi, dtype="traditional", **kw):
    return build_bands(config(conc, dtype, lo, hi), route_profile="iv_push",
                       split_strategy="balanced", **kw)


@pytest.mark.parametrize("conc,lo,hi,dtype", SPLIT_CASES)
def test_balanced_fills_differ_by_at_most_one_graduation(conc, lo, hi, dtype):
    for row in balanced_rows(conc, lo, hi, dtype):
        if not row["split_aligned"]:
            continue
        fills = fills_of(row)
        found = barrel_for(max(fills))
        assert found is not None
        _capacity, graduation, _usable = found
        assert max(fills) - min(fills) <= graduation + 1e-9, row["syringe_split"]
        for fill in fills:
            assert on_graduation(fill, graduation), row["syringe_split"]


@pytest.mark.parametrize("conc,lo,hi,dtype", SPLIT_CASES)
def test_balanced_keeps_every_fill_in_one_barrel(conc, lo, hi, dtype):
    """
    Two volumes on a label is the cost `balanced` asks the user to accept. Two
    volumes AND two syringe sizes is not, so the split is confined to one
    barrel.
    """
    for row in balanced_rows(conc, lo, hi, dtype):
        if not row["split_aligned"]:
            continue
        sizes = {barrel_for(fill)[0] for fill in fills_of(row)}
        assert len(sizes) == 1, f"{row['syringe_split']} spans {sizes}"


@pytest.mark.parametrize("conc,lo,hi,dtype", SPLIT_CASES)
def test_balanced_is_never_more_bands_than_equal(conc, lo, hi, dtype):
    """
    Relaxing a constraint cannot remove candidates, so it cannot cost bands.
    This is the property that justifies offering it at all.
    """
    equal = split_rows(conc, lo, hi, dtype)
    assert len(balanced_rows(conc, lo, hi, dtype)) <= len(equal)


@pytest.mark.parametrize("conc,lo,hi,dtype", SPLIT_CASES)
def test_balanced_holds_the_split_guarantees(conc, lo, hi, dtype):
    """Conservation, capacity and minimality, exactly as for `equal`."""
    largest = max(usable for _c, _g, usable in barrels())
    for row in balanced_rows(conc, lo, hi, dtype):
        fills = fills_of(row)
        assert abs(sum(fills) - row["volume_mL"]) < 1e-6
        assert row["n_syringes"] == max(1, math.ceil(row["volume_mL"] / largest - 1e-9))
        for fill in fills:
            found = barrel_for(fill)
            assert found is not None
            capacity, _graduation, usable = found
            assert fill <= usable + 1e-9
            assert fill <= HAZARDOUS_FILL_FRACTION * capacity + 1e-9


@pytest.mark.parametrize("conc,lo,hi,dtype", SPLIT_CASES)
def test_balanced_keeps_tolerance_and_coverage(conc, lo, hi, dtype):
    result = validate_tolerance_and_coverage(balanced_rows(conc, lo, hi, dtype),
                                             VARIANCE[dtype])
    assert result["violations"] == []
    assert result["gaps"] == []


def test_balanced_enumeration_is_a_superset_of_equal():
    syringes = build_syringe_set("iv_push")
    equal = {d for d, _k in enumerate_split_totals(2.0, 400.0, syringes,
                                                   strategy="equal")}
    balanced = {d for d, _k in enumerate_split_totals(2.0, 400.0, syringes,
                                                      strategy="balanced")}
    assert equal <= balanced
    assert len(balanced) > len(equal)


def test_equal_remains_the_default():
    """The default must not change under anyone's feet — it is the safer claim."""
    cfg = config(2.0, "traditional", 60.0, 220.0)
    default = build_bands(cfg, route_profile="iv_push")
    explicit = build_bands(cfg, route_profile="iv_push", split_strategy="equal")
    assert [r["band_dose_mg"] for r in default] == [r["band_dose_mg"] for r in explicit]
    for row in default:
        if row["n_syringes"] > 1 and row["split_aligned"]:
            assert len(set(fills_of(row))) == 1
