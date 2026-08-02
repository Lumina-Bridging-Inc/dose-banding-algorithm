"""
Regression tests for defects found and corrected during development, and for
the published validation figures.

The manuscript's declaration of AI assistance states that errors were found by
author review and corrected, naming three. Each is pinned here with the exact
input that exposed it, so the corrections cannot silently regress.
"""

import csv
import math
from pathlib import Path

import pytest

from dose_banding import (
    VARIANCE,
    build_bands,
    floor_to_step,
    generate_band_doses,
    get_dose_step_mg,
    get_vol_step_mL,
    verify_bands,
    vial_aware_applies,
)

CONC_20 = 20.0
CONC_6 = 6.0
TAU = VARIANCE["traditional"]
REF_6MGML = Path(__file__).resolve().parent.parent / "nhs_6mgml_ref.csv"


def traditional(conc, min_dose, max_dose, name="Test"):
    return {
        "drug_name": name,
        "concentration_mg_per_ml": conc,
        "drug_type": "traditional",
        "min_dose_mg": min_dose,
        "max_dose_mg": max_dose,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CORRECTION 1 — first band evaluated its graduation step at the wrong dose
# ─────────────────────────────────────────────────────────────────────────────

def test_first_band_step_is_evaluated_at_the_candidate_not_the_minimum():
    """
    `generate_band_doses` originally took the graduation step at `min_dose` but
    floored `min_dose * (1 + tau)`, which can land in a coarser volume tier.

    19.4 mg at 20 mg/mL is the minimal witness: min_dose is 0.97 mL (0.01 mL
    graduation) but the candidate is 1.02 mL, which is in the 0.1 mL tier. The
    old code emitted 20.4 mg = 1.02 mL, a volume no syringe can measure.
    """
    doses = generate_band_doses(19.4, 200.0, CONC_20, TAU)
    d0 = doses[0]

    # The defect, reproduced from the specification rather than from history.
    buggy = floor_to_step(19.4 * (1 + TAU), get_dose_step_mg(19.4, CONC_20))
    assert buggy == pytest.approx(20.4), "witness no longer reproduces the defect"
    assert buggy / CONC_20 == pytest.approx(1.02)

    assert d0 == pytest.approx(20.0)
    vol = d0 / CONC_20
    step = get_vol_step_mL(vol * CONC_20, CONC_20)
    assert abs(vol / step - round(vol / step)) < 1e-9, (
        f"first band {d0} mg = {vol} mL is off the {step} mL graduation"
    )


@pytest.mark.parametrize("min_dose", [19.4, 19.5, 19.6, 19.8, 19.9, 59.5, 199.5])
def test_first_band_lands_on_graduation_across_tier_crossings(min_dose):
    """The same crossing at each tier edge, not just the reported witness."""
    d0 = generate_band_doses(min_dose, min_dose * 20, CONC_20, TAU)[0]
    vol = d0 / CONC_20
    step = get_vol_step_mL(d0, CONC_20)
    assert abs(vol / step - round(vol / step)) < 1e-6
    assert min_dose - 1e-9 <= d0 <= min_dose * (1 + TAU) + 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# CORRECTION 2 — boundaries floored to 2 dp broke the tolerance guarantee
# ─────────────────────────────────────────────────────────────────────────────

def test_two_decimal_place_boundaries_would_breach_tolerance():
    """
    Flooring `from_mg` to 2 dp moves the boundary down by up to 0.01 mg, which
    at small absolute doses is a large relative perturbation and inflates the
    below-band variance computed from the *published* boundary.

    The worst case found in a 306-configuration sweep was the 2.25 mg band at
    25 mg/mL: 6.13% at 2 dp against a 6.00% limit. At 4 dp it is within limit.
    """
    rows = build_bands(traditional(25.0, 2.0, 100.0), strict=True)
    band = next(r for r in rows if float(r["band_dose_mg"]) == pytest.approx(2.25))

    frm_4dp = float(band["from_mg"])
    var_4dp = (2.25 - frm_4dp) / frm_4dp * 100.0
    assert var_4dp <= 6.0 + 1e-9, f"4 dp boundary gives {var_4dp:.3f}%"

    frm_2dp = math.floor(round(frm_4dp, 8) * 100) / 100
    var_2dp = (2.25 - frm_2dp) / frm_2dp * 100.0
    assert var_2dp > 6.0, (
        "the 2 dp witness no longer breaches tolerance — if the algorithm "
        "changed, re-derive the worst case before relaxing BOUNDARY_DP"
    )
    assert var_2dp == pytest.approx(6.13, abs=0.01)


def test_no_rounding_allowance_is_needed_at_four_decimal_places():
    """
    `within_tolerance` used to carry an ad-hoc +0.11% allowance to absorb 2 dp
    flooring. At 4 dp the published boundaries satisfy the limit exactly, so
    every band must pass on the true limit with only a floating-point epsilon.
    """
    rows = build_bands(traditional(25.0, 2.0, 100.0), strict=True)
    assert all(r["within_tolerance"] for r in rows)
    worst = max(
        max(abs(r["variance_below_pct"]), abs(r["variance_above_pct"])) for r in rows
    )
    assert worst <= 6.0 + 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# CORRECTION 3 — volume tiers above 25 mL were wrong
# ─────────────────────────────────────────────────────────────────────────────

def test_large_volumes_keep_one_millilitre_resolution():
    """
    Earlier revisions assumed 2 mL graduations for 50/60 mL syringes and 5 mL
    increments beyond 60 mL. Both were wrong: volumes past a single syringe are
    drawn in successive aliquots, so resolution plateaus at 1 mL.
    """
    for volume in (30.0, 50.0, 60.0, 100.0, 128.0, 500.0):
        assert get_vol_step_mL(volume * CONC_6, CONC_6) == 1.0


@pytest.mark.skipif(not REF_6MGML.exists(), reason="reference table not present")
def test_reference_table_evidences_one_millilitre_resolution_to_128_ml():
    """
    The empirical basis for the tier correction: in the NHS 6 mg/mL v2 table
    every band dose at or above 10 mL is a whole number of millilitres, the
    largest being 128 mL. Below 10 mL the finer graduations apply, so those
    rows are not evidence either way.
    """
    doses = [float(r["band_dose_mg"]) for r in csv.DictReader(REF_6MGML.open())]
    volumes = [d / CONC_6 for d in doses]

    large = [v for v in volumes if v >= 10.0]
    assert len(large) == 26
    assert max(large) == pytest.approx(128.0)
    for v in large:
        assert abs(v - round(v)) < 1e-9, f"{v} mL is not a whole millilitre"


# ─────────────────────────────────────────────────────────────────────────────
# PUBLISHED VALIDATION FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def test_primary_validation_20mgml_5_to_380mg():
    """Headline result: 44 bands, worst variance exactly 6.00%, no exceedance."""
    rows = build_bands(traditional(CONC_20, 5.0, 380.0), strict=True)
    assert len(rows) == 44
    assert verify_bands(rows, TAU, CONC_20) == []
    worst = max(
        max(abs(r["variance_below_pct"]), abs(r["variance_above_pct"])) for r in rows
    )
    assert worst == pytest.approx(6.0, abs=0.005)


def test_extended_validation_20mgml_5_to_1058mg():
    """The full extent of the NHS v7 reference table: 54 bands, still compliant."""
    rows = build_bands(traditional(CONC_20, 5.0, 1058.30), strict=True)
    assert len(rows) == 54
    assert verify_bands(rows, TAU, CONC_20) == []


@pytest.mark.skipif(not REF_6MGML.exists(), reason="reference table not present")
def test_out_of_sample_validation_6mgml():
    """
    Out-of-sample check against the NHS 6 mg/mL v2 table, which was not
    consulted while the algorithm or the volume framework was developed:
    the same band count, 60.6% exact agreement, and every NHS band dose inside
    the algorithm's tolerance.
    """
    ref = [
        (float(r["from_mg"]), float(r["to_a_mg"]), float(r["band_dose_mg"]))
        for r in csv.DictReader(REF_6MGML.open())
    ]
    rows = build_bands(
        traditional(CONC_6, ref[0][0], ref[-1][1], "Paclitaxel"), strict=True
    )

    assert len(ref) == 33
    assert len(rows) == 33
    assert verify_bands(rows, TAU, CONC_6) == []

    exact = 0
    for _, _, dose in ref:
        match = next(
            (r for r in rows if float(r["from_mg"]) <= dose < float(r["to_a_mg"])),
            None,
        )
        assert match is not None, f"NHS band dose {dose} mg is uncovered"
        algo = float(match["band_dose_mg"])
        assert abs(algo - dose) / dose <= TAU + 1e-9
        exact += abs(algo - dose) <= 0.05

    assert exact == 20
    assert exact / len(ref) == pytest.approx(0.606, abs=0.001)


def test_table_4_cerner_excerpt_boundaries():
    """
    The Cerner-format rows reproduced as Table 4. These are the values a
    reviewer checks by eye, and they all moved when boundaries went to 4 dp.
    """
    expected = {
        5.2: (5.0000, 5.5319),
        5.8: (5.5319, 6.1702),
        6.4: (6.1702, 6.8085),
        7.2: (6.8085, 7.6595),
        20.0: (19.1489, 21.2765),
        96.0: (93.6170, 102.1276),
        108.0: (102.1276, 114.8936),
        200.0: (195.7446, 212.7659),
        300.0: (297.8723, 319.1489),
        320.0: (319.1489, 340.4255),
    }
    rows = build_bands(traditional(CONC_20, 5.0, 380.0), strict=True)
    by_dose = {float(r["band_dose_mg"]): r for r in rows}

    for dose, (frm, to_a) in expected.items():
        row = by_dose.get(dose)
        assert row is not None, f"band {dose} mg is missing from the table"
        assert float(row["from_mg"]) == pytest.approx(frm, abs=1e-4)
        assert float(row["to_a_mg"]) == pytest.approx(to_a, abs=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# STRICT MODE IS THE DEFAULT
# ─────────────────────────────────────────────────────────────────────────────

def test_build_bands_is_strict_by_default():
    """
    Strict verification is opt-out, not opt-in. A caller that does nothing gets
    the guarantee; a caller that wants the old warn-and-display behaviour has
    to ask for it explicitly.
    """
    impossible = traditional(20.0, 0.2, 40.0)  # 0.01 mL minimum volume
    with pytest.raises(ValueError, match="cannot be banded"):
        build_bands(impossible)
    assert build_bands(impossible, strict=False)


def test_unknown_drug_type_is_rejected():
    bad = traditional(CONC_20, 5.0, 380.0)
    bad["drug_type"] = "biosimilar"
    with pytest.raises(ValueError, match="drug_type"):
        build_bands(bad)


# ─────────────────────────────────────────────────────────────────────────────
# VIAL OPTIMISATION (v2.0.0) MUST NOT WEAKEN THE GUARANTEE
# ─────────────────────────────────────────────────────────────────────────────

def test_vial_substitution_stays_inside_the_tolerance_window():
    """
    Vial optimisation may replace a band dose, but only with a value that is
    still within tau of every prescribed dose the band covers. The boundaries
    define the window; they are not moved to accommodate a vial.
    """
    cfg = traditional(20.0, 50.0, 800.0, "Vialled")
    rows = build_bands(cfg, vial_sizes=[100.0, 160.0], strict=True)

    for row in rows:
        D = float(row["band_dose_mg"])
        frm = float(row["from_mg"])
        to_a = float(row["to_a_mg"])
        assert (D - frm) / frm <= TAU + 1e-9
        assert (to_a - D) / to_a <= TAU + 1e-9


def test_vial_optimisation_does_not_change_the_bands_when_no_vials_given():
    """
    The manuscript's figures are generated without vial sizes. v2.0.0 must be
    band-for-band identical to the base algorithm on that path, otherwise the
    published tables would no longer describe the tagged code.
    """
    cfg = traditional(CONC_20, 5.0, 380.0)
    plain = build_bands(cfg, strict=True)
    empty = build_bands(cfg, vial_sizes=[], strict=True)
    assert [r["band_dose_mg"] for r in plain] == [r["band_dose_mg"] for r in empty]
    assert all(r["vial_combination"] == "" for r in plain)


# ─────────────────────────────────────────────────────────────────────────────
# VIAL-AWARE BAND PLACEMENT (v2.1.0)
# ─────────────────────────────────────────────────────────────────────────────

# Epirubicin 2 mg/mL, 50-260 mg, vials 10/50/200 — the configuration that
# exposed the limitation. Phase 2 alone cannot reach the single 200 mg vial.
EPIRUBICIN = (2.0, 50.0, 260.0)
EPI_VIALS = [10.0, 50.0, 200.0]


def test_substitution_window_lower_bound_collapses_to_the_band_dose():
    """
    Why vial-aware placement had to exist. `to_a_mg = D/(1-tau)`, so the
    Phase 2 window's lower bound `to_a_mg*(1-tau)` is D itself: substitution
    can only ever move a band dose UP. Any vial total below the greedy dose is
    unreachable however attractive — which is the whole defect.
    """
    conc, lo, hi = EPIRUBICIN
    rows = build_bands(traditional(conc, lo, hi), vial_sizes=EPI_VIALS, strict=True)

    for row in rows[:-1]:      # last band's to_a is max_dose, not D/(1-tau)
        D = float(row["band_dose_mg"])
        tol_low = float(row["to_a_mg"]) * (1 - TAU)
        assert tol_low == pytest.approx(D, abs=0.01)


def test_vial_aware_placement_reaches_a_vial_size_the_greedy_cannot():
    """
    The 200 mg vial. Greedy placement puts a band at 202 mg and then cannot
    substitute 200 mg because 200 < 202; vial-aware placement bands it at
    200 mg drawn from one vial, with no waste.
    """
    conc, lo, hi = EPIRUBICIN
    cfg = traditional(conc, lo, hi, "Epirubicin")

    greedy = build_bands(cfg, vial_sizes=EPI_VIALS, strict=True)
    aware = build_bands(cfg, vial_sizes=EPI_VIALS, strict=True, vial_aware=True)

    assert 200.0 not in [float(r["band_dose_mg"]) for r in greedy]

    band_200 = [r for r in aware if float(r["band_dose_mg"]) == 200.0]
    assert len(band_200) == 1
    assert band_200[0]["vial_combination"] == "1x200mg"
    assert float(band_200[0]["waste_mg"]) == 0.0


def test_vial_aware_placement_trades_band_count_for_zero_waste():
    """
    The trade the flag makes, pinned in both directions so neither side of it
    can regress unnoticed: more bands, far less waste.
    """
    conc, lo, hi = EPIRUBICIN
    cfg = traditional(conc, lo, hi, "Epirubicin")

    greedy = build_bands(cfg, vial_sizes=EPI_VIALS, strict=True)
    aware = build_bands(cfg, vial_sizes=EPI_VIALS, strict=True, vial_aware=True)

    zero = lambda rows: sum(1 for r in rows if float(r["waste_mg"]) == 0.0)
    waste = lambda rows: sum(float(r["waste_mg"]) for r in rows)

    assert len(aware) > len(greedy)
    assert zero(aware) > zero(greedy)
    assert waste(aware) < waste(greedy)


def test_vial_aware_is_off_by_default():
    """
    The flag changes published output, so it must never engage implicitly —
    a caller that does not ask for it gets the tagged v2.0.1 bands.
    """
    conc, lo, hi = EPIRUBICIN
    cfg = traditional(conc, lo, hi)
    default = build_bands(cfg, vial_sizes=EPI_VIALS, strict=True)
    explicit = build_bands(cfg, vial_sizes=EPI_VIALS, strict=True, vial_aware=False)
    assert default == explicit


def test_pooling_suppresses_vial_aware_placement():
    """
    Shared vials make the trade unprofitable — the residual is recovered for
    another patient, so the "saved" drug would largely have been used anyway,
    while the extra bands are real. A pooling site gets the greedy table.
    """
    conc, lo, hi = EPIRUBICIN
    cfg = traditional(conc, lo, hi)

    greedy = build_bands(cfg, vial_sizes=EPI_VIALS, vials_shared=True, strict=True)
    aware = build_bands(cfg, vial_sizes=EPI_VIALS, vials_shared=True,
                        strict=True, vial_aware=True)

    assert [r["band_dose_mg"] for r in aware] == [r["band_dose_mg"] for r in greedy]
    assert not vial_aware_applies(EPI_VIALS, vial_aware=True, vials_shared=True)
    assert vial_aware_applies(EPI_VIALS, vial_aware=True, vials_shared=False)
    assert not vial_aware_applies([], vial_aware=True, vials_shared=False)


def test_pooled_waste_cost_is_marked_indicative():
    """
    waste_mg, waste_pct and waste_cost must agree on whether a figure is real.
    An exact currency figure for drug that will in fact be reused overstates
    the saving — and the cost is the number a business case leans on.
    """
    conc, lo, hi = EPIRUBICIN
    cfg = traditional(conc, lo, hi)
    rows = build_bands(cfg, vial_sizes=EPI_VIALS, vials_shared=True,
                       cost_per_mg=3.20, strict=True)

    marked = 0
    for row in rows:
        indicative = str(row["waste_mg"]).startswith("~")
        assert str(row["waste_pct"]).startswith("~") == indicative
        assert str(row["waste_cost"]).startswith("~") == indicative
        if indicative:
            marked += 1
            waste = float(str(row["waste_mg"]).lstrip("~"))
            cost = float(str(row["waste_cost"]).lstrip("~"))
            assert cost == pytest.approx(waste * 3.20, abs=0.01)
    assert marked, "no band carried a non-zero waste figure to mark"

    # Unpooled, the same table quotes hard numbers.
    for row in build_bands(cfg, vial_sizes=EPI_VIALS, vials_shared=False,
                           cost_per_mg=3.20, strict=True):
        assert not str(row["waste_cost"]).startswith("~")


def test_vial_aware_without_vial_sizes_is_a_no_op():
    """There is nothing to place bands on, so the greedy table must come back."""
    cfg = traditional(CONC_20, 5.0, 380.0)
    plain = build_bands(cfg, strict=True)
    aware = build_bands(cfg, vial_sizes=[], strict=True, vial_aware=True)
    assert [r["band_dose_mg"] for r in plain] == [r["band_dose_mg"] for r in aware]


# ─────────────────────────────────────────────────────────────────────────────
# FEWEST VIALS FOR A GIVEN TOTAL
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("vial_aware", [False, True])
def test_a_total_is_prepared_from_the_fewest_vials(vial_aware):
    """
    With 10/30/60 mg vials, 60 mg must be reported as one 60 mg vial rather
    than two 30 mg vials, and 120 mg as two 60s rather than four 30s. Both the
    zero-waste path and the minimum-waste fallback tie-break on vial count.
    """
    cfg = traditional(2.0, 50.0, 260.0, "ThreeVials")
    rows = build_bands(cfg, vial_sizes=[10.0, 30.0, 60.0],
                       strict=True, vial_aware=vial_aware)

    by_dose = {float(r["band_dose_mg"]): r["vial_combination"] for r in rows}
    expected = {60.0: "1x60mg", 120.0: "2x60mg", 240.0: "4x60mg"}
    for dose, combo in expected.items():
        if dose in by_dose:
            assert by_dose[dose] == combo, f"{dose} mg prepared as {by_dose[dose]}"

    # Whatever the table, no reported combination may be beaten on vial count
    # by a different combination summing to the same total.
    from dose_banding import _count_vials, enumerate_vial_combinations

    combos = enumerate_vial_combinations([10.0, 30.0, 60.0], 300.0, 8)
    fewest = {}
    for total, label in combos:
        n = _count_vials(label)
        if total not in fewest or n < fewest[total]:
            fewest[total] = n
    for row in rows:
        label = row["vial_combination"]
        if not label:
            continue
        total = float(row["band_dose_mg"]) + float(row["waste_mg"])
        assert _count_vials(label) == fewest[round(total, 6)], (
            f"{label} is not the fewest-vial way to make {total} mg"
        )


# ─────────────────────────────────────────────────────────────────────────────
# RELEASE METADATA
# ─────────────────────────────────────────────────────────────────────────────

def test_packaged_version_matches_the_module_version():
    """
    pyproject and `__version__` must agree. A dependant pins the tag and reads
    the version back out of the module — the app stamps it into the header and
    the detail CSV — so a drift means the deployed app misreports which
    algorithm produced a published table. This went unnoticed through 2.0.1,
    which is why it is pinned here rather than left to release discipline.
    """
    import re
    from pathlib import Path

    import dose_banding

    pyproject = (Path(__file__).parent.parent / "pyproject.toml").read_text()
    declared = re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1)
    assert declared == dose_banding.__version__, (
        f"pyproject says {declared}, module says {dose_banding.__version__}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CORRECTION 4 — vial-aware placement produced bands beneath their own range
# ─────────────────────────────────────────────────────────────────────────────

def test_a_band_dose_is_never_below_its_own_range():
    """
    The walk originally accepted any vial total above the previous band dose.
    A total only slightly above it is admissible on tolerance alone, but the
    resulting band starts at D_prev/(1-tau) — above the dose itself — so the
    band is only as wide as the leftover, and the dose sits beneath the range
    it serves.

    Epirubicin on 10/50/200 mg vials produced a 2.13 mg-wide band at 80 mg
    directly after one at 78 mg: two bands no pharmacist could tell apart,
    which defeats the purpose of banding.
    """
    conc, lo, hi = EPIRUBICIN
    cfg = traditional(conc, lo, hi, "Epirubicin")

    for kwargs in ({}, {"vial_aware": True}):
        rows = build_bands(cfg, vial_sizes=EPI_VIALS, strict=True, **kwargs)
        for row in rows:
            D = float(row["band_dose_mg"])
            from_mg = float(row["from_mg"])
            assert D >= from_mg - 1e-9, (
                f"band {D} mg sits below its own range "
                f"{from_mg}-{row['to_a_mg']} (mode={kwargs or 'greedy'})"
            )


@pytest.mark.xfail(
    strict=True,
    reason="Known, unfixed: Phase 2 substitution searches from to_a_mg*(1-tau), "
           "which is below from_mg, so it can seat a band dose beneath its own "
           "range. Milder than the placement case — the boundaries are already "
           "fixed, so the band is not narrowed — but correcting it would change "
           "published v2.0.1 output for every vial user, so it is a decision "
           "rather than a bug fix. Remove this marker when that decision is made.",
)
def test_greedy_substitution_can_also_seat_a_dose_below_its_range():
    """
    Found by the property suite, not by hand: 1 mg/mL mab over 0.75-101.25 mg
    with a single 100 mg vial puts a 100 mg dose in a band starting at
    101.1111 mg. Recorded with its witness so the behaviour is not rediscovered
    from scratch, and so the suite tells us the day it changes.
    """
    cfg = {
        "drug_name": "Witness", "concentration_mg_per_ml": 1.0,
        "drug_type": "mab", "min_dose_mg": 0.75, "max_dose_mg": 101.25,
    }
    rows = build_bands(cfg, vial_sizes=[100.0], strict=True)

    for row in rows:
        assert float(row["band_dose_mg"]) >= float(row["from_mg"]) - 1e-9, (
            f"band {row['band_dose_mg']} mg below its range {row['from_mg']}"
        )


def test_the_near_duplicate_band_pair_is_gone():
    """The specific witness, pinned so the correction cannot silently regress."""
    conc, lo, hi = EPIRUBICIN
    rows = build_bands(traditional(conc, lo, hi), vial_sizes=EPI_VIALS,
                       strict=True, vial_aware=True)
    doses = [float(r["band_dose_mg"]) for r in rows]

    assert not (78.0 in doses and 80.0 in doses), (
        f"78 and 80 mg both present again: {doses}"
    )
    # The 3x50mg zero-waste band is what the mode exists for; keep it.
    assert 150.0 in doses
    combo_150 = [r for r in rows if float(r["band_dose_mg"]) == 150.0][0]
    assert combo_150["vial_combination"] == "3x50mg"
    assert float(combo_150["waste_mg"]) == 0.0


def test_no_two_bands_are_closer_than_a_useful_distance():
    """
    Consecutive band doses must differ by more than a rounding artefact.
    Bands separated by a couple of milligrams are a transcription hazard on a
    worksheet and give no clinical benefit over a single wider band.
    """
    conc, lo, hi = EPIRUBICIN
    rows = build_bands(traditional(conc, lo, hi), vial_sizes=EPI_VIALS,
                       strict=True, vial_aware=True)
    doses = [float(r["band_dose_mg"]) for r in rows]

    for lower, upper in zip(doses, doses[1:]):
        # Each band must at least span its own predecessor's tolerance reach.
        assert upper >= lower / (1 - TAU) - 1e-9, (
            f"bands {lower} and {upper} mg are closer than one tolerance step"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PLACEMENT DOES NOT DOMINATE SUBSTITUTION
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "conc,drug_type,lo,hi,vials",
    [
        (50.0, "mab", 75.0, 675.0, [100.0]),           # 440 mg vs greedy 400 mg
        (5.0,  "mab", 40.0,  80.0, [10.0, 50.0, 200.0]),   # 17 mg vs greedy 6 mg
    ],
)
def test_placement_can_waste_more_than_substitution(conc, drug_type, lo, hi, vials):
    """
    Both witnesses found by the property suite, not by hand.

    The two mechanisms are genuinely different — placement moves the bands,
    substitution keeps them and fits vials in afterwards — and neither wins
    everywhere. This matters beyond the algorithm: a front-end must not
    describe `vial_aware` as "least waste", and should show both figures so
    the choice is made on the numbers.

    If this test ever fails, placement has improved. That is good news, but
    the README claim and the app's comparison copy both need revisiting, so
    the suite should say so rather than let it pass unnoticed.
    """
    cfg = {
        "drug_name": "Witness", "concentration_mg_per_ml": conc,
        "drug_type": drug_type, "min_dose_mg": lo, "max_dose_mg": hi,
    }

    def waste(rows):
        return sum(float(str(r["waste_mg"]).lstrip("~") or 0) for r in rows)

    greedy = build_bands(cfg, vial_sizes=vials, strict=True)
    aware = build_bands(cfg, vial_sizes=vials, strict=True, vial_aware=True)

    assert waste(aware) > waste(greedy), (
        f"placement now matches or beats substitution here "
        f"({waste(aware)} vs {waste(greedy)} mg) — update the README and the "
        f"front-end copy, then retire this test"
    )
