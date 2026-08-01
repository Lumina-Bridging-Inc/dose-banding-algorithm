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
