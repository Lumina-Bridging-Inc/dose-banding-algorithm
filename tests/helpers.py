"""
Shared strategies and independent re-derivations used by the test suite.

Nothing in this module may call into `dose_banding` to decide what the correct
answer is — the checks have to be derived from the specification, otherwise the
tests only prove the code agrees with itself.

THE PRECONDITION HAS TWO PARTS
------------------------------
The tolerance guarantee is attainable only when the graduation step is small
enough relative to the dose. That resolves into two separate conditions, and
the stricter of the two is the one that binds:

  (a) ADVANCE.  Moving from one band to the next needs a dispensable dose in
      (D_prev, D_prev*(1+tau)/(1-tau)], an interval of relative width
      2*tau/(1-tau). Sufficient when

          volume >= step * (1 - tau) / (2 * tau)          ~ 0.078 mL at tau=6%

  (b) FIRST BAND.  The opening dose must be dispensable AND lie in
      [min_dose, min_dose*(1+tau)] — an interval of relative width only tau,
      roughly half the width available to (a). Sufficient when

          volume >= step / tau                            ~ 0.167 mL at tau=6%

Condition (b) is stronger than (a) by a factor of 2/(1-tau), and it is the one
that governs a table's minimum dose. It also bites at the *start of each tier*,
where the graduation coarsens but the volume has not yet grown to match: at
tau=6% the exposed windows are [1.0, 1.667), [3.0, 3.333) and [10, 16.667) mL.
Both bounds are sufficient, not necessary — a minimum dose that happens to sit
on the graduation grid is fine below either floor.
"""

from typing import Optional

from hypothesis import strategies as st

from dose_banding import VARIANCE, VOLUME_PRECISION_TIERS

# Concentrations in routine clinical use, including the two that the manuscript
# validates against (20 mg/mL NHS v7, 6 mg/mL NHS v2) and 25 mg/mL, which is
# where 2 dp boundary flooring was shown to break the tolerance guarantee.
CONCENTRATIONS = [1.0, 2.0, 3.0, 5.0, 6.0, 10.0, 20.0, 25.0, 50.0, 100.0]

DRUG_TYPES = sorted(VARIANCE)

# (lower_volume_inclusive, upper_volume_exclusive, graduation_step)
TIERS = [
    (lo, hi, step)
    for lo, (hi, step) in zip(
        [0.0] + [t[0] for t in VOLUME_PRECISION_TIERS[:-1]], VOLUME_PRECISION_TIERS
    )
]


def vol_step_for(volume_mL: float) -> float:
    """Graduation step for a volume — re-derived from the tier table."""
    for upper, step in VOLUME_PRECISION_TIERS:
        if volume_mL < upper:
            return step
    return VOLUME_PRECISION_TIERS[-1][1]


def min_advance_volume(tau: float, vol_step: float) -> float:
    """Condition (a): smallest volume at which a next band always exists."""
    return vol_step * (1.0 - tau) / (2.0 * tau)


def min_first_band_volume(tau: float, vol_step: float) -> float:
    """Condition (b): smallest volume at which an opening band always exists."""
    return vol_step / tau


def config(conc: float, dtype: str, min_dose: float, max_dose: float) -> dict:
    return {
        "drug_name": "Test",
        "concentration_mg_per_ml": conc,
        "drug_type": dtype,
        "min_dose_mg": min_dose,
        "max_dose_mg": max_dose,
    }


@st.composite
def well_posed_config(draw, max_ratio: float = 200.0):
    """
    Drug configurations satisfying both parts of the precondition.

    The minimum volume is drawn tier by tier, from the part of each tier that
    clears that tier's first-band floor, so the exposed windows at the bottom
    of the coarser tiers are excluded here. They are covered deliberately by
    `test_first_band_window_at_the_bottom_of_a_tier` instead.
    """
    conc = draw(st.sampled_from(CONCENTRATIONS))
    dtype = draw(st.sampled_from(DRUG_TYPES))
    tau = VARIANCE[dtype]

    lo, hi, step = draw(st.sampled_from(TIERS))
    floor = max(lo, min_first_band_volume(tau, step)) * 1.02
    ceiling = min(hi, 40.0)
    if floor >= ceiling:
        floor, ceiling = min_first_band_volume(tau, step) * 1.02, 40.0

    min_vol = draw(
        st.floats(
            min_value=floor,
            max_value=ceiling,
            allow_nan=False,
            allow_infinity=False,
            exclude_max=True,
        )
    )
    ratio = draw(
        st.floats(
            min_value=1.5,
            max_value=max_ratio,
            allow_nan=False,
            allow_infinity=False,
        )
    )

    min_dose = round(min_vol * conc, 4)
    max_dose = round(min_dose * ratio, 4)
    return config(conc, dtype, min_dose, max_dose)


@st.composite
def any_config(draw):
    """
    Arbitrary configurations, including ones that cannot be banded at all.

    Used for the soundness property: whatever it is handed, strict mode must
    either refuse or return a correct table.
    """
    conc = draw(st.sampled_from(CONCENTRATIONS))
    dtype = draw(st.sampled_from(DRUG_TYPES))
    min_dose = draw(
        st.floats(
            min_value=0.05, max_value=500.0, allow_nan=False, allow_infinity=False
        )
    )
    ratio = draw(
        st.floats(min_value=1.1, max_value=100.0, allow_nan=False, allow_infinity=False)
    )
    return config(conc, dtype, round(min_dose, 4), round(min_dose * ratio, 4))


def band_for_dose(rows: list[dict], dose: float) -> Optional[dict]:
    """
    Resolve a prescribed dose to its band using the *published* boundaries and
    the NHS half-open convention: [from_mg, to_a_mg), with the final band
    inclusive of its upper boundary.

    Returns None if the dose falls outside the table. Raises if it matches more
    than one band, which would itself be a defect.
    """
    hits = []
    for i, row in enumerate(rows):
        frm = float(row["from_mg"])
        to_a = float(row["to_a_mg"])
        last = i == len(rows) - 1
        inside = frm <= dose <= to_a if last else frm <= dose < to_a
        if inside:
            hits.append(row)
    if len(hits) > 1:
        raise AssertionError(
            f"dose {dose} matched {len(hits)} bands: "
            f"{[(r['from_mg'], r['to_a_mg']) for r in hits]}"
        )
    return hits[0] if hits else None
