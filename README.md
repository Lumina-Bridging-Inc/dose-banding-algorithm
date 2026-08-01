# Dose Banding Algorithm

Generates standardised chemotherapy dose bands with a guaranteed tolerance
bound, in a format that imports directly into a CPOE standardised dose range
table.

This repository is the algorithm and its validation — nothing else. It is the
codebase cited by the accompanying manuscript. A separate repository holds a
Streamlit web interface that calls this package; that interface is an
accessibility layer and forms no part of the method described in the paper.

## What it guarantees

For every configuration it accepts, the generated table satisfies three
properties, verified exactly before the table is returned:

1. **Dispensability** — every band dose is a whole number of syringe
   graduations for its own volume tier, so the dose can actually be drawn up.
2. **Tolerance** — every prescribed dose in range receives a band dose within
   ±6% (traditional cytotoxics) or ±10% (monoclonal antibodies) of it, judged
   at the *published* boundaries rather than the underlying reals.
3. **Coverage** — the bands tile the requested range with no gap and no
   overlap, so each prescribed dose maps to exactly one band.

Band count is minimal: at each step the algorithm takes the widest band that
keeps coverage gap-free, and no larger step exists.

Where the guarantee cannot be met, `build_bands` **raises** rather than
emitting a table that breaches tolerance. Pass `strict=False` to get the table
anyway, with the offending rows flagged `within_tolerance = False`.

### When it refuses

The guarantee needs the syringe graduation to be small relative to the dose.
Two conditions apply, both expressed as a floor on volume:

| | condition | floor at ±6% | floor at ±10% |
|---|---|---|---|
| Opening band | `volume ≥ step / τ` | 16.7 × step | 10 × step |
| Each advance | `volume ≥ step (1−τ) / 2τ` | 7.8 × step | 4.5 × step |

The opening band is the stricter of the two and is what governs a table's
minimum dose — it needs a dispensable dose inside `[min_dose, min_dose(1+τ)]`,
an interval only τ wide, against the 2τ/(1−τ) available to every later band.

Because the graduation coarsens at 1, 3 and 10 mL, the opening condition
leaves exposed windows just above each of those edges — at ±6%, minimum
volumes in `[1.0, 1.67)`, `[3.0, 3.33)` and `[10, 16.67)` mL. A minimum dose
landing on the graduation grid still works there; one that falls between
graduations cannot, and is refused. Raise the minimum dose, use a lower
concentration, or widen the tolerance.

Both floors are sufficient conditions, not necessary ones.

## Install

```bash
pip install git+https://github.com/Lumina-Bridging-Inc/dose-banding-algorithm@v2.0.1
```

Pure standard library — no runtime dependencies. Python 3.9+.

## Use

As a library:

```python
from dose_banding import build_bands

rows = build_bands({
    "drug_name": "Paclitaxel",
    "concentration_mg_per_ml": 6.0,
    "drug_type": "traditional",   # or "mab"
    "min_dose_mg": 27.58,
    "max_dose_mg": 817.41,
})
```

Optionally pass `vial_sizes=[100.0, 160.0]` to substitute whole-vial
combinations where one fits inside the band's tolerance window. The boundaries
are not moved to accommodate a vial.

As a command line tool:

```bash
dose-banding --template            # write drugs_input_template.csv
dose-banding drugs_input.csv       # generate tables into output/
dose-banding drugs_input.csv --validate   # + NHS reference cross-check
```

## Reproducing the published results

```bash
python validate_paper.py
```

Regenerates every table reported in the manuscript and prints the figures
quoted in the text, writing CSVs to `output/paper/`. This is the
reproducibility artefact: clone, run one command, obtain every published
number.

## Tests

```bash
pip install -e ".[test]"
pytest                              # fast, ~250 examples per property
pytest --hypothesis-profile=deep    # 5000 examples per property
```

`tests/test_properties.py` verifies the guarantees above against randomly
generated configurations rather than fixed examples — it stands in place of a
formal proof, and each test names the property it establishes.
`tests/test_regressions.py` pins the published validation figures and the
defects corrected during development, each with the input that exposed it.

Run the deep profile before tagging a release.

## Versions

| Tag | |
|---|---|
| `v1.0` | The base algorithm as described in the manuscript. |
| `v2.0.0` | Adds whole-vial optimisation, corrected volume tiers above 25 mL, 4 dp boundaries, and strict verification. |
| `v2.0.1` | No change to band generation. Adds packaging and the verification suite — the earliest version that can be installed, and so pinned, as a dependency. |

Band output is identical between the two when no vial sizes are supplied, so
v2.0.0 reproduces every figure in the paper; a regression test asserts this.
Generated tables carry `algorithm_version` so a downloaded CSV can always be
traced to the code that produced it.

## Validation

The algorithm was developed against the NHS England 20 mg/mL Version 7
reference table and validated out-of-sample against the 6 mg/mL Version 2
table, which was not consulted during development.

| | 20 mg/mL, 5–380 mg | 6 mg/mL, 27.58–817.41 mg |
|---|---|---|
| Algorithm bands | 44 | 33 (same as reference) |
| Worst boundary variance | 6.00% | 6.00% |
| Bands exceeding ±6% | 0 | 0 |
| Exact agreement with reference | 68.8% | 60.6% |
| Reference bands exceeding ±6% | 6 | 9 |

Divergences from the reference are the algorithm choosing a different
graduation-aligned dose, not a tolerance failure: every reference band dose
falls within the algorithm's tolerance.

## Disclosures

Developed without funding, on personal time. This repository contains no
sponsorship or advertising mechanism of any kind.

Development was assisted by a large language model (Anthropic Claude). All
output was reviewed by the author; defects found in that review are recorded
as regression tests in `tests/test_regressions.py`, each named for the
condition it corrects.

Not a medical device. Any table generated here must be reviewed and approved
by a qualified pharmacist against local policy before clinical use.

## Licence

MIT — see [LICENSE](LICENSE).
