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

If you have come from the paper and want to check its numbers, you do not need
this section — skip to [Reproducing the published
results](#reproducing-the-published-results), which needs no install at all.

To use the algorithm as a dependency:

```bash
pip install git+https://github.com/Lumina-Bridging-Inc/dose-banding-algorithm@v2.1.1
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

### Vial-aware band placement

Substitution alone is weak, because it runs after the band positions are
fixed. Since `to_a_mg = D/(1-τ)`, the window it searches starts at `D` itself,
so it can only ever move a band dose *up*, into a sliver left over from
rounding. A vial total just below the band dose is unreachable — with
10/50/200 mg vials the 202 mg band cannot take the single 200 mg vial.

`vial_aware=True` places the bands on whole-vial totals instead:

```python
rows = build_bands(drug, vial_sizes=[10.0, 50.0, 200.0], vial_aware=True)
```

For epirubicin 2 mg/mL over 50–260 mg that is 16 of 19 bands zero-waste
(14 mg total waste) against 6 of 15 (56 mg) — including a 200 mg band drawn
from one vial and a 150 mg band from three 50s, neither of which substitution
can reach. The cost is table size: a vial total below the widest admissible
dose narrows the band, so more bands are needed. Callers who want the
minimum-band table (P4) should leave the flag off.

**It is not guaranteed to beat the default on waste.** Placement moves the
bands; substitution keeps them and fits vials in afterwards. Neither dominates.
Across 336 clinical configurations placement was worse on total waste in 1 and
worse on zero-waste count in 13 — so present both figures to the user rather
than assuming the flag is an improvement.

A placed band dose always sits inside its own band. Allowing one beneath its
range is admissible on tolerance alone but produces a band only as wide as the
leftover: 2.1.0 emitted a 2 mg-wide band at 80 mg directly after one at 78 mg.
Corrected in 2.1.1.

It is opt-in because it changes published output, and it is suppressed when
`vials_shared=True` — pooling recovers the residual for another patient, so
the saving is notional while the extra bands are real. Call
`vial_aware_applies()` to report the override to a user.

As a command line tool:

```bash
dose-banding --template            # write drugs_input_template.csv
dose-banding drugs_input.csv       # generate tables into output/
dose-banding drugs_input.csv --validate   # + NHS reference cross-check
```

## Reproducing the published results

This is the reproducibility artefact for the manuscript: clone, run one
command, obtain every published number. Nothing needs to be installed and no
Python knowledge is required — the algorithm uses only the standard library,
and the whole run takes about two seconds.

**1. Check for Python 3.9 or later.** In a terminal (Terminal on macOS,
PowerShell on Windows):

```bash
python3 --version
```

If that reports anything below 3.9, or `command not found`, install Python from
[python.org/downloads](https://www.python.org/downloads/) — there are macOS and
Windows installers; on Linux use your distribution's package manager. On
Windows, type `python` wherever this section says `python3`.

**2. Get the code and run it.**

```bash
git clone https://github.com/Lumina-Bridging-Inc/dose-banding-algorithm
cd dose-banding-algorithm
python3 validate_paper.py
```

Without `git`, download the repository from its GitHub page instead — the green
**Code** button, then **Download ZIP** — unpack it, `cd` into the unpacked
folder, and run the third command there.

**3. Check it worked.** The output opens with the paper's headline table:

```
  5-380mg: 5–380 mg @ 20 mg/mL, ±6%
==========================================================================
  Algorithm bands ............ 44
  Max boundary variance ...... 6.00%   (limit ±6%)
  Bands exceeding limit ...... 0
  Coverage gaps .............. 0
  All volumes on-graduation .. yes
```

What follows regenerates each remaining table in turn — the band-by-band
comparison against the NHS 20 mg/mL reference, the reference bands that exceed
±6%, the Cerner CPOE excerpt, the comparison across the reference table's full
published extent, and the out-of-sample 6 mg/mL validation. Every figure quoted
in the manuscript appears in that output, and the tables behind it are written
as CSVs to `output/paper/`.

### Which version to run

Run `main`, as above. The manuscript describes the algorithm as tagged `v1.0`,
and band generation is unchanged since — the later releases add whole-vial
optimisation, which is inert when no vial sizes are supplied, and a regression
test asserts the outputs are identical. But the validation script itself has
grown: `v1.0` predates the comparison across the reference table's full
published extent, so checking out that tag reproduces most of the paper's
figures and not all of them.

## Tests

Unlike the validation script, the tests do need two libraries installed, so
work in a virtual environment to keep them out of your system Python:

```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
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
| `v2.1.0` | Adds opt-in `vial_aware` band placement. Default output is unchanged; `waste_cost` now carries the same `~` indicative marker as `waste_mg` when vials are pooled. |
| `v2.1.1` | `vial_aware` placement now requires each band dose to sit inside its own band. Fixes near-duplicate bands (78 and 80 mg on epirubicin). Affects `vial_aware=True` output only. |

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
