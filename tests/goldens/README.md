# Goldens

Recorded outputs that pin DAWIS behaviour, plus `provenance.toml` recording
for each one: the commit it came from, the seed, the tolerances, and who
regenerated it, when and why.

Goldens live in plain git, no LFS. Revisit that only if they grow past
about 100 MB; Tier 1 and Tier 2 together are about 10 MB.

They are a ratchet, not a shrine. Through the 1.0 refactor no golden is
regenerated, because nothing should move. When later science work does move
one, it is regenerated in a commit whose message says why.

**They have been ratcheted once: at #105.** Through #62 they recorded the
published Ellien+2021 output from `legacy-seeded` at `1434f66`. #82 chose a
different reconstruction arm for DAWIS 1.0, #96 landed it one axis at a time,
and #103 was the first of those axes to move a stock run's output — so from
#103 to #105 the twelve golden comparisons were `xfail(strict = True)` rather
than red, and at #105 both tiers were re-recorded from `refactor/1.0` at
`bijaoui_rue.BSPL.TRANSPOSE.as_written.HS.coarse`.

What a ratchet cannot do is guard itself: the suite the goldens gate is the
suite that recorded them, so a port error becomes the recorded truth rather
than a failure. What authorises this one is therefore evidence taken outside
the suite — every axis proved bit-exact against `restoration_tests/arms/` or
against #93's seam before it landed — and the whole chain, with what moved and
by how much, is in `.research/105-goldens-ratchet/`.

## `tier1_cutout/` — the per-commit gate (#36)

One seeded run on the agreed 256x256 patch of the gallery image, recorded from
`refactor/1.0` at `5978cc7` at the arm #82 chose. About 24 s, 19 iterations,
85 atoms.

Through #62 this was the published Ellien+2021 algorithm with the RNG pinned
and the drift reverted (#32), recorded from `legacy-seeded` at `1434f66`:
40 s, 20 iterations, 76 atoms, of which 43 took `SUM` and 33 `ADJOINT`. #105
ratcheted it.

| File | What it pins |
|---|---|
| `input_cutout.fits` | the exact patch the run was fed |
| `restored.fits` | the sum of every atom |
| `residuals.fits` | what the loop left behind |
| `atoms.csv` | one row per atom: position, scale, operator, flux |
| `convergence.csv` | one row per greedy iteration |
| `provenance.toml` | commit, seed, thread count, dependency versions, tolerances, pinned `max_iter`, author, date |

`input_cutout.fits` is a golden in its own right: a misread cutout coordinate
still lands on a valid patch, so comparing the input first turns a silent
difference into a loud failure. `provenance.toml` states the convention in
full.

What a Tier 1 run *is* lives in `tests/tier1.py`, shared by the recording
script and by the comparison in `tests/regression/test_golden_cutout.py`.

## `tier2_gallery/` — the checkpoint science check (#37)

One seeded run on the **whole** 900x900 gallery image at the **shipping** CG
`max_iter=200`, recorded from the same `refactor/1.0` commit `71c0a4d` as
Tier 1. 643 s, 16 iterations, 449 atoms.

Through #62 this was the published algorithm from `legacy-seeded` at
`1434f66`: 1708 s, 27 iterations, 463 atoms, of which 322 took `SUM` and 141
`ADJOINT`. #105 ratcheted it.

Tier 1 asks whether the carve broke the structure of the pipeline; Tier 2 asks
whether the pipeline still does the science it is meant to, at the solver
setting 1.0 actually ships. *Which* science that is moved at #105; the solver
setting did not move with it. Every Tier 2 test carries the `slow` marker and is excluded
from the default run, so it costs nothing per commit. It runs at named carve
checkpoints — after each subpackage is fully carved out — and nightly in CI
(#19).

```bash
pytest -m slow tests/regression/test_golden_gallery.py
```

| File | What it pins |
|---|---|
| `restored.fits` | the sum of every atom |
| `residuals.fits` | what the loop left behind |
| `atoms.csv` | one row per atom: position, scale, operator, flux |
| `convergence.csv` | one row per greedy iteration |
| `provenance.toml` | commit, seed, thread count, dependency versions, tolerances, shipping `max_iter`, measured runtime |

There is no `input_gallery.fits`. The image ships inside the package and is
already under version control, so `provenance.toml` records its sha256 and the
comparison checks that digest first — the same loud-failure guarantee Tier 1
buys with a recorded copy, without a second 3 MB of FITS.

### Budget

`[budget]` in `provenance.toml` carries the measured numbers, so checkpoint
scheduling is planned from a measurement rather than a guess. Budget **about
fifteen minutes** per checkpoint run on a 2017 laptop — 643 s measured at
#105, against 1708 s before the ratchet. #37 expected this tier to exceed #5's
3539 s baseline; it does not, because #5 measured the drifted Wolfe restoration
path with animation frames written inside the loop.

Do not read the 1708 → 643 s drop as what the new arm costs. It confounds
three changes at once: the arm, the shorter greedy loop (27 iterations to 16),
and `deconv` — the old recording came from `legacy-seeded`, where the `ADJOINT`
branch ran ten Richardson-Lucy steps per atom on top of the synthesis for 141
of its 463 atoms, and #101 removed that. #33's 1167.6 s is not a clean baseline
either: it ran seed 0 for 394 atoms, and recording additionally reads every
atom pickle back off disk. What the new default costs a full run is #106's
question.

## Shared machinery

| Module | What it owns |
|---|---|
| `tests/goldenrun.py` | what a golden run *is* — the published parameters, the pinned solver knob, how a run is driven and read back off disk |
| `tests/goldencompare.py` | how a golden is compared — #3's rule, one copy, both tiers |
| `tests/tier1.py`, `tests/tier2.py` | only what each tier changes: the input and `CG_MAX_ITER` |
| `tests/goldens/recording.py` | what the two recording scripts share |

Tolerances are deliberately *not* shared: each tier reads its own out of its
`provenance.toml`, so a golden cannot be recorded under one rule and compared
under another.

## What provenance does not record

`provenance-audit.md` is the standing answer, written for #62 after the thread
count turned up by accident: every field a golden records and whether it can
move output, every candidate input that is *not* recorded, and which of those
were measured to move output rather than assumed to.

Three move output. One was acted on and two were weighed and left, and Part 3
of the audit records that decision with its reasoning — so a later reader does
not re-derive the finding and assume it was missed.

The one acted on is `[dependencies]`. A golden used to record `numpy` and no
other package, while astropy sets the noise sigma behind every detection
threshold, photutils produces the segmentation, scikit-image supplies the
region morphology that picks an atom's restoration operator, and scipy clips
on the simple detector path. Those four move an **atom count**, not an atom's
last digits: on the Tier 1 cutout a noise sigma 1e-4 higher restores 74 atoms
in 19 iterations rather than 76 in 20, while the same sigma one float32 ulp
higher changes nothing at all.

Both tiers were re-recorded to carry the table, and **neither golden moved**:
`atoms.csv`, `convergence.csv`, `residuals.fits`, `restored.fits` and Tier 1's
`input_cutout.fits` are byte-identical to their previous recordings, and
`provenance.toml` is the only file that changed in either. Same 76 atoms in 20
iterations at Tier 1, same 463 in 27 at Tier 2. Every tolerance, including the
residual sigma each is derived from, came out unchanged.

`audit_environment.py` prints every candidate off a machine, so the audit is
re-run at the next release rather than re-invented. It shares its package list
with `recording.py`, so the audit cannot drift from what is recorded.

```bash
python tests/goldens/audit_environment.py --pin-threads 1
```

## Threads

Every golden is recorded and compared at `goldenrun.GOLDEN_N_THREADS` maths-library
threads — one. This is not a speed setting. numpy splits a single array
operation across threads on its own, and a sum split over four threads is added
in a different association order from one added on a single thread; on the Tier 1
cutout that moves an atom's flux in its last digit. Pinning it is what leaves the
*input* as the only thing a golden depends on.

Until #61 the pin was four `os.environ` calls above the numpy import in
`tests/conftest.py`, because BLAS reads its thread count once, when numpy loads
it, and nothing inside the package could act that late. DAWIS owns the knob now:
`Config.n_threads` carries it and `dawis.parallel.parallel_backend` applies it
with `threadpoolctl` to the already-loaded library. A golden run therefore pins
its own threads, through the same code path a user runs, and `conftest` asserts
the pin actually lands rather than hoping an environment variable did.

The `legacy` trees predate that knob, so the recording scripts pin them from the
harness instead (`goldenrun.pin_threads`) — which is what makes a recording
comparable against a 1.0 run at all.

Each `provenance.toml` records two separate facts: `[run].n_threads`, what the
run asked for, and `[machine].effective_n_threads` with `[machine].blas_vendor`,
what the loaded library actually gave. They can disagree, and a knob that
silently fails is the bug #61 exists to make visible.

Both goldens were re-recorded for #61, and both moved — they had been recorded
unpinned and compared at one thread, and nobody knew how much of the tolerance
that consumed. On Tier 2, 183 of 2778 measured values moved in their last digit
and **no decision moved at all**: same 463 atoms, same 27 iterations, same
levels, bounding boxes, filters and convergence flags. That is the signature the
thread count is supposed to have, and it is now recorded rather than assumed.

The vendor is not fixed across machines and does not need to be. The recordings
report `mkl`; the ray-bearing environment the Tier 2 suite runs in loads
`openblas`, pins just as well, and passes every golden comparison within the
recorded tolerances.

## Regenerating

Only in a commit whose message says why the golden moved.

```bash
python tests/goldens/record_cutout_golden.py \
    --source ~/dawis --goldens /tmp/g1 \
    --reason "why this golden moved"

python tests/goldens/record_gallery_golden.py \
    --source ~/dawis --goldens /tmp/g2 \
    --reason "why this golden moved"
```

`--source` must be a clean checkout on `recording.GOLDEN_BRANCH` — since #105
that is `refactor/1.0`, and through #62 it was `legacy-seeded`. The scripts
refuse anything else unless you pass `--allow-any-branch`. They stub `ray` and
`ultranest` if they are not installed — neither is reached on the serial path.

Record into scratch directories and copy the results in afterwards, as above,
whenever `--source` is this repository rather than a separate checkout.
Writing a tier's goldens straight into the tree makes it dirty, and the next
tier then refuses to record at all — correctly, since a golden's provenance
commit has to describe what produced it. Scratch directories are what let one
commit stand behind both tiers.
