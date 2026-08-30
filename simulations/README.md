# Simulation gallery

Synthetic FITS images with exact ground truth, for the restoration test
battery of #25. Hand-run, not a pytest tier, not CI. Spec: #67. This directory
was built by #68; #69 added the generator's full command line; #70 added the
`point` and `sersic_n1_halo` classes; #71 added `blend` and with it the last
of the four; #72 added the other two noise regimes and the other two
realisations, and with them the whole 72-entry grid.

DAWIS has one image it can run on today, and nobody knows the right answer for
it. Every claim about over-subtraction, flux conservation or shape recovery is
therefore an eyeball judgement. The gallery closes that gap: a researcher
points DAWIS at one of these images and can ask "is this **atom** the right
flux and the right shape?" and get a number.

GalSim never enters DAWIS's dependency set. The generator is a standalone tool
in its own conda environment. The images it produces are inert FITS files, and
DAWIS reads them exactly the way it reads the packaged gallery image.

## Setup

```
conda env create -f simulations/environment.yml
```

That creates `dawis-sim`, with GalSim pinned to an exact conda-forge version.
DAWIS itself never runs in this environment, and this environment never
imports DAWIS.

Python is pinned too, because the GalSim build depends on it. numpy and
astropy are not: #67 asks for the numpy version to be *recorded* per entry,
and `manifest.json` records the one that actually ran.

The pin is deliberate and is not routine maintenance. The images are ground
truth: a GalSim upgrade would silently change them, and every number already
recorded against them would become uncomparable with no signal that it had
happened. Changing the pin means regenerating the gallery.

## Generating

One invocation, one entry:

```
conda run -n dawis-sim python simulations/generate.py \
    --tier full --class sersic_n4 --regime balanced --realisation 0
```

Or every entry the recipe knows how to build:

```
conda run -n dawis-sim python simulations/generate.py --all
```

Without `--all`, all four axes are required and name exactly one entry. With
`--all`, any axis you give is held fixed and the rest are swept, so
`--all --tier small` builds every known entry of the cheap tier.

`--out` defaults to `~/dawis-data/simulations` and stays overridable, so
generating into a scratch location for inspection needs no code change.

### Resuming, and staleness

Generation is idempotent. An entry whose directory already holds a
`manifest.json` **matching the recipe** is skipped, so an interrupted `--all`
resumes where it stopped rather than starting over. `--force` rebuilds one
deliberately.

Matching, not merely present. The comparison covers every manifest key except
the three provenance ones — the GalSim version, the numpy version and the
source commit — because those record what drew the pixels rather than what the
recipe decided, and an upgrade alone moves no ground truth. It lives on the
recipe, as `Recipe.matches_manifest`, because that is the only object it reads.

An entry whose manifest describes a *different* recipe — or is unreadable —
is **stale**, and the generator stops rather than skipping it or overwriting
it:

```
generate.py: .../sersic_n4_balanced_r0 holds a manifest that does not
describe this entry: it is unreadable, or it was written by a recipe that has
since changed. Rebuild it with --force, or move it aside to keep it.
```

Skipping would leave last month's ground truth on disk under a name that says
it is this month's, which is exactly the silent drift #67 stores the gallery
to prevent. Overwriting would be a rebuild nobody asked for, and a researcher
may already have numbers recorded against those images. Both need a human, so
both are refused.

An entry interrupted *before* its manifest is written counts as absent, not
stale — the manifest goes last for that reason — so a killed run rebuilds only
the entry it was in the middle of.

## What one entry holds

```
<data-root>/simulations/<tier>/<class>_<regime>_r<N>/
    image.fits           # the simulated observation; what DAWIS reads
    truth_preconv.fits   # sources only, no PSF, no noise
    truth_conv.fits      # sources convolved with the PSF, no noise
    psf.fits             # the exact kernel applied
    sources.csv          # one row per source
    manifest.json        # provenance and parameters
```

This is a stated contract. The metric half of #25 is written against these
names.

`sources.csv` carries nine columns for the isolated classes and twelve for
`blend`, which adds the companion id, the separation and the flux ratio #67
lists for it. Those three describe a *pair*, and a class of isolated sources
has none: carrying them everywhere would put three columns of zeros in every
other entry and invite a reader to take "separation 0" for a measured
separation rather than for an absence. `manifest.json` is conditional the same
way and for the same reason — the blend keys appear on blend entries only, so
adding the class made no entry already on disk stale.

`image.fits` says nothing about the noise, in its header or anywhere else.
DAWIS estimates the noise itself, as it would on real data. The true sky
level, read noise and gain are in `manifest.json`, for a human reading a
result — never for a run to pick up.

The sky is **left in** `image.fits`, not subtracted back out after the noise is
drawn. #67 anchors the balanced regime to the gallery image's median *and* its
standard deviation; subtracting would keep the second and throw away the
first, and would leave the manifest's `sky_level` describing a level no longer
in the pixels — which is the comparison it is recorded for.

### The three noise regimes

| regime | sky level (ADU) | read noise (e⁻) | gain (e⁻/ADU) | read share of variance |
|---|---|---|---|---|
| `read_noise` | 0.00216 | 1.735 | 155 | 90% |
| `balanced` | 0.0108 | 1.293 | 155 | 50% |
| `photon` | 0.0194 | 0.578 | 155 | 10% |

`balanced` is the anchored one: its sky level is the packaged gallery image's
median and its σ is that image's standard deviation, so the numbers correspond
to a real observation. The other two are **displacements of it**, computed
from it rather than written down beside it, so a correction to the anchor
carries the whole axis with it.

All three hold the background σ and the gain fixed at the balanced regime's
values and move only the *split* of the variance between the photon and read
terms. That is deliberate, and it is #72's reading of #67 rather than a number
#67 states.

A source's flux is set from a peak SNR, and an SNR is measured against σ. The
grid geometry survives a change of depth — the wing radius is a ratio of
profile to background, and the spacing follows the profile — but every flux
and every magnitude scales with it, and #67 asks for residuals binned by
brightness. A magnitude bin holding a different source in each regime would
not be one bin.

Depth is not lost by holding it. The ladder already spans peak SNR 3 to 1000
*within* every image, which is a wider range of depth than three regimes could
carry between them: depth is the grid's column axis, not this one. What the
regime axis adds, and all it adds, is which noise source that depth is made of
— which is what DAWIS's own noise estimator has to cope with, and #67 puts
that estimator inside the measurement on purpose.

Nine parts to one, rather than a term switched off entirely: a regime with no
photon noise at all would be a different noise model, not a regime DAWIS
claims to handle. The two flanking regimes are points in camera-parameter
space rather than two particular cameras, and their read noise is not to be
read off against a detector datasheet — the gain is 155 because the anchored
image is in calibrated units rather than raw counts, so an "electron" here is
a unit of that calibration and not a photoelectron.

## What is built

**All 72 entries.** #68 was the tracer bullet: one narrow path through all
three layers, for `sersic_n4` / `balanced` / `r0`. #70 added the other two
**isolated** classes, `point` and `sersic_n1_halo`, #71 added `blend`, and #72
added the two remaining noise regimes and the two remaining realisations.
`--all` now sweeps the whole grid #67 specifies: 2 tiers × 4 classes × 3
regimes × 3 realisations.

A value that is not on an axis at all is still **refused**, not guessed.
Asking for `r3`, or for a fifth class, raises `AxisNotBuiltError` with a
sentence saying so rather than a `KeyError` from the middle of a layout. #25
already names two classes beyond #67's four, and #67 wants the realisation
count raisable, so the next undecided value is a matter of when rather than
whether.

### The three isolated classes

| class | profile | half-light radius | truncation | bright-end magnitude |
|---|---|---|---|---|
| `sersic_n4` | de Vaucouleurs, sheared | 0.30" | 8 r_e | 18.1 |
| `point` | delta function | — | — | 20.4 |
| `sersic_n1_halo` | exponential, sheared | 0.50" | 6 r_e | 16.1 |

They share their whole layout and differ only in `SOURCE_PROFILES`: the same
grid, the same jitter, the same peak-SNR ladder from roughly 1000 down to 3,
on the same two tiers. #70 therefore added source construction, not grid
logic.

The **same** SNR range for all three is the point of the design, not an
oversight in the halo's case. What makes the halo the ICL-like
under-restoration case is its *surface* brightness: at a matched peak SNR it
needs about seven times the total flux of the n=4 class, spread seven times
more thinly. Holding the peak SNR common is what lets a result on one class be
read against another at matched detectability.

By total flux the halo is therefore the **brightest** class in the gallery,
which is deliberate and worth stating plainly: "faint" here means faint per
pixel, not faint overall. Giving the halo a shorter ladder instead would have
decided a number #67 does not state, and would have cost the matched-SNR
comparison that makes the three classes readable against each other.

A point source is a delta function, so it has no Sérsic index, no half-light
radius and no truncation. `sources.csv` is a fixed-column table, so all three
are recorded as zero — literally true for the radius, and not a value any real
Sérsic index takes.

Its **true `e1`/`e2` are zero, and that is a post-PSF statement**, not a
missing value: #67's PSF is a circular Moffat, so a delta convolved with it is
round, and zero is exactly the truth value #25's shape-bias fit compares
against. Adaptive moments on the rendered `truth_conv.fits` measure
`|e| < 1e-5`.

GalSim refuses to draw a bare delta function — it has no scale, so there are
no drawing bounds to choose — and it is right to. `truth_preconv.fits` for the
`point` class is therefore written directly: each source's whole flux in the
one pixel that contains it, and nothing anywhere else. The **convolved** truth
beside it is still GalSim's own work, and is what the flux check runs on. That
hand-written path gets a check of its own; see below.

### The blend class

`blend` is the one class that is a **layout** rather than a profile, and #71
built it as one. Every grid cell holds a **pair**:

| | profile | brightness | varies |
|---|---|---|---|
| bright component | `sersic_n4`, at the cell centre | fixed, peak SNR 1000 | nothing |
| faint companion | `point`, offset from it | peak SNR 100 down to 3 | flux ratio, across the columns |

and the separation between the two steps **down the rows**, from one PSF FWHM
out to the bright component's own wing radius. The result is the
two-dimensional map #67 asks for: where over-subtraction around the bright
source starts consuming its neighbour. Brightness binning is already covered
by the three isolated classes, which is exactly why this grid spends its axes
on something else.

Both components come straight out of `SOURCE_PROFILES`, so the class costs no
new shape numbers, and three of its four numbers are reused rather than
chosen:

- The bright component is the **brightest rung of the isolated ladder**. The
  hole it makes is therefore literally the hole the isolated `sersic_n4` entry
  measures, and a blend result at the widest separation reads straight against
  that entry's first column — what changed is the companion and nothing else.
- The companion's ladder ends on `PEAK_SNR_FAINTEST`, the same marginally
  detectable rung every isolated ladder ends on. Its bright end is an order of
  magnitude below the bright component, which is what makes the pair a
  bright-plus-faint one rather than two comparable sources.
- Both ends of the separation ladder are **measured, not chosen**: one PSF
  FWHM at the tight end, where the pair is a single blob and no deblender can
  win, and the bright component's wing radius at the wide end — the radius at
  which it finally drops under the background, and so exactly the boundary the
  class exists to locate. The widest row lands *on* that boundary rather than
  passing it; what lies outside it is the isolated `sersic_n4` entry, which is
  the same bright source with no companion at all.

The axis is recorded as a flux ratio, per #67, but decided in peak SNR so that
both ends are anchored to the detectability the rest of the gallery is laid
out by. The two are not the same number: a point source concentrates its light
about eight times more than the bright Sérsic, so a companion at a tenth of
the bright component's peak carries about a **hundredth** of its flux. The
ratios on the grid run from about 1:82 to 1:2700.

**The companion is a point source, and that is a measurement decision rather
than an astrophysical one.** Its whole true flux sits inside one PSF
footprint, so "how much of the companion survived" stays a statement about the
hole even at the tightest separation, where the companion is inside the bright
source's core. An extended companion's own wings would overlap the bright
source at every separation on the ladder, and no measurement could then
separate light lost to over-subtraction from light that was never separable.
It buys grid room too: a point source has no truncation to clear, so the pair
fits the isolated classes' own margins and the separation ladder reaches as
far as the refusals allow.

Neither grid axis is free to buy independent samples the way the isolated
classes' jitter row does. The **pair centre** is jittered instead, and the
companion is placed at a random position angle, so no two cells repeat a
sub-pixel phase or an orientation while both axes stay spent on what the class
is for.

`magnitude_ladder` and `peak_snr_ladder` describe the **companion** for this
class, because the column axis is the flux ratio. The bright component does
not step across the grid at all, so it is recorded once, as `bright_magnitude`
and `bright_peak_snr`. Those two shared keys therefore change subject with the
class, so the manifest says so outright in `ladders_describe` rather than
leaving a consumer to infer it from the keys beside them.

`manifest.json` carries more than the keys #68 lists. That list is a minimum,
not a schema: the extra keys are provenance a later reader is better off
having, and nothing reads the manifest by position.

`--all` is the complement of the refusals rather than a second list beside
them: it walks all 72 axis combinations and keeps the ones the recipe does not
refuse. That is why each ticket grew the sweep with no list here to forget to
update — #70 and #71 lifted class refusals and took it from two entries to six
to eight, and #72 lifted the last two and took it to all 72, without a line of
sweep logic changing. `--realisation` carries no `choices` list for the same
reason: `r3` is refused by the recipe, with a sentence saying the gallery has
three realisations, rather than by argparse with "invalid choice".

A *geometry* failure is not filtered out of the sweep. If a decided entry
trips the wing-radius, truncation or edge-margin refusal, `--all` stops on it:
that is a broken decision, and #67 puts those checks in the generator so a
human sees them. Entries already written stay written, so the rerun after the
fix still resumes.

## The three layers

| module | what it does | imports |
|---|---|---|
| `recipe.py` | describes one entry completely — grid, sources, seed, noise, paths, manifest | numpy only |
| `render.py` | hands a recipe to GalSim, returns arrays | GalSim |
| `write.py` | arrays to FITS, CSV and JSON; owns the idempotent skip | astropy |

`recipe.py` is the main test seam, and that is the whole reason the split
exists. It holds every decision of #67 and imports no GalSim, so those
decisions are unit-tested in the ordinary DAWIS test environment in seconds:
`tests/unit/test_simulation_recipe.py`.

`write.py` is a seam too, for one thing only: the skip. It imports the render
layer under `TYPE_CHECKING` and not at run time, so `entry_state` — which
decides whether an entry is absent, built or stale — is reachable without
GalSim. `generate.py` takes its renderer as an argument for the same reason,
so a stub can drive the resume and rebuild paths. That is what
`tests/unit/test_simulation_generator.py` runs on. Nothing there asserts
anything about pixels; the drawn flux is checked by the generator itself, as
#67 asks.

`recipe.py` predicts a peak SNR ladder with its own numpy model of a
PSF-convolved truncated Sérsic. That model is a **design** calculation, not a
second renderer: it lands within about 5% of what GalSim actually draws, which
is well inside the "roughly 3 to 1000" the ladder is specified to.

## Checks the generator makes on itself

All at generation time, when a human is present. None of them is a pytest
assertion: #67 wants a rendering or layout mistake to stop the generator, not
to surface later in a suite nobody was running.

**Flux.** After rendering, the convolved truth image's total is compared
against the recipe's analytic sum, and the entry is refused outside a 1%
tolerance. Measured deviation on the built entries is under 1e-5; the
tolerance is set by GalSim's own drawing accuracy, not by the physics. This
catches a shear applied to the wrong object, a flux normalisation dropped, a
source drawn twice.

That total is checked **one brightness bin at a time** as well as as a whole,
which is #71's doing. The gallery is built out of sources three orders of
magnitude apart, so a single number for a whole entry is a statement about its
brightest sources and almost nothing else: an isolated class's faintest column
is about three parts in a thousand of its total, and a blend entry's
twenty-five companions about five parts in a thousand between them. Every one
of them could fail to draw and the 1% tolerance on the total would still be
met — which for `blend` would mean the check could not tell a companion the
hole ate from a companion that was never rendered, and the companion is the
entry's whole point. So the convolved truth is drawn a bin at a time, each bin
is weighed against its own analytic sum, and the bins are then added up and
the total checked as before.

**Geometry.** Four refusals, all on the recipe, all before a pixel is drawn.

The brightest source's wing radius — the radius at which it falls below the
background — must be smaller than the grid spacing, or the isolated-galaxy
class has silently become a second blend class. That radius is measured on a
profile probed far past where the rendered one is truncated: measuring it on
the rendered profile would make it the cut's prisoner, and no brightness could
ever trip the refusal.

The truncation must in turn sit past the wing radius, or the cut lands above
the noise and shows as a hard circular edge in the truth image. The `point`
class is exempt from this one and only this one: a delta function has no cut,
so there is no cut that could show.

Every source must clear the image edge by its own truncation radius plus the
PSF's reach, or the flux check above could not pass. For `blend` that covers
the companion, which the grid layout does not know about: it is displaced from
the cell centre by up to a whole separation.

The fourth is the `blend` class's own, and it is deliberately distinct from
the first. That one asks whether a *source* reaches its neighbour's cell; this
one asks whether a whole **pair** does, wings and separation together, because
the within-pair separation is the thing this class exists to vary. Four grid
steps on a 512-pixel image would leave 96 pixels between cells — roomy enough
for the bright component's 52 pixels of wings, and not for the 103 a pair at
the widest separation reaches. Without a refusal of its own that entry would
build, and every separation on the map would have been measured across a
boundary another pair was already sitting on.

It also parts company with the first on the jitter. #67 states the isolated
refusal against the *nominal* grid spacing and that wording is left alone, but
the pair refusal compares against the closest two cells can actually come —
the spacing less twice the jitter amplitude. A pair reaches half as far again
as its bright component does, so the 24 pixels the jitter can take are a real
part of that margin rather than slack around it.

All four are live: brightening the ladder a hundredfold trips the first two,
and the cramped grid above trips the fourth.

**Point placement.** The pre-convolution truth of a point source is the only
thing in the gallery GalSim does not draw, so it is the only place a
coordinate-convention mistake would survive the flux check — moving all the
flux one pixel sideways changes no total. Each point source's hand-placed
truth pixel is therefore compared against where GalSim's own convolved draw of
that same source peaks.

Each source is drawn **alone** for that comparison, on a small stamp. #71 is
why: a blend pair puts a faint companion beside a source up to three thousand
times brighter, at separations that start well inside the comparison window,
so in the summed truth images neither peak near the companion is the
companion's. Isolating the source is the only way to keep the check live for
the class that most needs it.

Two questions need two pictures, though, so the delivered array is read too:
the whole of each point source's flux must be present in the truth pixel of
`truth_preconv` itself. The stamps say the *convention* is right; that says the
array that reaches disk actually got the flux, which a check on private draws
alone would no longer notice.

## Packaging

`simulations/` is not in `[tool.setuptools] packages`, so it does not ship in
the wheel. `tests/unit/test_package_surface.py` and
`tests/unit/test_import_graph.py` both walk the installed `dawis` package
only, so neither traverses this directory.

GalSim appears in no DAWIS dependency group, including the optional extras.

## Cost

The reference measurement in #67 is about 262.5 s for a 900x900 DAWIS run, so
the `small` tier is roughly 85 s per run and `full` roughly 340 s. Running the
whole 72-entry gallery through DAWIS once is about 4.2 hours. *Generating* an
entry is cheap by comparison — about 25 s for the `full` tier — so `--all`,
which since #72 really does build all 72, is well under an hour and is meant
to be run in one go. It resumes if it is not.
