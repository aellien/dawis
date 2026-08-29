# Simulation gallery

Synthetic FITS images with exact ground truth, for the restoration test
battery of #25. Hand-run, not a pytest tier, not CI. Spec: #67. This directory
was built by #68; #69 added the generator's full command line.

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

`image.fits` says nothing about the noise, in its header or anywhere else.
DAWIS estimates the noise itself, as it would on real data. The true sky
level, read noise and gain are in `manifest.json`, for a human reading a
result — never for a run to pick up.

The sky is **left in** `image.fits`, not subtracted back out after the noise is
drawn. #67 anchors the balanced regime to the gallery image's median *and* its
standard deviation; subtracting would keep the second and throw away the
first, and would leave the manifest's `sky_level` describing a level no longer
in the pixels — which is the comparison it is recorded for.

## What is built

#68 was the tracer bullet: one narrow path through all three layers, for
`sersic_n4` / `balanced` / `r0`. Both tiers of that path build, so `--all`
sweeps **two** of the 72 entries #67 specifies.

Every other combination of the four axes is **refused**, not guessed. Asking
for the `blend` class or the `photon` regime raises `AxisNotBuiltError` with a
message saying the numbers belong to the ticket that decides them. A recipe
that invented them would put an entry on disk that looks exactly as
authoritative as a decided one.

The parts that #67 decides outright are already general: seed derivation over
all four classes and three regimes, both tiers' geometry, the entry and file
naming, the manifest schema.

**The tier axis is not refused**, unlike the other three. #67 decides both
tiers completely — an image size and a grid side each, and nothing else — so
there is no undecided number to guess at, and `small` generates. The other
three axes each need parameters no ticket has written down.

`manifest.json` carries more than the keys #68 lists. That list is a minimum,
not a schema: the extra keys are provenance a later reader is better off
having, and nothing reads the manifest by position.

`--all` is the complement of those refusals rather than a second list beside
them: it walks all 72 axis combinations and keeps the ones the recipe does not
refuse. A ticket that decides the `point` class lifts one refusal and the
sweep grows by itself, with no list here to forget to update. `--realisation`
carries no `choices` list for the same reason: `r3` is refused by the recipe,
with a message naming the ticket that owns it, rather than by argparse with
"invalid choice".

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
tolerance. Measured deviation on the built entry is about 5e-7; the tolerance
is set by GalSim's own drawing accuracy, not by the physics. This catches a
shear applied to the wrong object, a flux normalisation dropped, a source
drawn twice.

**Geometry.** Three refusals, all on the recipe, all before a pixel is drawn.

The brightest source's wing radius — the radius at which it falls below the
background — must be smaller than the grid spacing, or the isolated-galaxy
class has silently become a second blend class. That radius is measured on a
profile probed far past where the rendered one is truncated: measuring it on
the rendered profile would make it the cut's prisoner, and no brightness could
ever trip the refusal.

The truncation must in turn sit past the wing radius, or the cut lands above
the noise and shows as a hard circular edge in the truth image.

Every source must clear the image edge by its own truncation radius plus the
PSF's reach, or the flux check above could not pass.

All three are live: brightening the ladder a hundredfold trips the first two.

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
entry is cheap by comparison — about 25 s for the `full` tier.
