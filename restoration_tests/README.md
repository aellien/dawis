# Restoration test battery

The research code of #74 — the map that decides which reconstruction
algorithm DAWIS 1.0 ships. Hand-run, not a pytest tier, not CI. This
directory imports DAWIS; DAWIS never imports it, and it is not in
`[tool.setuptools] packages`, so it does not ship in the wheel. Same
arrangement as `simulations/`.

The directory is `restoration_tests/` with an underscore, where #74 and #75
write `restoration-tests/`. `tests/unit/test_restoration_fixtures.py` imports
it, and a hyphen is not a Python identifier. The branch keeps the hyphen.

Built so far: the **fixture layer** (#75). The metrics and arm runner (#77),
the proximal arms (#78) and the reporting (#79) are not here yet.

## What a fixture is

One interscale tree, cut out of one simulation-gallery entry by one mock
iteration, together with the injected source it is trying to reconstruct.
Enough to run a reconstruction and score it with nothing to look up
elsewhere:

- the wavelet, support and label tubes
- the tree's geometry — interscale maximum position and level, bbox, extent,
  eccentricity
- the matched source's row of `sources.csv`, and the PSF-convolved truth cut
  to the tree's box
- what the published default arm made of the tree, and which filter and
  operator it chose

## The mock iteration

Deliberately not a faithful DAWIS iteration. One pass:

```
image -> background -> Anscombe -> a trous, 8 levels -> detection
      -> interscale trees (min_span=2, max_span=8)
```

No reconstruction inside the loop, no subtraction, no inpainting, no greedy
loop. #74 is explicit about why: the greedy loop exists to compensate for a
weak reconstruction, so measuring the whole loop would hide what the map is
about.

`ceps=1e-3` is recorded in every manifest and read by nothing. It is the
greedy loop's convergence threshold, and a single pass has no loop to stop.
#75 names it as part of the frozen configuration, so it is carried; saying it
is inert is cheaper than a later reader deciding it must have done something.

## tau is 0.01, and #75 writes 0.1

Measured this session. At `tau=0.1` one mock iteration on a small-tier entry
returns three trees, and all three sit on the **brightest magnitude column** —
the faint columns produce no tree at all, for every one of the three classes.
A fixture set with one brightness bin cannot carry the brightness-binned
residuals #67 asks for and #25 specifies. `tau=0.01` adds the middle column.

The faintest column, at peak SNR 3, appears at neither value. That is the
greedy loop's job and this is one pass.

What #75 freezes is the *pairing* — every arm must see the same tubes — not
the number. 0.01 is frozen just as hard. Amael's call, this session.

## Matching a tree to a source

A tree is matched when its interscale maximum falls inside the injected
source's half-light radius, and inside **exactly one** source's. Two sources
sharing a maximum is a blend under another name, so the tree is refused
rather than assigned to the nearer of the two.

Detection performance is not under test in #74, so an unmatched tree is
dropped and counted rather than scored. The counts are in every manifest:
they are what says whether a fixture set is about the sources or about
detection.

**The PSF is the floor on the radius.** A point source's half-light radius is
zero — literally true, since a delta encloses all of its light at every radius
— and no interscale maximum falls inside a radius of zero, so the whole
`point` class would come back unmatched. The light a point source actually
puts in the image is the PSF's, and its FWHM is the width that light is
spread over. A floor rather than an added term: the extended classes'
half-light radii are already several times the PSF, and widening them further
would start reaching for neighbours.

### The two coordinate conventions

They are transposes of each other, and nothing raises if they are swapped.

`sources.csv` measures from the lower-left corner of the first pixel, `x`
along the **columns** and `y` down the **rows** — GalSim's way of seeing an
image. A DAWIS region's `x_max`, `y_max` are a numpy **row** and **column**:
`make_regions_full_props` reads them out of skimage's `coords`, which are
`(row, col)`. So a source's `x` is a tree's `y_max`.

The gallery lays its sources on a grid, and a transposed grid is still a
grid, so a swapped axis quietly matches a tree to a *different source of the
same class and the same brightness* — a fixture set no metric could tell from
a correct one. `match_tree` therefore names its arguments `peak_row` and
`peak_col`, and `tests/unit/test_restoration_fixtures.py` runs one end-to-end
pass over an image whose two blobs sit off each other's row and column.

## Building

One entry:

```
python -m restoration_tests.fixtures.build --entry \
    ~/dawis-data/simulations/small/sersic_n4_balanced_r0
```

Or every entry the gallery holds:

```
python -m restoration_tests.fixtures.build
```

Any axis given is held fixed and the rest are swept, the same way
`simulations/generate.py` works: `--tier`, `--class`, `--regime`,
`--realisation`. Blend entries are always skipped — #75 scores single sources
first.

The sweep is resumable per entry, and the resume compares **settings**, not
just the presence of a manifest. An entry already built under the settings
you are running is skipped; an entry built under different ones is rebuilt.
`--force` rebuilds regardless.

The axes are read off each entry's own `manifest.json`, never parsed out of
its directory name. An entry is `<class>_<regime>_r<N>` and the `read_noise`
regime has an underscore in it, so splitting on underscores makes
`blend_read_noise_r0` a `blend_read` of regime `noise` — which is how the
first real sweep stopped on a blend entry.

`--tau` overrides the frozen value. A dump under a different tau is a
different fixture set, and the manifest records which — which is why the
resume reads the settings. Testing presence alone meant a second sweep at
another tau skipped every entry and reported success, leaving the first set
on disk under a manifest naming the first tau.

Panels are drawn **before** the manifest is written, because the manifest is
what the resume reads. matplotlib is an optional extra, so the other order
would let a dump on an install without it write the manifest, raise from the
drawing, and leave an entry marked built whose panels never exist. The panel
directory is cleared rather than overwritten, so a rebuild yielding fewer
fixtures leaves no orphaned `treeNNN.png` behind.

## What one fixture directory holds

```
~/dawis-tests/restoration-fixtures/<tier>/<class>_<regime>_r<N>/
    fixtures.h5          # one group per fixture, gzip-compressed arrays
    manifest.json        # entry, settings, match counts, dtypes, commits
    quicklook/tree000.png ...
```

HDF5 rather than one `.npz` per fixture: the tubes are the bulk of the bytes
and most of a support tube is zero, so per-dataset gzip earns its place, and
an arm sweeping one entry opens one file rather than thirty. h5py is already
a hard DAWIS dependency. One file per **entry** rather than one for the whole
gallery, because the sweep is resumable per entry and a half-written
gallery-wide file is not.

Fixtures are numbered in the group name and read back in numeric order. h5py
iterates a group alphabetically, where `tree10` sorts before `tree2`, and a
quicklook filename and a row of a results table both index a set by position.

## The quicklook

Three panels per fixture: the injected truth over the tree's box, what the
published default arm made of it, and the residual between them.

Truth and reconstruction share one linear scale, because the panel exists to
be compared across — a reconstruction at half the truth's flux, scaled to its
own maximum, looks exactly like the truth. The residual gets its own scale,
symmetric about zero on a diverging map, because what it carries is sign.

Linear, not logarithmic: a tree's box is a patch of one source, so the range
inside it is that source's own profile, and a log stretch would spend the
whole colour bar on the wings.

`quicklook.py` is the only module here that imports matplotlib, the way
`dawis.viz` is for DAWIS. `--no-quicklook` never reaches it.

## The reference arm is a baseline, not a target

Every fixture carries what `restore_object` made of its tree under the
pipeline's own defaults, read off `dawis.config.Config` rather than written
down again. It is the "current default arm" #75 wants the quicklook to draw,
and the baseline #77 reads every other arm against.

It is recorded, not aimed at. The whole point of #74 is that this arm may be
the wrong one.

## Layers

| module | what it does | imports |
|---|---|---|
| `matching.py` | which source a tree belongs to | — |
| `entry.py` | one gallery entry, read off disk | astropy |
| `store.py` | what a fixture is; write and read a set | h5py |
| `mock.py` | the one-pass mock iteration | DAWIS |
| `build.py` | match, cut, reconstruct; the sweep and the CLI | the above |
| `quicklook.py` | the three-panel figure | matplotlib |

Split the way `simulations/` is: the decisions on one side of the seam, the
machinery that needs a heavy import on the other. `matching.py` is the main
test seam and holds every decision of #75 that is not about pixels.

## Tests

`tests/unit/test_restoration_fixtures.py`, in the ordinary suite. Two seams,
agreed before a line was written: the pure layer directly, and one end-to-end
mock iteration on a small synthetic image — which exists for the coordinate
convention above, and for nothing else.
