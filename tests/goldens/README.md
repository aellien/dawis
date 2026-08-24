# Goldens

Recorded outputs that pin DAWIS behaviour, plus `provenance.toml` recording
for each one: the `legacy-seeded` commit it came from, the seed, the
tolerances, and who regenerated it, when and why.

Goldens live in plain git, no LFS. Revisit that only if they grow past
about 100 MB.

They are a ratchet, not a shrine. Through the 1.0 refactor no golden is
regenerated, because nothing should move. When later science work does move
one, it is regenerated in a commit whose message says why.

## `tier1_cutout/` — the per-commit gate (#36)

One seeded run on the agreed 256x256 patch of the gallery image, recorded from
`legacy-seeded` at `1434f66`: the published Ellien+2021 algorithm with the RNG
pinned and the post-publication drift reverted (#32). About 40 s, 20
iterations, 76 atoms.

| File | What it pins |
|---|---|
| `input_cutout.fits` | the exact patch the run was fed |
| `restored.fits` | the sum of every atom |
| `residuals.fits` | what the loop left behind |
| `atoms.csv` | one row per atom: position, scale, operator, flux |
| `convergence.csv` | one row per greedy iteration |
| `provenance.toml` | commit, seed, tolerances, pinned `max_iter`, author, date |

`input_cutout.fits` is a golden in its own right: a misread cutout coordinate
still lands on a valid patch, so comparing the input first turns a silent
difference into a loud failure. `provenance.toml` states the convention in
full.

What a Tier 1 run *is* lives in `tests/tier1.py`, shared by the recording
script and by the comparison in `tests/regression/test_golden_cutout.py`. The
tolerances come out of `provenance.toml`, so a golden cannot be recorded under
one rule and compared under another.

### Regenerating

Only in a commit whose message says why the golden moved.

```bash
python tests/goldens/record_cutout_golden.py \
    --source ~/dawis-legacy-seeded \
    --reason "why this golden moved"
```

`--source` must be a clean `legacy-seeded` checkout; the script refuses
anything else unless you pass `--allow-any-branch`. It stubs `ray` and
`ultranest` if they are not installed — neither is reached on the serial path.

Tier 2 goldens are recorded by #37.
