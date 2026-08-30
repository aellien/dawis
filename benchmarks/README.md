# Benchmarks

Hand-run wall-time measurement for DAWIS. Not a pytest tier, not CI.

`run_benchmark.py` measures one full synthesis-by-analysis run on the 900x900
reference image and writes `benchmark.json` into the run directory: wall time
overall, wall time per pipeline phase (restoration in particular), peak RSS,
iteration and atom counts, and the full configuration, source commit and
machine identification needed to reproduce the number.

Results live in `results/<date>-<machine>/`, one directory per measurement
session, holding a `README.md` and each run's `benchmark.json`.

The script needs `psutil` for the peak-RSS number, which is not a runtime
dependency of DAWIS. Install it with `pip install 'dawis[bench]'`. The script
refuses to start without it rather than quietly reporting 0 MB.

## The two `max_iter` knobs

They are unrelated and both appear in the pipeline.

- `--pipeline-max-iter` caps the greedy outer loop. This is `Config.max_iter`
  on the 1.0 tree and the entry point's own `max_iter` on `legacy-seeded`, and
  the reference parameter set leaves it at 500.
- `--cg-max-iter` is the conjugate-gradient solver's iteration cap inside
  `InterscaleTree.CG_minimization`. `restore_object` never passes it, so the
  shipped value is the method's default. RFC #29 restores that default from the
  3 introduced by `7c5ba8e` to the published 200. This is the knob the baseline
  sweeps, and the script forces it rather than editing the tree it measures.

## Running the baseline

Both runs go on one machine, single core, against the `legacy-seeded` checkout
(#32) so the numbers describe the algorithm as published in Ellien+2021:

```
git worktree add ~/dawis-legacy-seeded legacy-seeded

python benchmarks/run_benchmark.py --source ~/dawis-legacy-seeded \
    --outdir ~/dawis-bench/cg003 --cg-max-iter 3 \
    --label "legacy-seeded, CG max_iter=3"

python benchmarks/run_benchmark.py --source ~/dawis-legacy-seeded \
    --outdir ~/dawis-bench/cg200 --cg-max-iter 200 \
    --label "legacy-seeded, CG max_iter=200"

python benchmarks/run_benchmark.py --report ~/dawis-bench
```

`--report` aggregates every `benchmark.json` under a directory into the
markdown table that goes into `results/`.

`data_dump` and `gif` are pinned off, so the measurement is CPU work and not
disk or matplotlib work. Neither is a flag any more: #46 took the animation
out of the configuration and #66 deleted `data_dump` outright, so on the 1.0
tree neither knob exists and on `legacy` / `legacy-seeded` both are pinned off
by the script. `--gif` survives as a flag for the legacy trees alone, and it
draws frames inside the loop -- 52% of it, per #5 -- so turn it on only when
that is the thing being measured.

`--indir` / `--infile` point the run at a different image. The baseline never
uses them -- it is the 900x900 reference or it is not comparable to #5 -- but
they make the script smoke-testable on a small cutout in seconds, which is how
you check a change to the script itself before spending 24 minutes on it.

## Missing optional imports

Legacy DAWIS imports `ray` and `ultranest` at module scope. Neither is reached
on the serial path, and `ultranest` is never referenced at all, so the script
installs do-nothing stand-ins for whichever of the two is not installed and
records the substitution in `benchmark.json`. Pass `--no-stub-missing` to fail
instead.
