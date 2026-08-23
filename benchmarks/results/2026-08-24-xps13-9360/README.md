# Baseline: 900x900 reference image, single core, 2026-08-24

The step-zero baseline #13 asked for and RFC #29 ticketed as #33. Every speed
number measured on the 1.0 tree is compared against this file, and this file
states its own relationship to #5's published 3539 s.

Each run's raw output is committed beside this file as
`cg-max-iter-3/benchmark.json` and `cg-max-iter-200/benchmark.json`, carrying
the full effective configuration -- every pipeline parameter with the value it
actually held, defaults resolved -- plus source commit, package versions and
machine identification.

## Machine

| | |
|---|---|
| Host | `amael-XPS-13-9360` |
| CPU | Intel Core i7-7500U @ 2.70 GHz, 4 logical CPUs |
| RAM | 14.7 GB |
| OS | Linux 7.0.0-29-generic, x86_64, glibc 2.43 |
| Python | 3.13.9 |
| numpy / scipy / astropy | 2.3.5 / 1.16.3 / 7.1.1 |
| h5py / photutils / scikit-image | 3.15.1 / 3.0.0 / 0.25.2 |
| matplotlib / psutil | 3.10.6 / 7.0.0 |

Both runs come from one session on this one machine, back to back, nothing else
running. `ray` and `ultranest` are not installed; the benchmark stubbed both.
Neither is reached on the serial path and `ultranest` is never referenced at
all, so this changes no measured work -- but it does mean peak RSS excludes
whatever a real `import ray` would have added.

## Source under measurement

Branch `legacy-seeded` at `1434f66` (`v0.0.7-h-5-g1434f66`), clean tree. This
is the #32 branch: the published Ellien+2021 algorithm, with the RNG pinned and
four post-publication drifts reverted. #32 specified three (the Wolfe call-site
swap, the hardcoded quality cut, the CG `max_iter`); `1434f66` is a fourth,
found while implementing #32, which forwards `deconv` on the serial restoration
path. Seed `0`.

## Configuration

`test.py`'s parameter set verbatim, which is also what #5 profiled:

```
n_levels=9, tau=0.1, gamma=0.5, ceps=1e-4, min_span=2, max_span=3,
lvl_sep_big=5, starting_level=2, n_cpus=1, max_iter=500 (outer loop),
data_dump=False, gif=False, resume=False, seed=0
```

Everything else took the entry point's defaults. The ones that matter for
reading these numbers: `deconv=False` (so no Richardson-Lucy runs inside the CG
loop), `conditions='loop'`, `rm_gamma_for_big=False`, `inpaint_res=True`,
`n_sigmas=5`, `deblend_contrast=0.1`, `lvl_deblend=3`, `extent_sep=0.1`,
`ecc_sep=0.95`, `lvl_sep_lin=2`, `lvl_sep_op=3`. The committed JSON carries all
32.

Image: `dawis/gallery/tidal_group_f814w_sci.fits`, 900x900.

`data_dump` and `gif` are off, so these numbers are CPU work only. #5 measured
frame writing separately at 52% of the loop; #29 removes that cost
architecturally by moving the animation out of the iteration loop.

The swept knob is the CG solver's `max_iter` inside
`interscale_tree.CG_minimization`. `restore_object` never passes it, so the
shipped value is the method's default; the benchmark forces it per run rather
than editing the tree it measures. The default in this source is already 200,
so the `max_iter=200` run measures the tree unmodified (`cg_max_iter_forced:
false`) and only the `max_iter=3` run is patched (`cg_max_iter_forced: true`).

## Results

| Run | CG `max_iter` | Wall (s) | Loop (s) | Restoration (s) | Restoration share | Iterations | Atoms | Peak RSS (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy-seeded, CG max_iter=3 | 3 | 262.5 | 249.7 | 205.1 | 82.1% | 22 | 418 | 949.0 |
| legacy-seeded, CG max_iter=200 | 200 | 1167.6 | 1155.0 | 1125.0 | 97.4% | 19 | 394 | 678.1 |

Per phase, seconds:

| Phase | `max_iter=3` | `max_iter=200` |
|---|---:|---:|
| inpaint + resume | 0.7 | 0.7 |
| wavelet + detect | 32.2 | 20.8 |
| interscale trees | 7.7 | 5.3 |
| **restoration** | **205.1** | **1125.0** |
| atoms + convergence + write | 4.0 | 3.2 |

An earlier session on the same machine, with an earlier revision of the
benchmark script, produced 250.1 s / 1163.6 s wall and the same iteration and
atom counts (22 / 418 and 19 / 394). The trajectory is reproducible at a fixed
seed; wall time carries a few percent of run-to-run noise.

## What the two numbers say

**Cost of restoring the published `max_iter`.** Whole run 4.45x, loop 4.63x,
restoration stage 5.49x. Normalizing for the different atom counts,
restoration costs 0.491 s/atom at 3 against 2.855 s/atom at 200 -- **5.82x per
atom**.

This is below the **9.3x to 11.0x** that #29's grilling session measured with a
throwaway script. The two are not measuring the same thing: this run holds the
whole pipeline fixed and lets the trajectory move, and the trajectory does move
(22 iterations / 418 atoms against 19 / 394), so a different set of atoms is
restored in a different order. Quote 5.82x per atom as the like-for-like cost
and 4.45x as what a user feels. Neither number disturbs #29's decision, which
rested on the flux delta rather than on the cost.

**Restoration's share rises with the knob**, from 82.1% to 97.4% of the loop.
#5 recorded 98.4% on the Wolfe path. #13's step 2, the mandatory re-profile
after the `atrous` vectorization, should expect restoration to still dominate.

**Peak RSS is lower at 200, not higher** (678 MB against 949 MB). This is a
trajectory effect, not a memory saving: the run converges in three fewer
iterations on a different set of atoms. Memory is not a function of the CG
budget.

## Relationship to #5's 3539 s

**#5's 3539 s is not a like-for-like anchor for either number above. The reason
is the minimizer, not the iteration count.**

#5 profiled `legacy` at `ae19cb2`, where `restore_objects.py:139` sent the
serial path through `restore_object_test` ->
`CG_minimization_wolfe_synthesis_options` (`max_iter=50`, `gtol=1e-6`, measured
at 246 objective evaluations per atom). `legacy-seeded` reverts that call-site
swap (`0576045`, reversing `ae19cb2`) because the Wolfe minimizer is
unvalidated post-publication drift called with five shifted arguments --
decided on #20. So `legacy-seeded` runs `restore_object` -> `CG_minimization`,
the published path, and `CG_minimization`'s `max_iter` is the knob this file
sweeps. On the code #5 measured, that knob was never reached at all.

Two further differences ride along with that revert, and both change what the
restoration stage does rather than only how fast it does it:

- **The argument shift.** On `ae19cb2` the serial call passes eight positional
  arguments into a signature that carries a vestigial `sigma` in slot four, so
  the run #5 timed used `extent_sep=0.95`, `ecc_sep=2`, `lvl_sep_lin=5`,
  `lvl_sep_big=3` -- which changes which trees get the SUM synthesis operator
  and which get ADJOINT. `legacy-seeded` restores the intended values.
- **The quality cut.** `ae19cb2` carried the hardcoded
  `sum_wr > 1000 & norm_wr > 100` cut that zeroes an atom outright;
  `legacy-seeded` reverts it (`ccc5d69`). Cheap in time, but it means #5's 438
  atoms and this file's 394-418 are not the same population.

`deconv` is *not* one of the differences: `CG_minimization_wolfe_synthesis_options`
has no `deconv` parameter and no Richardson-Lucy call, so the flag is inert on
the path #5 measured, and it is `False` here.

Comparable across the three, restoration seconds per atom:

| Source | Serial minimizer | Atoms | Restoration (s) | s/atom |
|---|---|---:|---:|---:|
| #5, `legacy` `ae19cb2` | Wolfe, `max_iter=50` | 438 | 3463.9 | 7.908 |
| this file, `max_iter=3` | published CG | 418 | 205.1 | 0.491 |
| this file, `max_iter=200` | published CG | 394 | 1125.0 | 2.855 |

**The shipping 1.0 configuration is about 2.8x cheaper per atom than what #5
measured, and the whole run is 3.03x faster (1167.6 s against 3539 s)** -- even
though #29 restores `max_iter` from 3 to 200. Restoring the published value
made the run slower than the drifted value it replaced; reverting the Wolfe
call site made it much faster than #5's anchor; and the second effect is the
larger of the two.

Caveat on the machine: #5 records only "a 4-CPU / 14 GB laptop" and no
hostname, so the two sessions cannot be proven to be the same box. The
specification matches this one. Treat cross-comparisons with #5 as
approximately-same-hardware, and treat only the two rows in this file as
strictly one machine, one session.

## Reproducing

The commands are in [`../../README.md`](../../README.md). Wall time for the
pair on this machine is about 24 minutes.
