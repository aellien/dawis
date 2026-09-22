# #110 — the iterate a proximal solve returns is now the iterate it tests

[A diverged proximal solve is indistinguishable from a converged
one](https://github.com/aellien/dawis-dev/issues/110), on
[Map: the chosen reconstruction lands in
`dawis/`](https://github.com/aellien/dawis-dev/issues/96).

The ticket named two defects. The first was already settled at
[#104](https://github.com/aellien/dawis-dev/issues/104): `restore_object`
tests `norm_wr` as well as the image, so a proximal arm's zeroed atom comes
back with `flag_convergence = False` instead of passing through as a converged
one. That was an *agreement* with `restoration_tests/arms/base.py`, so it
moved nothing and needed no equality statement.

This is the second, and it is a **departure**.

## The defect

`_forward_backward` and `_condat_vu` measure the wavelet residual at the
iterate the step *starts from*, take the step, and then call `_diverging` with
the new image and the old residual norm. So iterate *k* is only scored at
iteration *k + 1* — and a loop that ends on `max_iter` or on the flux rule has
no iteration *k + 1*. **The iterate that is actually returned was the one
nothing scored.**

`Config.cg_max_iter` is a user knob and this repo has benchmarked at 3
(`benchmarks/results/.../cg-max-iter-3/`), so a low cap is not hypothetical.

## The repair

`_finish` scores the iterate it is about to return, using the residual it
computes for `sum_wr` and `norm_wr` anyway — so the repair costs nothing. With
it in place every iterate is tested: the earlier ones one step late by the
loop, the last one by `_finish`. The lag stops being a hole.

It takes `first_norm_wr` as an argument, `None` when `max_iter` left the loop
no iteration to measure one in — nothing moved, so there is no growth to test.

`restoration_tests/arms/proximal.py` keeps the hole. It stays the record of
what every #74, #81 and #92 number was taken with, which is the same reason
[#109](https://github.com/aellien/dawis-dev/issues/109) left the reference
dividing by zero.

## What it moves

### `verdict_shift.py` — #81's own 540 rows

The repair can only flag a solve whose **returned** iterate has
`norm_wr > 1e3 * first_norm_wr`. #81 saved the returned residual; it did not
save the starting one, but that costs one step rather than two hundred — the
initial iterate is `clip(synthesise(w), 0, None)` and depends on neither the
step length nor the cap. So each row is settled by re-running its own arm on
its own fixture at `max_iter = 1` with one power iteration, reading
`records[0].norm_wr` off the reference arm itself.

```
rows           : 540
already flagged: 108     (the two HAAR.SUM arms, 54 fixtures each)
re-scored      : 432
worst growth   : x1.214  (fista.BSPL.SUM on small/sersic_n1_halo_balanced_r1/tree005)
VERDICTS MOVED : 0
```

**No #81 verdict moves, and none is close to moving.** At #81's and #92's
operating point of `max_iter = 200`, a solve that grows past the factor has
already been flagged by the loop long before step 200; every solve that
survives to return an atom ends within a factor 1.214 of where it started,
three orders of magnitude below the `1e3` that flags. Nothing downstream of
#81 — #92's table, #94's in-loop run, #82's choice — reads a verdict that
moves.

### `departure_sweep.py` — the window where it is not inert

#102's 28 cells on #84's field, crossed with an iteration ladder covering the
benchmarked cap of 3 and #81's 200. Each cell solved by `dawis/` and by the
reference arm, and classified.

```
CELLS: 28   CAPS: 14   RUNS: 392   DEPARTURES: 1   FAILURES: 0

minimizer   filt  operator  coarse  cap  first_norm_wr  final_norm_wr  growth
fista       HAAR  SUM       False   30   294.94         3.6263e+05     x1230
```

`FAILURES: 0` is three checks, and the third is the claim:

1. **The repair never unflags.** No cell the reference calls diverged is one
   `dawis/` returns as an atom.
2. **It never moves an atom that is returned.** Where both sides return an
   atom the reconstruction, `sum_wr` and `norm_wr` are bit-identical — this
   changes what is *reported about* a solve, not what the solve computes.
3. **Every departure is justified by the reference's own numbers.** The one
   departure has the reference returning an iterate whose residual is 1230
   times the one it started from.

The window is one iteration wide per diverging cell, which is what a one-step
lag predicts: at a smaller cap the residual has not reached the factor yet, at
a larger one the loop's own test has fired. The cell is `fista.HAAR.SUM` with
the coarse channel off, which is #74's sighting arm — the one that diverges
where `projected_gradient` converges on the same tube.

## What this does not fix — `growth_ladder.py`

The same cell, cap by cap, reading the reference arm's own records:

```
cap 1    x1.032      cap 15   x55.62      cap 28   x780.4
cap 2    x1.082      cap 20   x145.8      cap 29   x978.1
cap 3    x1.135      cap 22   x216.7      cap 30   x1230
cap 5    x1.693      cap 24   x327.3      cap 31   diverged
cap 10   x16.22      cap 25   x404.6      cap 40   diverged
```

The reference first flags at 31 and `dawis/` at 30 — **exactly one step
earlier, which is the size of the lag.**

The guard's **threshold** is `DEFAULT_DIVERGENCE_FACTOR = 1e3`, and nothing in
DAWIS reads the growth *rate*. A solve capped below the step where it crosses
the factor still returns an atom whose residual has climbed every iteration:
at cap 20 this one returns an atom at 145.8 times its starting residual, and
is right to, by this rule. That is the factor's business and not the lag's,
and it is left as a finding on the ticket rather than widened into here.

At the benchmarked cap of 3 the growth is 1.135, so a `cg_max_iter = 3` run
is nowhere near either boundary — the knob that made a low cap realistic is
not the knob that makes it dangerous.

The ticket's own framing was wider than the measurement supports. It read as
though `max_iter` 1 to 24 each returned an atom 1700 times its starting
residual; the growth is monotone in the cap, so only the top of any such
range is over the factor at all, and on #84's field no cap at or below 24 is.

## Running them

```bash
cd ~/dawis                       # research/restoration-tests
PYTHONPATH=~/dawis-land:~/dawis python ~/dawis-land/.research/110-divergence-guard/verdict_shift.py
PYTHONPATH=~/dawis-land:~/dawis python ~/dawis-land/.research/110-divergence-guard/departure_sweep.py
```

`dawis-land` first on the path: the reference arm comes from the research
checkout, `dawis` from the branch the repair lands on.

`verdict_shift.py` needs `~/dawis-tests/restoration-fixtures` and
`~/dawis-tests/restoration-results-81`.

## Where this lives

The four earlier port proofs sit in `~/dawis/.research/` on
`research/restoration-tests`, beside the reference arm they compare against.
This one is in `dawis-land/.research/` and committed on `refactor/1.0` with
the change it proves, because the session that ran it was isolated to that
worktree. Move it if the convention matters more than the adjacency.
