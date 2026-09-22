"""#110 part B — what the repaired guard moves, cell by cell and cap by cap.

``verdict_shift.py`` shows the repair is inert on #81's whole population,
because #81 ran at ``max_iter = 200`` and a solve that grows past the factor
has already been flagged by then. This is the other side: the window where it
is *not* inert, which is small ``max_iter`` — and ``Config.cg_max_iter`` is a
user knob this repo has benchmarked at 3.

#102's 28 cells — the two forward-backward arms by three synthesis operators
by two filters with the #99 coarse channel off and on, and Condat-Vu by two
filters by the channel — crossed with an iteration ladder, on #84's field.
Each cell is solved twice: by ``dawis/`` on the branch the repair lands on,
and by ``restoration_tests/arms/proximal.py``, which keeps the hole and stays
the record of what every #74, #81 and #92 number was taken with.

Three things are checked, and the third is the claim:

1. **The repair never unflags.** A cell the reference calls diverged is a
   cell ``dawis/`` calls diverged.
2. **It never moves an atom that is returned.** Where both sides return an
   atom, the reconstruction and both residual quantities are bit-identical —
   so this is a change to *what is reported about* a solve, not to what the
   solve computes.
3. **Every departure is a solve whose returned iterate is over the factor.**
   The reference's own final residual is read off and divided by its own
   first residual, so the departure is justified by the reference's numbers
   rather than by ``dawis/``'s.

Run from a checkout of ``research/restoration-tests``, with ``dawis`` taken
from the branch the repair landed on:

    PYTHONPATH=/home/amael/dawis-land python departure_sweep.py
"""

from __future__ import annotations

import sys
from dataclasses import replace

import numpy as np

import dawis
from dawis.restoration.inpainting import subtract_background
from dawis.trees import minimize
from dawis.wavelets import bspl_atrous
from restoration_tests.arms.grid import FILTER_ARRAYS, ArmSpec
from restoration_tests.arms.proximal import (
    DEFAULT_DIVERGENCE_FACTOR,
    condat_vu,
    fista,
    projected_gradient,
)
from restoration_tests.fixtures.mock import FROZEN_SETTINGS, run_mock_iteration

print('dawis          :', dawis.__file__)

N_LEVELS = 5
SIGMA_FLUX = 1e-3

#: The window the repair can act in. 3 is what this repo benchmarked
#: `cg_max_iter` at; 200 is #81's and #92's operating point.
CAPS = (1, 2, 3, 5, 10, 15, 20, 22, 24, 25, 30, 50, 100, 200)

ARMS = {
    'fista': fista,
    'projected_gradient': projected_gradient,
    'condat_vu': condat_vu,
}

# --- #84's field, the one #98, #99, #100 and #102 proved their axes on ----
shape = (128, 128)
rows, columns = np.indices(shape).astype(float)
image = np.zeros(shape)
for row, column, amplitude, width in ((30.0, 90.0, 400.0, 3.0),
                                      (95.0, 40.0, 250.0, 4.0)):
    image += amplitude * np.exp(
        -((rows - row) ** 2 + (columns - column) ** 2) / (2.0 * width ** 2))
image += np.random.RandomState(7).normal(loc = 10.0, scale = 1.0, size = shape)

iteration = run_mock_iteration(image, replace(FROZEN_SETTINGS,
                                              n_levels = N_LEVELS))
subtracted = subtract_background(image, n_sigmas = 3, bins = 200)[0]
coarse_dc, _ = bspl_atrous(subtracted, N_LEVELS, None, coarse = True)

tree = max(iteration.trees, key = lambda t: t.interscale_maximum.level)
print('tree level     :', int(tree.interscale_maximum.level))
print('factor         :', DEFAULT_DIVERGENCE_FACTOR)

CELLS = ([('fista', f, o, c) for c in (False, True)
          for f in ('BSPL', 'HAAR')
          for o in ('SUM', 'ADJOINT', 'TRANSPOSE')]
         + [('projected_gradient', f, o, c) for c in (False, True)
            for f in ('BSPL', 'HAAR')
            for o in ('SUM', 'ADJOINT', 'TRANSPOSE')]
         + [('condat_vu', f, None, c) for c in (False, True)
            for f in ('BSPL', 'HAAR')])

failures = 0
runs = 0
departures = []


def compare(minimizer, filter_kw, operator, coarse, cap):
    """Solve one cell at one cap on both sides, and classify the pair."""

    global failures, runs

    coarse_cube = coarse_dc if coarse else None
    wavelet_tube, support_tube, _ = tree.tubes(
        iteration.wavelet_cube, iteration.label_cube,
        coarse_cube = coarse_cube)

    spec = ArmSpec(minimizer, filter_kw, operator, coarse = coarse)
    records = []
    research = ARMS[minimizer](
        spec, wavelet_tube, support_tube,
        observer = lambda record, _: records.append(record),
        max_iter = cap, sigma_flux = SIGMA_FLUX)

    landed, sum_wr, norm_wr = minimize(
        tree, iteration.wavelet_cube, iteration.label_cube,
        filter = FILTER_ARRAYS[filter_kw], minimizer = minimizer,
        synthesis_operator = operator, coarse_cube = coarse_cube,
        max_iter = cap, sigma_flux = SIGMA_FLUX)

    runs += 1
    flagged = bool(np.isnan(norm_wr))

    # 1. the repair never unflags.
    if research.diverged and not flagged:
        failures += 1
        print('UNFLAGGED  %s.%s.%s coarse=%s cap=%d' % (
            minimizer, filter_kw, operator, coarse, cap))
        return

    if flagged and not research.diverged:
        # 3. a departure: the reference returned this iterate as an atom, so
        # its own numbers say whether it should have.
        ratio = research.norm_wr / records[0].norm_wr
        departures.append((minimizer, filter_kw, operator, coarse, cap,
                           records[0].norm_wr, research.norm_wr, ratio,
                           len(records)))
        if ratio <= DEFAULT_DIVERGENCE_FACTOR:
            failures += 1
            print('UNJUSTIFIED %s.%s.%s coarse=%s cap=%d ratio=%.4g' % (
                minimizer, filter_kw, operator, coarse, cap, ratio))
        return

    # 2. where both return an atom, nothing moved.
    if not flagged:
        same = (np.array_equal(landed, research.reconstruction)
                and sum_wr == research.sum_wr
                and norm_wr == research.norm_wr)
        if not same:
            failures += 1
            print('MOVED      %s.%s.%s coarse=%s cap=%d max|d|=%.3e' % (
                minimizer, filter_kw, operator, coarse, cap,
                np.max(np.abs(landed - research.reconstruction))))


for cap in CAPS:
    for minimizer, filter_kw, operator, coarse in CELLS:
        compare(minimizer, filter_kw, operator, coarse, cap)

print()
print('%-20s %-5s %-10s %-7s %-5s %-12s %-12s %-10s'
      % ('minimizer', 'filt', 'operator', 'coarse', 'cap', 'first_norm_wr',
         'final_norm_wr', 'growth'))
for minimizer, filter_kw, operator, coarse, cap, start, final, ratio, steps \
        in departures:
    print('%-20s %-5s %-10s %-7s %-5d %-12.5g %-12.5g x%-10.4g'
          % (minimizer, filter_kw, operator or '-', coarse, cap, start, final,
             ratio))

print()
print('CELLS:', len(CELLS))
print('CAPS:', len(CAPS))
print('RUNS:', runs)
print('DEPARTURES:', len(departures))
print('FAILURES:', failures)

sys.exit(1 if failures else 0)
