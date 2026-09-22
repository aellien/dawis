"""#110 part A — which of #81's 540 verdicts the repaired guard moves.

#110 makes ``dawis/``'s proximal arms test the iterate they *return*. The
loop's own guard reads a residual measured before the step, so iterate *k* is
only scored at iteration *k + 1*, and a loop that ends on ``max_iter`` or on
the flux rule has no iteration *k + 1*: the last iterate is the one nothing
scored. That is a **departure** from
``restoration_tests/arms/proximal.py``, which keeps the hole and stays the
record of what every #74, #81 and #92 number was taken with — so the
departure owes an answer to *what moves*, and #81's 540 proximal rows are the
population it can move.

The repaired guard flags a solve iff

    not diverged  and  norm_wr > DIVERGENCE_FACTOR * first_norm_wr

``norm_wr`` is the returned iterate's residual, which #81 saved in
``results.csv``. ``first_norm_wr`` is the residual at the iterate the solve
*started* from, which #81 did not save — but it costs one step, not two
hundred: the initial iterate is ``clip(synthesise(w), 0, None)`` and depends
on neither the step length nor the iteration count. So every row is settled by
re-running its own arm on its own fixture with ``max_iter = 1`` and one power
iteration, and reading ``records[0].norm_wr`` — the reference arm's own
number, not a reimplementation of it.

The rows #81 already flagged are skipped rather than re-scored: the repair
adds a test to a solve that reached ``_finish`` unflagged, so it can turn a
returned atom into a flagged one and never the other way.

Run from a checkout of ``research/restoration-tests``:

    python verdict_shift.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from restoration_tests.arms.grid import ArmSpec
from restoration_tests.arms.proximal import (
    DEFAULT_DIVERGENCE_FACTOR,
    condat_vu,
    fista,
    projected_gradient,
)
from restoration_tests.fixtures.store import read_fixtures

RESULTS = Path.home() / 'dawis-tests/restoration-results-81/results.csv'
FIXTURES = Path.home() / 'dawis-tests/restoration-fixtures'

ARMS = {
    'fista': fista,
    'projected_gradient': projected_gradient,
    'condat_vu': condat_vu,
}


def first_norm_wr(spec: ArmSpec, fixture) -> float:
    """The residual norm the solve starts from, at the cost of one step."""

    records = []
    ARMS[spec.minimizer](
        spec, fixture.wavelet_tube, fixture.support_tube,
        observer = lambda record, image: records.append(record),
        max_iter = 1, n_power_iterations = 1)

    return records[0].norm_wr


def main() -> int:
    rows = list(csv.DictReader(RESULTS.open()))
    print('rows           :', len(rows))
    print('factor         :', DEFAULT_DIVERGENCE_FACTOR)

    by_entry: dict[str, dict] = {}
    moved = []
    kept_diverged = 0
    worst = (0.0, ('', ''))

    for row in rows:
        entry = row['entry']
        if entry not in by_entry:
            by_entry[entry] = {
                fixture.fixture_id: fixture
                for fixture in read_fixtures(FIXTURES / entry)}
        fixture = by_entry[entry][row['fixture_id']]

        if row['diverged'] == 'True':
            kept_diverged += 1
            continue

        spec = ArmSpec(row['minimizer'], row['filter'],
                       row['operator'] or None, deconv = False)
        start = first_norm_wr(spec, fixture)
        final = float(row['norm_wr'])
        ratio = final / start

        if ratio > worst[0]:
            worst = (ratio, (row['arm'], row['fixture_id']))

        if ratio > DEFAULT_DIVERGENCE_FACTOR:
            moved.append((row['arm'], row['fixture_id'], start, final, ratio,
                          int(row['n_iterations']),
                          row['converged'] == 'True'))

    print('already flagged:', kept_diverged)
    print('re-scored      :', len(rows) - kept_diverged)
    print('worst growth   : x%.4g  (%s on %s)'
          % (worst[0], worst[1][0], worst[1][1]))
    print('VERDICTS MOVED :', len(moved))

    for arm, fixture_id, start, final, ratio, iterations, converged in moved:
        print('  %-32s %-40s first %.4g  final %.4g  x%.4g  iters %d  %s'
              % (arm, fixture_id, start, final, ratio, iterations,
                 'flux rule' if converged else 'max_iter'))

    return 0


if __name__ == '__main__':
    sys.exit(main())
