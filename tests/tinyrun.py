"""One whole pipeline run, small enough to sit inside Tier 0's budget.

Two unit modules drive a complete run rather than a piece of one — the seed and
reproducibility tests of #53, and the entry-point tests of #54 — because that
is the only way to assert what their criteria actually say. A run on a 64x64
frame capped at two iterations takes about half a second, so they can.

What they share lives here: the frame, the knobs that keep it small, and the
one awkward thing about calling the pipeline from inside pytest. Nothing here
imports pytest, and nothing here is a fixture — a caller that wants one wraps
it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dawis.results import RunResult

#: Knobs that keep a run small. Everything else takes its default, which is
#: the point: the defaults have to survive a small image.
TINY_PARAMS = dict(max_iter = 2, resume = False)

#: What a tiny run calls the files it leaves behind.
TINY_NAME = 'tiny'


def tiny_image(seed: int = 7) -> np.ndarray:
    """A 64x64 noise frame with one source bright enough to be restored."""
    rng = np.random.default_rng(seed)
    image = rng.normal(100.0, 5.0, (64, 64)).astype('float32')
    image[28:36, 28:36] += 300.0
    return image


def run_quiet(*args, **kwargs) -> 'RunResult':
    """Call ``dawis.run`` with its stdout handler suppressed.

    A run that finds no logging configured attaches a stdout handler, so that
    the default experience of DAWIS is not twenty silent minutes (#55). Under
    pytest that is noise on every test that drives a run, and the run's own
    log file is what these tests read anyway.

    Installing a handler that discards *is* "the caller configured logging",
    so the run adds no second stream. From then on the caller's level is the
    one that decides what reaches the run's log file, which is why this sets
    it explicitly rather than leaving it to be inherited. ``propagate`` goes
    off so the records do not land in pytest's capture either.
    """
    import logging

    import dawis
    from dawis.logs import PACKAGE_LOGGER

    package = logging.getLogger(PACKAGE_LOGGER)
    saved = (list(package.handlers), package.level, package.propagate)
    package.handlers = [logging.NullHandler()]
    package.setLevel(logging.DEBUG)
    package.propagate = False
    try:
        return dawis.run(*args, **kwargs)
    finally:
        package.handlers, package.level, package.propagate = saved
