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
TINY_PARAMS = dict(max_iter = 2, data_dump = False, resume = False)

#: What a tiny run calls the files it leaves behind.
TINY_NAME = 'tiny'


def tiny_image(seed: int = 7) -> np.ndarray:
    """A 64x64 noise frame with one source bright enough to be restored."""
    rng = np.random.default_rng(seed)
    image = rng.normal(100.0, 5.0, (64, 64)).astype('float32')
    image[28:36, 28:36] += 300.0
    return image


def run_quiet(*args, **kwargs) -> 'RunResult':
    """Call ``dawis.run`` without letting it steal pytest's log handlers.

    The pipeline attaches its own file and stdout handlers through
    ``logging.basicConfig``, which is a no-op once the root logger has any —
    and both pytest and the previous run leave some there. Without this a run
    would write its log, and therefore its seed, into whatever file was
    attached before it. One run per process is the real-world case, so this is
    a test-only inconvenience rather than a defect in the run; #55 owns how the
    pipeline attaches its handlers.
    """
    import logging

    import dawis

    root = logging.getLogger()
    saved = list(root.handlers)
    root.handlers.clear()
    try:
        return dawis.run(*args, **kwargs)
    finally:
        for handler in root.handlers:
            handler.close()
        root.handlers[:] = saved
