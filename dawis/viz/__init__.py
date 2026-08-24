"""Viz: every line of matplotlib in DAWIS, and nothing else.

The sixth block carved out of the flat package, and the only one no other
block imports. Everything here draws; nothing here computes. That is the
whole rule the block exists to make executable: after #46, no module outside
:mod:`dawis.viz` imports matplotlib, so the ``[viz]`` extra is genuinely
optional and ``import dawis`` never pulls a plotting library into a headless
run.

Because of that rule this package is *not* imported by :mod:`dawis`. Import
it explicitly, and only where a picture is wanted::

    import dawis.viz as viz
    viz.rebuild_animation('/data/run/image')

Five modules, split by what they draw.

:mod:`dawis.viz.cubes` draws cubes. The four display methods that used to
hang off the cube classes are functions here, and the two versions of the
plane grid — the plain one and the symmetric-log one that wavelet
coefficients need — are one :func:`plot_cube` that dispatches on the type of
the cube it is given, so a caller no longer has to know which class it holds.

:mod:`dawis.viz.regions` and :mod:`dawis.viz.trees` draw one detection and
one interscale tree, replacing ``region.plot`` and ``interscale_tree.plot``.

:mod:`dawis.viz.animation` is the run animation. It is the reason this ticket
is more than a move: the pipeline used to write one frame per iteration from
inside the greedy loop, which #5 measured at 52% of the loop's time.
:func:`rebuild_animation` instead rebuilds the whole animation *after* a run,
from the atoms the run already stored, so the cost is gone from the loop
architecturally rather than being hidden behind a flag — and an animation can
be made from any finished run, including one that has been sitting on disk
for a year.

:mod:`dawis.viz.convergence` draws the convergence trace, from the
``.cpar.txt`` every run writes.

The old flat module ``dawis.gif`` re-exports the animation names and is
deleted by the contract ticket (#50).
"""

import logging

# matplotlib's font manager logs a handful of DEBUG lines the first time it is
# imported, and a DAWIS run configures the root logger at DEBUG — so quieten it
# here, before the imports below pull matplotlib in. ``dawis.gif`` did the same,
# but after its own import, which was too late to have any effect.
logging.getLogger('matplotlib').setLevel(logging.WARNING)

from dawis.viz.animation import make_gif, plot_frame, rebuild_animation  # noqa: E402
from dawis.viz.convergence import (  # noqa: E402
    plot_convergence,
    rebuild_convergence_plot,
)
from dawis.viz.cubes import (  # noqa: E402
    plot_cube,
    plot_label_counts,
    plot_noise_histogram,
    plot_wavelet_stds,
)
from dawis.viz.regions import plot_region  # noqa: E402
from dawis.viz.trees import plot_interscale_tree  # noqa: E402

__all__ = [
    'make_gif',
    'plot_convergence',
    'plot_cube',
    'plot_frame',
    'plot_interscale_tree',
    'plot_label_counts',
    'plot_noise_histogram',
    'plot_region',
    'plot_wavelet_stds',
    'rebuild_animation',
    'rebuild_convergence_plot',
]
