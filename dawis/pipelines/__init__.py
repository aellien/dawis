"""Pipelines: compositions of the blocks, and the entry points users run.

The last block carved out of the flat package (#49), and the one the layout is
arranged around. Everything else in DAWIS is a building block that can be run
on its own; a pipeline is a particular composition of those blocks into a
method, and synthesis by analysis is the composition the science is done with.

The rule the layout rests on runs one way: **pipelines depend on blocks, blocks
never depend on pipelines.** A block that imported this package would be a
block that cannot be used without the whole solver behind it, and the toolbox
of #4 would collapse back into the flat package it came from. The rule is
checkable, and #58 makes it a test, in
``tests/unit/test_import_graph.py``.

That is also why this package is thin. It holds the order the steps run in, the
conditions the loop stops under, and the files a run leaves behind — no
mathematics. Every number a pipeline produces is produced by a call into a
block, so a science change is a change to a block and never to a composition.

:mod:`dawis.pipelines.synthesis_by_analysis` holds the one pipeline there is
today, and the two functions this package exists to expose:

``run(image, config)``
    The front door. An array in, a :class:`~dawis.results.RunResult` out, and
    no path anywhere below it (#54).
``run_fits(path, config)``
    The thin wrapper. It opens the FITS file and delegates to ``run``. It is
    the only place in DAWIS that knows the format the image arrived in.

Those two names replaced ``synthesis_by_analysis``, the fifteen-keyword entry
point the flat package shipped, in #54. The module keeps the name because the
module is what the *pipeline* is called; ``run`` is what *calling it* is
called. When a second composition lands it becomes a second module here, and
``run`` is where the choice between them goes.
"""

from dawis.pipelines.synthesis_by_analysis import run, run_fits

__all__ = ['run', 'run_fits']
