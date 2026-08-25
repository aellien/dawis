"""Pipelines: compositions of the blocks, and the entry points users run.

The last block carved out of the flat package (#49), and the one the layout is
arranged around. Everything else in DAWIS is a building block that can be run
on its own; a pipeline is a particular composition of those blocks into a
method, and :func:`synthesis_by_analysis` is the composition the science is
done with.

The rule the layout rests on runs one way: **pipelines depend on blocks, blocks
never depend on pipelines.** A block that imported this package would be a
block that cannot be used without the whole solver behind it, and the toolbox
of #4 would collapse back into the flat package it came from. The rule is
checkable, and #59 makes it a test.

That is also why this package is thin. It holds the order the steps run in, the
conditions the loop stops under, and the files a run leaves behind — no
mathematics. Every number a pipeline produces is produced by a call into a
block, so a science change is a change to a block and never to a composition.

:mod:`dawis.pipelines.synthesis_by_analysis` holds the one pipeline there is
today. ``dawis.run`` and ``run_fits`` (#54) become the array-in front door to
it, and this is where they land.

The old flat module ``dawis.synthesis_by_analysis`` re-exports the entry point
and is deleted by the contract ticket (#50) once no caller names it.

Note the deliberate shadowing below: the submodule and the function it holds
share a name, and the import binds the function, which is the name a caller
means by ``dawis.pipelines.synthesis_by_analysis``. Reach the module itself,
if you ever need it, through ``sys.modules``.
"""

from dawis.pipelines.synthesis_by_analysis import synthesis_by_analysis

__all__ = ['synthesis_by_analysis']
