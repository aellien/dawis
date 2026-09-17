"""Trees: grouping detections across scales into the unit that gets restored.

The third block carved out of the flat package. It depends on
:mod:`dawis.cubes`, on the regions :mod:`dawis.detection` produces, and on the
à trous transform of :mod:`dawis.wavelets` — all already carved, so nothing
here points backwards. Nothing in :mod:`dawis.restoration` is imported either,
since #101 removed the Richardson-Lucy step this block used to call. The edge
between the two blocks runs one way only: restoration names
:class:`~dawis.trees.construction.InterscaleTree` in its annotations alone,
behind ``TYPE_CHECKING``, so they do not form a cycle.

An **interscale tree** is one structure seen at every scale at once: a root
region on a coarse plane plus the finer regions that belong to the same
object. Trees are what the pipeline restores; one tree becomes one atom.

Three modules split the work along the seam identified by the target
architecture (#4).

:mod:`dawis.trees.construction` holds :class:`InterscaleTree` — what a tree
is once its members are known. It fixes the bounding box, cuts the tree out of
the cubes as *tubes*, and synthesises it back into an image with the published
conjugate-gradient solver.

:mod:`dawis.trees.deblending` holds the decision that comes first: which
regions belong to which root, and therefore where one structure stops and the
next begins. :func:`make_interscale_trees` is the entry point the pipeline
calls; :func:`interscale_connectivity` does the attaching. The two copies of
that function — one plain, one under ``@ray.remote`` — collapsed into this one
when the parallelism seam landed (#47), and where a patch of roots runs is now
:mod:`dawis.parallel`'s business rather than this block's.

:mod:`dawis.trees.minimizers` holds the choice of *which* algorithm does that
synthesis — the minimizer axis (#102). :func:`minimize` is the one door over
all four: the published conjugate gradient, and the three proximal arms #74
measured against it. It depends on :mod:`dawis.trees.construction` and nothing
there depends back on it, so the published solve stays readable on its own.

The watershed pass that once enforced monomodality inside a tree was deleted
upstream in ``ae19cb2``, well before this carve; its dead ``monomodality`` and
``threshold_rel`` knobs are dropped here.

The old flat module ``dawis.make_interscale_trees`` held these names until
#50 deleted it. This package is the only address they have.
"""

from dawis.trees.construction import InterscaleTree
from dawis.trees.deblending import (
    interscale_connectivity,
    make_interscale_trees,
)
from dawis.trees.minimizers import minimize

__all__ = [
    'InterscaleTree',
    'interscale_connectivity',
    'make_interscale_trees',
    'minimize',
]
