"""Trees: grouping detections across scales into the unit that gets restored.

The third block carved out of the flat package. It depends on
:mod:`dawis.cubes`, on the regions :mod:`dawis.detection` produces, and on the
à trous transform of :mod:`dawis.wavelets` — all already carved, so nothing
here points backwards. The Richardson-Lucy deconvolution that the synthesis
calls under ``deconv`` moved to :mod:`dawis.restoration` with #44 and is
imported from there. That edge runs one way only: restoration names
:class:`~dawis.trees.construction.interscale_tree` in its annotations alone,
behind ``TYPE_CHECKING``, so the two blocks do not form a cycle.

An **interscale tree** is one structure seen at every scale at once: a root
region on a coarse plane plus the finer regions that belong to the same
object. Trees are what the pipeline restores; one tree becomes one atom.

Two modules split the work along the seam identified by the target
architecture (#4).

:mod:`dawis.trees.construction` holds :class:`interscale_tree` — what a tree
is once its members are known. It fixes the bounding box, cuts the tree out of
the cubes as *tubes*, and synthesises it back into an image with the published
conjugate-gradient solver.

:mod:`dawis.trees.deblending` holds the decision that comes first: which
regions belong to which root, and therefore where one structure stops and the
next begins. :func:`make_interscale_trees` is the entry point the pipeline
calls; :func:`interscale_connectivity_serial` and its ray twin do the
attaching.

The watershed pass that once enforced monomodality inside a tree was deleted
upstream in ``ae19cb2``, well before this carve; its dead ``monomodality`` and
``threshold_rel`` knobs are dropped here.

The old flat module ``dawis.make_interscale_trees`` re-exports every name
below and is deleted once no caller names it.
"""

from dawis.trees.construction import interscale_tree
from dawis.trees.deblending import (
    interscale_connectivity_para,
    interscale_connectivity_serial,
    make_interscale_trees,
)

__all__ = [
    'interscale_connectivity_para',
    'interscale_connectivity_serial',
    'interscale_tree',
    'make_interscale_trees',
]
