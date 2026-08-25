"""Restoration: rebuilding detected structures as images and subtracting them.

The fourth block carved out of the flat package, and the one the run time
lives in — #5 measured this block at 98.4% of the loop.

At runtime it imports only :mod:`dawis.cubes`. It consumes the trees that
:mod:`dawis.trees` builds, but takes them as arguments rather than importing
the package. The reference to the tree class in the annotations of
:mod:`dawis.restoration.synthesis` sits behind ``TYPE_CHECKING``. That is
deliberate: the edge runs the other way — the solver on the tree calls the
deconvolution here — so importing trees for real would close a cycle.

Restoration is the *synthesis* half of synthesis by analysis. Detection and
tree building say where a structure is and which scales it spans; this block
turns that into an image, and the pipeline subtracts it from the running
residual before analysing what is left.

Four modules.

:mod:`dawis.restoration.synthesis` is the entry point.
:func:`restore_objects_default` walks the trees of one iteration —
in process, or over workers, as :mod:`dawis.parallel` decides — and each
becomes an :class:`Atom`. :func:`restore_object` makes the per-tree
choice of filter and synthesis operator that the solver then runs with.

:mod:`dawis.restoration.deconvolution` holds the optional PSF step inside that
solve: Richardson-Lucy against the à trous B3-spline kernel that
:func:`twod_B2spline` builds, at one of two orders picked from the level of
the interscale maximum. Only the ADJOINT branch of the solver reads it.

:mod:`dawis.restoration.inpainting` closes the loop. A slightly over-restored
atom digs a hole in the residual, and the next iteration would detect that
hole as structure, so the residual is refilled with noise measured from
itself before the next pass. :func:`subtract_background` measures the same
noise once, at the start of a run, and levels the image on it — everything
downstream assumes the noise is centred on zero.

:mod:`dawis.restoration.accumulation` holds the running images themselves.
:class:`Reconstruction` is the residual, the restored image, the restored
image split by wavelet scale and the accumulated detection error, together
with the one operation that advances all four: subtract an iteration's atoms
from the first and add them to the rest. It arrived here with #49, which took
that loop out of the pipeline: the pipeline says *when* atoms are subtracted,
and this says what subtracting them means.

The conjugate-gradient solver is not here, and #44 asked for it. It lives on
the tree, as
:meth:`~dawis.trees.construction.interscale_tree.CG_minimization`, where #43
carved it, because it works on the tubes the tree cuts out of the cubes.
Prising a method off a class is a change of shape rather than a move, and #44
is explicit that this block changes by motion only, on 98.4% of the loop. So
the deviation is deliberate and it is a deviation: what belongs here is the
*choice* of how to call the solver, and that is what this block holds. Moving
the solver itself waits for the minimizer interface of #22.

The old flat modules ``dawis.restore_objects``, ``dawis.deconvolution`` and
``dawis.inpainting`` held these names until #50 deleted them. This package is
the only address they have.
"""

from dawis.restoration.accumulation import Reconstruction
from dawis.restoration.deconvolution import (
    B2spline,
    pad,
    richardson_lucy_np,
    twod_B2spline,
    unpad,
)
from dawis.restoration.inpainting import (
    inpaint_with_gaussian_noise,
    sample_noise,
    subtract_background,
)
from dawis.restoration.synthesis import (
    Atom,
    restore_object,
    restore_objects_default,
    restore_patch,
)

__all__ = [
    'Atom',
    'B2spline',
    'Reconstruction',
    'inpaint_with_gaussian_noise',
    'pad',
    'restore_object',
    'restore_objects_default',
    'restore_patch',
    'richardson_lucy_np',
    'sample_noise',
    'subtract_background',
    'twod_B2spline',
    'unpad',
]
