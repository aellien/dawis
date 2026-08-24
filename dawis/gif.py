"""Deprecated location of the run animation.

The code moved to :mod:`dawis.viz` (#46): the frame and the ``ffmpeg`` call to
:mod:`dawis.viz.animation`, the convergence trace to
:mod:`dawis.viz.convergence`. This module re-exports the three surviving
names. It is deleted by the contract ticket (#50) once no caller names it.

Two things did not travel.

``remake_frames`` is replaced by :func:`dawis.viz.animation.rebuild_animation`
rather than moved. It was the same idea — redraw the frames of a finished run
— but it could not run: it called ``read_objects_from_pickle``, a name that
exists nowhere in the package, and read it from ``.frame.itNNN.pkl``, a file
the pipeline never wrote. The replacement reads the atom dumps the run really
leaves behind, and recovers the image the loop worked on from the run's own
restored and residual images instead of asking the caller for the input file
and the value of ``gamma``.

Unlike every earlier shim, this module is **not** imported by
:mod:`dawis.__init__`, and its names are no longer bound on the ``dawis``
namespace. Importing it imports matplotlib, and the point of #46 is that
``import dawis`` does not. Reach for the animation through
:mod:`dawis.viz`::

    from dawis.viz import rebuild_animation
"""

from dawis.viz.animation import make_gif, plot_frame  # noqa: F401
from dawis.viz.convergence import plot_convergence  # noqa: F401
