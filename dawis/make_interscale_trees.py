"""Deprecated location of interscale trees.

The code moved to :mod:`dawis.trees` (#43): the :class:`interscale_tree` class
to :mod:`dawis.trees.construction`, and the connectivity and tree-building
functions to :mod:`dawis.trees.deblending`. This module re-exports all four
names, along with everything the old module leaked into its own namespace —
including imports that the demolition ticket (#38) left unused, kept because a
star import from here bound them into the ``dawis`` namespace. It is deleted
by the contract ticket (#50) once no caller names it.

The ``__main__`` demo block goes with the move; its role belongs to the
tutorial notebooks (#12).

The two matplotlib names this module used to leak, ``plt`` and
``SymLogNorm``, are gone with #46 along with ``interscale_tree.plot``, which
is now :func:`dawis.viz.trees.plot_interscale_tree`. They are not mirrored
here: matplotlib is the optional ``[viz]`` extra, and binding it in a module
that ``import dawis`` reaches would make the extra mandatory again.
"""

import gc  # noqa: F401
import logging  # noqa: F401
import pdb  # noqa: F401
import _pickle as pickle  # noqa: F401
from datetime import datetime  # noqa: F401

import numpy as np  # noqa: F401
from scipy import ndimage as ndi  # noqa: F401
from skimage.feature import peak_local_max  # noqa: F401
from skimage.measure import label, regionprops  # noqa: F401
from skimage.segmentation import watershed  # noqa: F401

from dawis.restoration.deconvolution import (  # noqa: F401
    B2spline,
    pad,
    richardson_lucy_np,
    twod_B2spline,
    unpad,
)
from dawis.trees.construction import interscale_tree  # noqa: F401
from dawis.trees.deblending import (  # noqa: F401
    interscale_connectivity_para,
    interscale_connectivity_serial,
    make_interscale_trees,
)
from dawis.wavelet_transforms import bspl_atrous  # noqa: F401
from dawis.wavelets import atrous  # noqa: F401

# Mirrored rather than re-exported from :mod:`dawis.trees.deblending`, because
# ``MissingRay`` is bound there only when the extra is absent and this module
# leaked both names into ``dawis`` when it was.
try:
    import ray  # noqa: F401
except ImportError:  # ray is the optional [ray] extra
    from dawis._optional import MissingRay  # noqa: F401
    ray = MissingRay()
