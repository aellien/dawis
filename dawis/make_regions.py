"""Deprecated location of region property construction.

The code moved to :mod:`dawis.detection.regions` (#42), the two pickle
helpers on to :mod:`dawis.io.legacy` (#45), and ``region.plot`` on to
:func:`dawis.viz.regions.plot_region` (#46). This module re-exports what
remains importable, along with the names the old module leaked into its own
namespace — skimage's ``regionprops`` and the noise table — so a caller
importing any name from here still gets it. It is deleted by the contract
ticket (#50) once none remain.

The two matplotlib names this module used to leak, ``plt`` and
``AnchoredText``, are gone with #46 and are not mirrored here: matplotlib is
the optional ``[viz]`` extra, and binding it in a module that ``import dawis``
reaches would make the extra mandatory again. They live in
:mod:`dawis.viz.regions`.
"""

import logging  # noqa: F401
import pdb  # noqa: F401
import pickle  # noqa: F401
from datetime import datetime  # noqa: F401

import numpy as np  # noqa: F401
from skimage.measure import regionprops  # noqa: F401

from dawis.detection.noise import table_noise_stds  # noqa: F401
from dawis.detection.regions import (  # noqa: F401
    make_regions_crucial_props,
    make_regions_full_props,
    region,
)
# The two pickle helpers travelled on to :mod:`dawis.io.legacy` with #45.
from dawis.io.legacy import (  # noqa: F401
    read_regions_from_pickle,
    write_regions_to_pickle,
)
