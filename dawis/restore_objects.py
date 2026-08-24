"""Deprecated location of atom synthesis.

The code moved to :mod:`dawis.restoration.synthesis` (#44). This module
re-exports all four names, along with everything the old module leaked into
its own namespace — including imports that nothing in the moved code reads,
kept because a star import from here bound them into the ``dawis`` namespace.
It is deleted by the contract ticket (#50) once no caller names it.

The ``__main__`` demo block goes with the move; its role belongs to the
tutorial notebooks (#12).
"""

import gc  # noqa: F401
import logging  # noqa: F401
import pdb  # noqa: F401
import _pickle as pickle  # noqa: F401
from datetime import datetime  # noqa: F401

import numpy as np  # noqa: F401

from dawis.restoration.synthesis import (  # noqa: F401
    restore_object,
    restore_objects_default,
    restore_patch,
    restored_object,
)

# Mirrored rather than re-exported from :mod:`dawis.restoration.synthesis`,
# because ``MissingRay`` is bound there only when the extra is absent and this
# module leaked both names into ``dawis`` when it was. ``plt`` is mirrored for
# the same reason: the moved code never used it, so it does not travel.
try:
    import matplotlib.pyplot as plt  # noqa: F401
except ImportError:  # matplotlib is the optional [viz] extra
    from dawis._optional import MissingDependency  # noqa: F401
    plt = MissingDependency('matplotlib', 'viz')
try:
    import ray  # noqa: F401
except ImportError:  # ray is the optional [ray] extra
    from dawis._optional import MissingRay  # noqa: F401
    ray = MissingRay()
