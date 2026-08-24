"""Deprecated location of atom synthesis.

The code moved to :mod:`dawis.restoration.synthesis` (#44). This module
re-exports all four names, along with everything the old module leaked into
its own namespace — including imports that nothing in the moved code reads,
kept because a star import from here bound them into the ``dawis`` namespace.
It is deleted by the contract ticket (#50) once no caller names it.

The ``__main__`` demo block goes with the move; its role belongs to the
tutorial notebooks (#12).

The ``plt`` this module used to leak went with #46, matplotlib now living only
in :mod:`dawis.viz`, and the ``ray`` beside it went with #47: ray is named
nowhere outside :mod:`dawis.parallel`, so binding it here would undo the seam
the moment ``import dawis`` ran.
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
