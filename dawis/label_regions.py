"""Deprecated location of region labelling.

The code moved to :mod:`dawis.detection.labelling` (#42). This module
re-exports it, along with the cube namespace the old module pulled in with a
star import and skimage's ``label``, so a caller importing any name from here
still gets it. It is deleted by the contract ticket (#50) once none remain.
"""

import logging  # noqa: F401

from dawis.cubes import *  # noqa: F401,F403
from skimage.measure import label  # noqa: F401

from dawis.detection.labelling import label_regions  # noqa: F401
