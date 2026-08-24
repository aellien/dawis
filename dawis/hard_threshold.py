"""Deprecated location of hard thresholding.

The code moved to :mod:`dawis.detection.thresholding` (#42). This module
re-exports it, along with the cube namespace the old module pulled in with a
star import and the noise table it called, so a caller importing any name
from here still gets it. It is deleted by the contract ticket (#50) once none
remain.
"""

import logging  # noqa: F401

from scipy.stats import normaltest, sigmaclip  # noqa: F401

from dawis.cubes import *  # noqa: F401,F403

from dawis.detection.noise import table_noise_stds  # noqa: F401
from dawis.detection.thresholding import hard_threshold  # noqa: F401
