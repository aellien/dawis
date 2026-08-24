"""Deprecated location of multiscale detection and deblending.

The code moved to :mod:`dawis.detection.deblending` (#42). This module
re-exports it, along with the names the old module pulled in — photutils'
``detect_sources`` and ``deblend_sources``, astropy's ``sigma_clip``, the
label cube and the noise table — so a caller importing any name from here
still gets it. It is deleted by the contract ticket (#50) once none remain.

The ``plt`` this module used to leak is gone with #46: matplotlib now lives
only in :mod:`dawis.viz`, and mirroring it here would make the optional
``[viz]`` extra mandatory again for anyone who imports ``dawis``.
"""

import logging  # noqa: F401

import numpy as np  # noqa: F401
from astropy.stats import sigma_clip  # noqa: F401
from photutils.segmentation import deblend_sources, detect_sources  # noqa: F401

from dawis.cubes import label_datacube  # noqa: F401
from dawis.detection.deblending import ms_detect_and_deblend  # noqa: F401
from dawis.detection.noise import table_noise_stds  # noqa: F401
