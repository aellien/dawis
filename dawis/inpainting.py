"""Deprecated location of the residual inpainting.

The code moved to :mod:`dawis.restoration.inpainting` (#44). This module
re-exports both names, along with everything the old module bound into its own
namespace — including the unused ``logging`` import, kept because this
module's namespace is part of what callers could reach. It is deleted by the
contract ticket (#50) once no caller names it.
"""

import logging  # noqa: F401

import numpy as np  # noqa: F401
from astropy.stats import sigma_clip  # noqa: F401
from skimage.feature import peak_local_max  # noqa: F401

from dawis.restoration.inpainting import (  # noqa: F401
    inpaint_with_gaussian_noise,
    sample_noise,
)
