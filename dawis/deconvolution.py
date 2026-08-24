"""Deprecated location of the Richardson-Lucy deconvolution.

The code moved to :mod:`dawis.restoration.deconvolution` (#44). This module
re-exports all five names, along with everything the old module bound into its
own namespace — including the unused ``logging`` and ``fftshift`` imports,
kept because this module's namespace is part of what callers could reach. It
is deleted by the contract ticket (#50) once no caller names it.
"""

import logging  # noqa: F401
import math  # noqa: F401

import numpy as np  # noqa: F401
from numpy.fft import fftn, fftshift, ifftn, ifftshift  # noqa: F401

from dawis.restoration.deconvolution import (  # noqa: F401
    B2spline,
    pad,
    richardson_lucy_np,
    twod_B2spline,
    unpad,
)
