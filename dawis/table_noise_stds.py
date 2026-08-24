"""Deprecated location of the wavelet noise conversion table.

The code moved to :mod:`dawis.detection.noise` (#42). This module re-exports
it, along with the exception the old module pulled in, so a caller importing
any name from here still gets it. It is deleted by the contract ticket (#50)
once none remain.
"""

import numpy as np  # noqa: F401

from dawis.exceptions import DawisUnknownType  # noqa: F401

from dawis.detection.noise import table_noise_stds  # noqa: F401
