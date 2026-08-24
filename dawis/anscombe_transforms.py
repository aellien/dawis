"""Deprecated location of the Anscombe transform.

The code moved to :mod:`dawis.wavelets.anscombe` (#40). This module
re-exports it, along with the ``sigma_clip`` the old module pulled in from
astropy, so a caller importing any name from here still gets it. It is
deleted by the contract ticket (#50) once none remain.
"""

from astropy.stats import sigma_clip  # noqa: F401

from dawis.wavelets.anscombe import (  # noqa: F401
    anscombe_transform,
    inverse_anscombe,
    pg_noise_bissection,
)
