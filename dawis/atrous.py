"""Deprecated location of the à trous transform.

The code moved to :mod:`dawis.wavelets.atrous` (#40). This module re-exports
it, along with the exception family the old module pulled in, so a caller
importing any name from here still gets it. It is deleted by the contract
ticket (#50) once none remain.
"""

from dawis.exceptions import *  # noqa: F401,F403
from dawis.wavelets.atrous import (  # noqa: F401
    atrous,
    boundary_conditions,
    ser_a_trous,
    ser_iuwt_decomposition,
)
