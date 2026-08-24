"""Deprecated location of the wavelet cube entry points.

The code moved to :mod:`dawis.wavelets.transforms` (#40). This module
re-exports it, along with everything the old module put in its namespace —
the à trous names it pulled in from :mod:`dawis.atrous`, and the cube classes
it pulled in from :mod:`dawis.datacube` — so a caller importing any name from
here still gets it. It is deleted by the contract ticket (#50) once none
remain.
"""

from dawis.atrous import (  # noqa: F401
    atrous,
    boundary_conditions,
    ser_iuwt_decomposition,
)
from dawis.datacube import *  # noqa: F401,F403
from dawis.wavelets.transforms import bspl_atrous, haar_atrous  # noqa: F401
