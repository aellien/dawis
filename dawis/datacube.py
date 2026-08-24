"""Deprecated location of the cube classes.

The code moved to :mod:`dawis.cubes` (#41) — the cubes are shared vocabulary,
used by every block, so they sit at the top level next to the configuration
and parallelism modules rather than inside any one subpackage. This module
re-exports every name the old module put in its namespace, so a caller
importing any name from here still gets it. It is deleted by the contract
ticket (#50) once none remain.

Pure motion: no class and no attribute is renamed here. The rename to the
``WaveletCube`` vocabulary is its own later ticket.
"""

from dawis.cubes import *  # noqa: F401,F403
