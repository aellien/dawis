"""The simulation gallery generator (spec #67).

A hand-run tool, a sibling of ``benchmarks/`` and not part of the distributed
package. It produces the synthetic FITS images with exact ground truth that
the restoration battery of #25 measures against.

The generator runs in its own conda environment, ``dawis-sim``, because it
needs GalSim and DAWIS deliberately does not. See ``environment.yml`` and
``README.md``.

Three layers, and only the first is a test seam:

``recipe``
    Pure description of one entry — grid, sources, seed, noise, paths,
    manifest. Imports numpy only. Every decision of #67 lives here.
``render``
    Hands a recipe to GalSim and returns arrays. Thin.
``write``
    Arrays to FITS, CSV and JSON on disk, including the idempotent skip. Thin.

Importing this package imports nothing but the standard library, so a test in
the ordinary DAWIS environment can reach ``simulations.recipe`` without GalSim
being installed.
"""

from __future__ import annotations

__all__ = []
