"""Wavelet analysis: the à trous transform and variance stabilisation.

The first block carved out of the flat package, and a leaf: it depends on
nothing else in DAWIS except the cube containers and the exception family.

Two things live here. The **transform** decomposes an image into wavelet
planes, one per scale — :func:`bspl_atrous` is the B3-spline entry point the
pipeline runs, :func:`atrous` the general-filter version, and
:func:`ser_iuwt_decomposition` the fast kernel underneath. The
**stabilisation** makes a single noise threshold valid at every flux level:
:func:`pg_noise_bissection` estimates the noise, :func:`anscombe_transform`
applies the map and :func:`inverse_anscombe` undoes it.

The old flat modules ``dawis.atrous``, ``dawis.wavelet_transforms`` and
``dawis.anscombe_transforms`` re-export every name below and are deleted once
no caller names them.
"""

from dawis.wavelets.anscombe import (
    anscombe_transform,
    inverse_anscombe,
    pg_noise_bissection,
)
from dawis.wavelets.atrous import (
    atrous,
    boundary_conditions,
    ser_iuwt_decomposition,
)
from dawis.wavelets.transforms import bspl_atrous, haar_atrous

__all__ = [
    'anscombe_transform',
    'atrous',
    'boundary_conditions',
    'bspl_atrous',
    'haar_atrous',
    'inverse_anscombe',
    'pg_noise_bissection',
    'ser_iuwt_decomposition',
]
