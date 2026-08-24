"""Deprecated location of region property construction.

The code moved to :mod:`dawis.detection.regions` (#42). This module
re-exports it, along with the names the old module leaked into its own
namespace — skimage's ``regionprops``, matplotlib's ``AnchoredText``, and the
noise table — so a caller importing any name from here still gets it. It is
deleted by the contract ticket (#50) once none remain.
"""

import logging  # noqa: F401
import pdb  # noqa: F401
import pickle  # noqa: F401
from datetime import datetime  # noqa: F401

import numpy as np  # noqa: F401
from skimage.measure import regionprops  # noqa: F401

from dawis.detection.noise import table_noise_stds  # noqa: F401
from dawis.detection.regions import (  # noqa: F401
    AnchoredText,
    plt,
    make_regions_crucial_props,
    make_regions_full_props,
    read_regions_from_pickle,
    region,
    write_regions_to_pickle,
)
