"""Deprecated location of the synthesis-by-analysis pipeline.

The code moved to :mod:`dawis.pipelines.synthesis_by_analysis` (#49). This
module re-exports the entry point, along with every name the old module bound
into its own namespace — the standard-library imports, the block names it
called, and the four star imports it opened — whether or not the pipeline ever
read them. That is what an expand-contract shim is for: a caller that reached
through this module for any of those names still finds it. The whole module is
deleted by the contract ticket (#50) once no caller names it.

Two functions do not come across. ``load_iteration``, marked deprecated where
it stood, could not run: it called ``read_interscale_trees_from_pickle``, a
name that exists nowhere in the package, and read ``.wdc.itNNN.fits`` and
``.ldc.itNNN.fits``, files no version of the pipeline writes. The ``__main__``
block below it called ``main``, which does not exist either; its demo role
belongs to the tutorial notebooks (#12).
"""

import glob as glob  # noqa: F401
import logging  # noqa: F401
import os  # noqa: F401
import pdb  # noqa: F401
import sys as sys  # noqa: F401
from datetime import datetime  # noqa: F401

import numpy as np  # noqa: F401
from astropy.io import fits  # noqa: F401

from dawis.anscombe_transforms import *  # noqa: F401,F403
from dawis.cubes import *  # noqa: F401,F403
from dawis.detection import (  # noqa: F401
    hard_threshold,
    label_regions,
    make_regions_full_props,
    ms_detect_and_deblend,
    region,
)
from dawis.io import (  # noqa: F401
    read_image,
    read_ol_from_hdf5,
    read_ol_from_pickle,
    read_regions_from_pickle,
    write_itl_to_hdf5,
    write_itl_to_pickle,
    write_ol_to_hdf5,
    write_ol_to_pickle,
    write_regions_to_pickle,
)
from dawis.make_interscale_trees import *  # noqa: F401,F403
from dawis.parallel import parallel_backend  # noqa: F401
from dawis.pipelines.synthesis_by_analysis import (  # noqa: F401
    synthesis_by_analysis,
)
from dawis.restoration import (  # noqa: F401
    inpaint_with_gaussian_noise,
    sample_noise,
)
from dawis.restore_objects import *  # noqa: F401,F403
from dawis.wavelet_transforms import bspl_atrous  # noqa: F401
