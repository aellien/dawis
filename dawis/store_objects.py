"""Deprecated location of the object store.

The code moved to :mod:`dawis.io` (#45): the HDF5 store to
:mod:`dawis.io.store`, the pickle format to :mod:`dawis.io.legacy`. This
module re-exports all fourteen names, along with everything the old module
leaked into its own namespace — including imports that nothing in the moved
code reads, kept because a star import from here bound them into the
``dawis`` namespace. It is deleted by the contract ticket (#50) once no
caller names it.

The ``__main__`` block, a two-argument pickle-to-HDF5 converter, travels
verbatim to :mod:`dawis.io.store`. Unlike the demo blocks the earlier carves
deleted, it works and has no replacement until the Phase B command line, so
losing it here would be a capability lost to a move.
"""

import gc  # noqa: F401
import glob  # noqa: F401
import logging  # noqa: F401
import os  # noqa: F401
import pickle  # noqa: F401
import sys  # noqa: F401

import h5py  # noqa: F401
import numpy as np  # noqa: F401

from dawis.detection import region  # noqa: F401
from dawis.restoration import restored_object  # noqa: F401
from dawis.trees import interscale_tree  # noqa: F401

from dawis.io.legacy import (  # noqa: F401
    read_itl_from_pickle,
    read_ol_from_pickle,
    write_itl_to_pickle,
    write_ol_to_pickle,
)
from dawis.io.store import (  # noqa: F401
    load_interscale_tree_from_hdf5_group,
    load_region_from_hdf5_group,
    load_restored_object_from_hdf5_group,
    object_recursive_save_to_hdf5,
    pkl_to_hdf5,
    read_image_atoms,
    read_itl_from_hdf5,
    read_ol_from_hdf5,
    write_itl_to_hdf5,
    write_ol_to_hdf5,
)
