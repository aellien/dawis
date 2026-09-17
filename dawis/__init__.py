"""DAWIS — Detection Algorithm with Wavelets for Intracluster light Studies.

A toolbox of wavelet-based analysis methods (#4).
:func:`run` drives the composition the science is done with; everything it
calls is a building block that can be run on its own.

The layout is one subpackage per block::

    dawis.wavelets      the à trous transform and variance stabilisation
    dawis.detection     thresholds, labels, regions
    dawis.trees         interscale trees
    dawis.restoration   atom synthesis, inpainting
    dawis.selection     atoms into source models (reserved, #17)
    dawis.pipelines     the compositions users run
    dawis.io            FITS, the object store, the legacy pickle format
    dawis.viz           every line of matplotlib

with five modules at the top level, above the blocks because they are not one
block's business: :mod:`dawis.cubes` (the containers), :mod:`dawis.config`
(the knobs of a run), :mod:`dawis.results` (what a run reports) and
:mod:`dawis.exceptions` are shared vocabulary, :mod:`dawis.parallel` is the
parallelism seam, and :mod:`dawis.convergence` is the stopping test the
pipelines share. ``cli`` joins them in Phase B.

:func:`run` takes an array and a :class:`~dawis.config.Config` and returns a
:class:`~dawis.results.RunResult`; :func:`run_fits` opens a FITS file and
delegates to it. Nothing below the pipeline sees a path (#54).

This module re-exports the public names of the blocks so the flat spellings
users already have — ``dawis.bspl_atrous``, ``dawis.hard_threshold`` — keep
working. It is an explicit list: the star imports over fourteen flat modules
that used to stand here also bound every module those modules imported, so
``dawis.np``, ``dawis.pdb`` and ``dawis.h5py`` were part of what a caller
could reach. They are not part of the public surface and they are gone (#50),
along with the flat modules themselves.

Two names leave with them on purpose, and they are the only two that were
bound here deliberately rather than by accident: ``normaltest`` and
``sigmaclip``. Both are :mod:`scipy.stats` functions that
``dawis.hard_threshold`` imported and the carve kept reachable so no caller
broke mid-move. DAWIS neither wrote them nor wraps them, so re-exporting them
would be this package promising scipy's surface as its own. Import them from
``scipy.stats``.

Two things are deliberately not re-exported flat, and are reached through
their module instead. :mod:`dawis.parallel` and :mod:`dawis.convergence` are
machinery rather than vocabulary — ``dawis.parallel.pmap``,
``dawis.convergence.ConvergenceMonitor``. :mod:`dawis.config` is both, and is
split: :class:`~dawis.config.Config` is what a user writes and is re-exported
flat, while the seed machinery around it — ``dawis.config.resolve_seed``,
``dawis.config.make_rng``, ``dawis.config.DEFAULT_CG_MAX_ITER`` — is what a
pipeline calls and stays on the module. And :mod:`dawis.viz` is not
imported at all: importing it imports matplotlib, the optional ``[viz]``
extra, and ``import dawis`` must work without it. Draw things with an
explicit ``import dawis.viz``.
"""

from dawis.config import Config
from dawis.cubes import (
    CoarseCube,
    Cube,
    LabelCube,
    SupportCube,
    WaveletCube,
)
from dawis.detection import (
    hard_threshold,
    label_regions,
    make_regions_crucial_props,
    make_regions_full_props,
    ms_detect_and_deblend,
    region,
    table_noise_stds,
)
from dawis.exceptions import (
    ConfigError,
    DawisDimensionError,
    DawisError,
    DawisValueError,
)
from dawis.io import (
    load_atom_from_hdf5_group,
    load_interscale_tree_from_hdf5_group,
    load_region_from_hdf5_group,
    object_recursive_save_to_hdf5,
    pkl_to_hdf5,
    read_image,
    read_image_atoms,
    read_itl_from_hdf5,
    read_itl_from_pickle,
    read_ol_from_hdf5,
    read_ol_from_pickle,
    read_regions_from_pickle,
    write_itl_to_hdf5,
    write_itl_to_pickle,
    write_ol_to_hdf5,
    write_ol_to_pickle,
    write_regions_to_pickle,
)
from dawis.pipelines import run, run_fits
from dawis.results import InputDiagnostics, RunResult
from dawis.restoration import (
    Atom,
    Reconstruction,
    inpaint_with_gaussian_noise,
    restore_object,
    restore_objects_default,
    restore_patch,
    sample_noise,
    subtract_background,
)
from dawis.trees import (
    InterscaleTree,
    interscale_connectivity,
    make_interscale_trees,
)
from dawis.wavelets import (
    AnscombeStabilisation,
    anscombe_transform,
    atrous,
    boundary_conditions,
    bspl_atrous,
    haar_atrous,
    inverse_anscombe,
    max_levels,
    pg_noise_bissection,
    ser_iuwt_decomposition,
)

__all__ = [
    'AnscombeStabilisation',
    'Atom',
    'CoarseCube',
    'Config',
    'ConfigError',
    'Cube',
    'DawisDimensionError',
    'DawisError',
    'DawisValueError',
    'InputDiagnostics',
    'InterscaleTree',
    'LabelCube',
    'Reconstruction',
    'RunResult',
    'SupportCube',
    'WaveletCube',
    'anscombe_transform',
    'atrous',
    'boundary_conditions',
    'bspl_atrous',
    'haar_atrous',
    'hard_threshold',
    'inpaint_with_gaussian_noise',
    'interscale_connectivity',
    'inverse_anscombe',
    'label_regions',
    'load_atom_from_hdf5_group',
    'load_interscale_tree_from_hdf5_group',
    'load_region_from_hdf5_group',
    'make_interscale_trees',
    'make_regions_crucial_props',
    'make_regions_full_props',
    'max_levels',
    'ms_detect_and_deblend',
    'object_recursive_save_to_hdf5',
    'pg_noise_bissection',
    'pkl_to_hdf5',
    'read_image',
    'read_image_atoms',
    'read_itl_from_hdf5',
    'read_itl_from_pickle',
    'read_ol_from_hdf5',
    'read_ol_from_pickle',
    'read_regions_from_pickle',
    'region',
    'restore_object',
    'restore_objects_default',
    'restore_patch',
    'run',
    'run_fits',
    'sample_noise',
    'ser_iuwt_decomposition',
    'subtract_background',
    'table_noise_stds',
    'write_itl_to_hdf5',
    'write_itl_to_pickle',
    'write_ol_to_hdf5',
    'write_ol_to_pickle',
    'write_regions_to_pickle',
]
