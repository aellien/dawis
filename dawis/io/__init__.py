"""IO: everything DAWIS reads from and writes to disk.

The fifth block carved out of the flat package. Its edges all point at blocks
already carved — reading rebuilds atoms, trees and regions, so it imports
:mod:`dawis.restoration`, :mod:`dawis.trees` and :mod:`dawis.detection` — and
they run one way: none of the three imports :mod:`dawis.io`.

Three modules, split by format rather than by direction, because a format is
what a file is compatible with.

:mod:`dawis.io.store` is the atom store: one HDF5 file per run, every atom's
pixels in one long array with one index row per atom (#122). The pipeline
writes it once per iteration and a resumed run reads it back.
:func:`read_image_atoms` is the run-level read; :func:`read_atom_index` reads
the index alone.

:mod:`dawis.io.legacy` holds the two formats DAWIS wrote before 1.0 —
per-iteration pickles and per-iteration HDF5 files — and
:func:`read_legacy_run` reads such a run. It stays because the published
``v0.0.7-h`` outputs are recorded as pickles and the regression harness (#3)
reads its goldens through it. A 1.0 run cannot resume from these files.

:mod:`dawis.io.fits` is the boundary the image itself comes across, kept thin
so that the entry point of #54 is the only caller that knows about paths.
:func:`read_image` reads the first HDU that holds an image, in file order,
unless the caller names an index (#65) — a metadata-only primary with the
pixels in an extension is the normal shape of survey data.

The region pickle helpers ``write_regions_to_pickle`` and
``read_regions_from_pickle`` travelled here from
:mod:`dawis.detection.regions` in the same commit, which is what #42's note
on them said would happen. They sit in :mod:`dawis.io.legacy`.

The old flat module ``dawis.store_objects`` held fourteen of the names below
until #50 deleted it. :func:`read_image` is new here, the two region helpers
came from ``dawis.make_regions``, and the atom store's four names from #122.

Pickles written before the carve name the flat modules, so
:mod:`dawis.io.legacy` maps those paths onto the carved classes when it
reads. Deleting the shims is what made that map necessary.
"""

from dawis.io.fits import read_image
from dawis.io.legacy import (
    load_atom_from_hdf5_group,
    load_interscale_tree_from_hdf5_group,
    load_region_from_hdf5_group,
    object_recursive_save_to_hdf5,
    pkl_to_hdf5,
    read_itl_from_hdf5,
    read_itl_from_pickle,
    read_legacy_run,
    read_ol_from_hdf5,
    read_ol_from_pickle,
    read_regions_from_pickle,
    write_itl_to_hdf5,
    write_itl_to_pickle,
    write_ol_to_hdf5,
    write_ol_to_pickle,
    write_regions_to_pickle,
)
from dawis.io.store import (
    AtomStore,
    read_atom_index,
    read_frame_shape,
    read_image_atoms,
    read_synthesised_image,
)

__all__ = [
    'AtomStore',
    'load_atom_from_hdf5_group',
    'load_interscale_tree_from_hdf5_group',
    'load_region_from_hdf5_group',
    'object_recursive_save_to_hdf5',
    'pkl_to_hdf5',
    'read_image',
    'read_atom_index',
    'read_frame_shape',
    'read_image_atoms',
    'read_synthesised_image',
    'read_itl_from_hdf5',
    'read_itl_from_pickle',
    'read_legacy_run',
    'read_ol_from_hdf5',
    'read_ol_from_pickle',
    'read_regions_from_pickle',
    'write_itl_to_hdf5',
    'write_itl_to_pickle',
    'write_ol_to_hdf5',
    'write_ol_to_pickle',
    'write_regions_to_pickle',
]
