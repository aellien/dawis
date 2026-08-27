"""IO: everything DAWIS reads from and writes to disk.

The fifth block carved out of the flat package. Its edges all point at blocks
already carved — reading rebuilds atoms, trees and regions, so it imports
:mod:`dawis.restoration`, :mod:`dawis.trees` and :mod:`dawis.detection` — and
they run one way: none of the three imports :mod:`dawis.io`.

Three modules, split by format rather than by direction, because a format is
what a file is compatible with.

:mod:`dawis.io.store` is the object store: what one iteration of the pipeline
leaves on disk, and what a resumed run reads back. Its writer is generic — it
mirrors an object's ``__dict__`` into an HDF5 group — so a new attribute on an
atom needs no change here. :func:`read_image_atoms` is the run-level read.

:mod:`dawis.io.legacy` is the pickle format DAWIS wrote before the store. It
stays because the published ``v0.0.7-h`` outputs are recorded in it and the
regression harness (#3) reads its goldens through it; old runs resume through
the same call. It is the leaf of this block — it imports no other block at
run time — so the store can read a legacy run without any edge pointing back.

:mod:`dawis.io.fits` is the boundary the image itself comes across, kept thin
so that the entry point of #54 is the only caller that knows about paths.
:func:`read_image` reads the first HDU that holds an image, in file order,
unless the caller names an index (#65) — a metadata-only primary with the
pixels in an extension is the normal shape of survey data.

The on-disk format is unchanged in Phase A, which is what makes the goldens
diagnostic; the HDF5-backed ``ObjectStore`` proper is Phase B.

The region pickle helpers ``write_regions_to_pickle`` and
``read_regions_from_pickle`` travelled here from
:mod:`dawis.detection.regions` in the same commit, which is what #42's note
on them said would happen. They sit in :mod:`dawis.io.legacy`, which names
:class:`~dawis.detection.regions.region` in an annotation alone, behind
``TYPE_CHECKING``, so the move adds no runtime edge and closes no cycle.

The old flat module ``dawis.store_objects`` held fourteen of the seventeen
names below until #50 deleted it. The other three never lived there:
:func:`read_image` is new here, and the two region helpers came from
``dawis.make_regions``.

Pickles written before the carve name the flat modules, so
:mod:`dawis.io.legacy` maps those paths onto the carved classes when it
reads. Deleting the shims is what made that map necessary.
"""

from dawis.io.fits import read_image
from dawis.io.legacy import (
    read_itl_from_pickle,
    read_ol_from_pickle,
    read_regions_from_pickle,
    write_itl_to_pickle,
    write_ol_to_pickle,
    write_regions_to_pickle,
)
from dawis.io.store import (
    load_atom_from_hdf5_group,
    load_interscale_tree_from_hdf5_group,
    load_region_from_hdf5_group,
    object_recursive_save_to_hdf5,
    pkl_to_hdf5,
    read_image_atoms,
    read_itl_from_hdf5,
    read_ol_from_hdf5,
    write_itl_to_hdf5,
    write_ol_to_hdf5,
)

__all__ = [
    'load_atom_from_hdf5_group',
    'load_interscale_tree_from_hdf5_group',
    'load_region_from_hdf5_group',
    'object_recursive_save_to_hdf5',
    'pkl_to_hdf5',
    'read_image',
    'read_image_atoms',
    'read_itl_from_hdf5',
    'read_itl_from_pickle',
    'read_ol_from_hdf5',
    'read_ol_from_pickle',
    'read_regions_from_pickle',
    'write_itl_to_hdf5',
    'write_itl_to_pickle',
    'write_ol_to_hdf5',
    'write_ol_to_pickle',
    'write_regions_to_pickle',
]
