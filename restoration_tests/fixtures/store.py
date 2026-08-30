"""What a fixture is, and how a set of them is written and read.

A fixture holds enough to run a reconstruction and score it with nothing to
look up elsewhere. That is #75's wording and it is the constraint the format
is built to: an arm in #78 or a metric in #77 opens a fixture and has the
tubes, the geometry, the injected source's parameters and its truth image,
without reaching back into the gallery or re-running detection.

The format
----------
One directory per gallery entry, holding ``fixtures.h5`` and
``manifest.json``::

    <root>/<tier>/<class>_<regime>_r<N>/
        fixtures.h5
        manifest.json
        quicklook/tree000.png ...

HDF5 rather than one ``.npz`` per fixture, for two reasons. The tubes are the
bulk of the bytes and most of a support tube is zero, so per-dataset gzip
earns its place; and an arm sweeping one entry opens one file rather than
thirty. h5py is already a hard DAWIS dependency, so the format costs nothing
new. One file per **entry** rather than one for the whole gallery, because
the sweep is resumable per entry and a half-written gallery-wide file is not.

Fixtures are numbered in the group name and read back in numeric order.
``h5py`` iterates a group alphabetically, where ``tree10`` sorts before
``tree2``; a quicklook filename and a row of a results table both index a set
by position, so the order has to be the one they were written in.

The manifest
------------
Everything a reader cannot get out of the HDF5 file: which gallery entry the
fixtures came from and that entry's own manifest, the frozen settings they
were dumped under, the DAWIS commit that produced them, the array dtypes —
read off the arrays rather than written down, so the table cannot claim a
dtype the file does not hold — and the match counts — how many trees matched a source, how many matched none and
how many matched two. That last one is the number that says whether a dump is
about the sources or about detection.

An entry that matched nothing still gets a manifest. A mock iteration that
found no scorable tree is a result, and one a resumable sweep has to be able
to record.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np

from restoration_tests.fixtures.matching import GallerySource
from restoration_tests.fixtures.mock import MockSettings

__all__ = ['ARRAY_NAMES', 'Fixture', 'TreeGeometry', 'dawis_commit',
           'read_fixtures', 'read_manifest', 'write_fixtures']

#: The arrays a fixture carries, in the order they are written. Named once so
#: the round trip and the manifest's dtype table cannot drift apart.
ARRAY_NAMES = ('wavelet_tube', 'support_tube', 'label_tube', 'truth_cutout',
               'reference_reconstruction')

FIXTURES_FILE = 'fixtures.h5'
MANIFEST_FILE = 'manifest.json'


@dataclass(frozen = True)
class TreeGeometry:
    """Where a tree sits and what shape it is.

    The descriptors the restoration branches on, carried so a fixture can be
    routed, binned or plotted without rebuilding the tree.

    Attributes
    ----------
    peak_row, peak_col : int
        The interscale maximum's pixel in the **full image** — the root
        region's ``x_max`` and ``y_max``, which are a numpy row and column.
    level : int
        The scale the interscale maximum sits on, and so the coarsest scale
        the tree spans. The tubes have ``level + 1`` planes.
    bbox : tuple of int
        ``(row_min, col_min, row_max, col_max)`` of the tree in the full
        image — the widened, clipped box the tubes are cut to, so
        ``truth_cutout`` and the tube planes share it.
    extent : float
        Fraction of the tree's unwidened box covered by its regions.
    eccentricity : float
        Eccentricity of the interscale maximum.
    """

    peak_row: int
    peak_col: int
    level: int
    bbox: tuple[int, int, int, int]
    extent: float
    eccentricity: float


@dataclass(frozen = True)
class Fixture:
    """One interscale tree, and the injected source it should reconstruct.

    Attributes
    ----------
    fixture_id : str
        ``<tier>/<class>_<regime>_r<N>/tree<NNN>``. Unique across the whole
        battery, and the name a result is recorded against.
    entry : str
        The gallery entry, ``<tier>/<class>_<regime>_r<N>``.
    source : GallerySource
        The matched injected source, straight from the entry's
        ``sources.csv``.
    geometry : TreeGeometry
        Where the tree sits and what shape it is.
    wavelet_tube, support_tube, label_tube : numpy.ndarray
        The tree cut out of the cubes, of shape
        ``(rows, columns, level + 1)``. What an arm reconstructs from.
    truth_cutout : numpy.ndarray
        The entry's PSF-convolved, noiseless truth over ``geometry.bbox`` —
        what a reconstruction is scored against. Every source that falls in
        the box is in it, not only the matched one, because that is what the
        truth of that patch of sky is.
    reference_reconstruction : numpy.ndarray
        What the published default arm made of this tree, over the same box.
        Not a ground truth and not a target: a baseline recorded once, so the
        quicklook has something to draw and #77 has something to read every
        other arm against.
    reference_arm : dict
        What that default arm actually did — the filter and operator
        :func:`~dawis.restoration.synthesis.restore_object` chose for this
        tree, whether it deconvolved, its final residual norm and whether it
        converged.
    """

    fixture_id: str
    entry: str
    source: GallerySource
    geometry: TreeGeometry
    wavelet_tube: np.ndarray
    support_tube: np.ndarray
    label_tube: np.ndarray
    truth_cutout: np.ndarray
    reference_reconstruction: np.ndarray
    reference_arm: dict


def dawis_commit() -> str:
    """The DAWIS commit the fixtures were dumped from.

    Returns
    -------
    str
        The short sha, with ``-dirty`` appended when the working tree is not
        clean, or ``'unknown'`` when there is no git to ask. The gallery
        records its own source commit the same way, and for the same reason:
        a fixture set is only comparable with another one built from the same
        detection code.
    """

    try:
        sha = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                             cwd = Path(__file__).resolve().parent,
                             capture_output = True, text = True,
                             check = True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain'],
                               cwd = Path(__file__).resolve().parent,
                               capture_output = True, text = True,
                               check = True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'

    return '%s-dirty' % sha if dirty else sha


def write_fixtures(
    directory: str | Path,
    fixtures: list[Fixture],
    entry_name: str,
    settings: MockSettings,
    entry_manifest: dict,
    match_counts: dict,
) -> Path:
    """Write one entry's fixtures and their manifest.

    Parameters
    ----------
    directory : str or pathlib.Path
        Where to write. Created if it does not exist. An existing
        ``fixtures.h5`` is replaced.
    fixtures : list of Fixture
        The fixtures, in the order they should be read back. May be empty.
    entry_name : str
        The gallery entry, ``<tier>/<class>_<regime>_r<N>``. Passed rather
        than read off the first fixture, because a set that matched nothing
        has no first fixture — and that is exactly the set a reader most
        needs to be able to identify.
    settings : MockSettings
        The frozen settings the mock iteration ran under.
    entry_manifest : dict
        The gallery entry's own ``manifest.json``, copied whole. It carries
        the generator's source commit, the noise parameters and the ladders,
        and a fixture result is not readable without them.
    match_counts : dict
        How many trees ended in each matching outcome.

    Returns
    -------
    pathlib.Path
        The directory written to.
    """

    directory = Path(directory)
    directory.mkdir(parents = True, exist_ok = True)

    with h5py.File(directory / FIXTURES_FILE, 'w') as store:
        for index, fixture in enumerate(fixtures):
            group = store.create_group('fixture%06d' % index)
            group.attrs['fixture_id'] = fixture.fixture_id
            group.attrs['entry'] = fixture.entry
            group.attrs['source'] = json.dumps(asdict(fixture.source))
            group.attrs['geometry'] = json.dumps(asdict(fixture.geometry))
            group.attrs['reference_arm'] = json.dumps(fixture.reference_arm)

            for name in ARRAY_NAMES:
                group.create_dataset(name, data = getattr(fixture, name),
                                     compression = 'gzip', compression_opts = 4)

    manifest = {
        'entry': entry_name,
        'n_fixtures': len(fixtures),
        'settings': settings.as_dict(),
        'match_counts': match_counts,
        'gallery_entry': entry_manifest,
        'dawis_commit': dawis_commit(),
        'dtypes': {name: str(getattr(fixtures[0], name).dtype)
                   for name in ARRAY_NAMES} if fixtures else {},
    }
    (directory / MANIFEST_FILE).write_text(json.dumps(manifest, indent = 2))

    return directory


def read_manifest(directory: str | Path) -> dict:
    """Read one entry's fixture manifest.

    Parameters
    ----------
    directory : str or pathlib.Path
        A directory :func:`write_fixtures` wrote.

    Returns
    -------
    dict
    """

    return json.loads((Path(directory) / MANIFEST_FILE).read_text())


def read_fixtures(directory: str | Path) -> list[Fixture]:
    """Read one entry's fixtures back, in the order they were written.

    Parameters
    ----------
    directory : str or pathlib.Path
        A directory :func:`write_fixtures` wrote.

    Returns
    -------
    list of Fixture
        Empty when the entry's mock iteration matched no tree.
    """

    path = Path(directory) / FIXTURES_FILE
    fixtures = []

    with h5py.File(path, 'r') as store:
        for name in sorted(store):
            group = store[name]
            arrays = {array: np.asarray(group[array]) for array in ARRAY_NAMES}
            geometry = json.loads(group.attrs['geometry'])
            geometry['bbox'] = tuple(geometry['bbox'])

            fixtures.append(Fixture(
                fixture_id = str(group.attrs['fixture_id']),
                entry = str(group.attrs['entry']),
                source = GallerySource(**json.loads(group.attrs['source'])),
                geometry = TreeGeometry(**geometry),
                reference_arm = json.loads(group.attrs['reference_arm']),
                **arrays,
            ))

    return fixtures
