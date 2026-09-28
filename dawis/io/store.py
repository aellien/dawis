"""The atom store: every atom a run restored, in one HDF5 file (#122).

A run writes ``<outdir>/<name>.atoms.hdf5``, and resumes from it. The layout
is the one #120 ruled, after #118 timed a selection built from it 21–29× faster
than from the per-iteration files it replaces:

- ``image`` and ``det_err_image``: every atom's pixels, raveled in C order and
  concatenated into one 1-D float64 array each;
- ``index``: one row per atom — the iteration, the atom's place in it, the
  offset and length of its pixels, every scalar of the :class:`Atom`, four
  scalars measured off its pixels as they are written
  (:data:`COMPUTED_COLUMNS`), the scalars of the tree it was restored from
  under a ``tree_`` prefix, and four of that tree's interscale maximum under a
  ``maximum_`` prefix (:data:`INDEX_DTYPE`);
- the attribute ``shape``: the frame the atoms sit in.

A tree's regions and its ``det_err_array`` are not written. The regions are
read only during a run, and the noise vector is already inside every atom's
``det_err_image``.

Only the run's main process writes, once per iteration: pixels, then index
rows, then a flush. An iteration counts as written once its index rows are on
disk, so a run killed half way through leaves pixels past the last row and
nothing else wrong; :class:`AtomStore` cuts them off when it resumes. Plain
HDF5, not SWMR: a killed writer left both readable in #120's kill test, and
SWMR needs ``h5clear``, which pip's h5py does not ship.

Pre-1.0 runs, per-iteration ``.ol.itNNN``/``.itl.itNNN`` files, are read
through :mod:`dawis.io.legacy`.
"""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING

import h5py
import numpy as np

from dawis.restoration import Atom

if TYPE_CHECKING:
    from dawis.trees import InterscaleTree

#: What the store's file is called, after the run's path stem.
STORE_SUFFIX = '.atoms.hdf5'

#: One pixel chunk, in values: 8 MB, as #118 timed it.
_CHUNK = 1 << 20

#: Atom fields that are text, all from closed sets far shorter than the width.
_TEXT = ('filter', 'algorithm', 'synthesis_operator')

#: Scalars copied from the tree and from its interscale maximum.
_TREE_FIELDS = ('x_size', 'y_size', 'n_regions', 'span_levels', 'extent',
                'eccentricity', 'lowb', 'highb', 'clip')
_MAXIMUM_FIELDS = ('level', 'x_max', 'y_max', 'norm_max_intensity')

#: One index row. Missing values are NaN in float columns, -1 in integer ones.
INDEX_DTYPE = np.dtype([
    ('iteration', 'i4'), ('atom_id', 'i4'), ('offset', 'i8'), ('length', 'i8'),
    ('bbox', 'i4', (4,)), ('level', 'i4'), ('eccentricity', 'f8'),
    ('filter', 'S32'), ('flag_convergence', '?'), ('sum_wr', 'f8'),
    ('norm_wr', 'f8'), ('algorithm', 'S32'), ('synthesis_operator', 'S32'),
    ('tree_bbox', 'i4', (4,)), ('tree_x_size', 'i4'), ('tree_y_size', 'i4'),
    ('tree_n_regions', 'i4'), ('tree_span_levels', 'i4'), ('tree_extent', 'f8'),
    ('tree_eccentricity', 'f8'), ('tree_lowb', 'f8'), ('tree_highb', 'f8'),
    ('tree_clip', 'i4'),
    ('maximum_level', 'i4'), ('maximum_x_max', 'i4'), ('maximum_y_max', 'i4'),
    ('maximum_norm_max_intensity', 'f8'),
    ('flux', 'f8'), ('centroid_x', 'f8'), ('centroid_y', 'f8'),
    ('kurtosis', 'f8'),
])

#: The index columns measured off an atom's pixels rather than copied off an
#: object, added by #125 so that a selection on flux, on atom position or on
#: kurtosis reads no pixels. The pipeline already holds the pixels when it
#: writes the row, so they cost four scalars; a store written before #125 does
#: not carry them, which :func:`read_atom_index` warns about.
COMPUTED_COLUMNS = ('flux', 'centroid_x', 'centroid_y', 'kurtosis')


def _computed(atom: Atom) -> tuple[float, float, float, float]:
    """An atom's flux, its flux-weighted centroid, and its excess kurtosis.

    The centroid is in frame coordinates — the bounding box's origin is added
    back — along the axes of :attr:`Atom.bbox`, so ``centroid_x`` runs down
    ``image``'s first axis. It is NaN when the flux is not positive, which is
    where an atom position falls back to the centre of the bounding box
    (``CONTEXT.md``, *Atom position*).

    Kurtosis is the **excess** kurtosis over every pixel of the bounding box —
    ``scipy.stats.kurtosis(..., fisher = True)``'s definition, which is what
    the CFIS artefact cut measured (``make_synthesis_sdssrm_xcs.py:505``), and
    what a unit test pins it against. Written out rather than called: this
    runs once per atom inside the write, and scipy's per-call overhead costs
    about a millisecond on a small atom, which is a minute over a run of
    50 000 of them. It is NaN where it is not defined — an atom whose pixels
    are all one value, a one-pixel atom — rather than a warning per atom.
    """

    image = atom.image
    flux = float(image.sum())
    if flux > 0.:
        x0, y0 = int(atom.bbox[0]), int(atom.bbox[1])
        centroid_x = x0 + float(np.arange(image.shape[0]) @ image.sum(axis = 1)) / flux
        centroid_y = y0 + float(np.arange(image.shape[1]) @ image.sum(axis = 0)) / flux
    else:
        centroid_x = centroid_y = np.nan
    squared = (image - image.mean()) ** 2
    variance = float(squared.mean())
    excess = (float((squared * squared).mean()) / variance ** 2 - 3.
              if variance > 0. else np.nan)
    return flux, centroid_x, centroid_y, excess


def _warn_if_written_before_the_computed_columns(index: np.ndarray) -> np.ndarray:
    """Warn, once per read, if the index predates :data:`COMPUTED_COLUMNS`.

    A warning and not an error: level, size and bbox-centre cuts run on such a
    store exactly as they always did. Only a criterion that reads a missing
    column raises, and numpy raises it by name.
    """

    missing = [name for name in COMPUTED_COLUMNS
               if name not in (index.dtype.names or ())]
    if missing:
        warnings.warn('this atom store was written before the index carried %s, so a '
                      'selection on flux, atom position or kurtosis will fail on it; '
                      're-run to get them' % ', '.join(missing), stacklevel = 3)
    return index


def _missing(column: str):
    return np.nan if INDEX_DTYPE[column].kind == 'f' else -1


def _index_rows(iteration: int, offset: int, atoms: list[Atom],
                trees: list[InterscaleTree]) -> np.ndarray:
    """The index rows of one iteration's atoms, starting at pixel ``offset``."""

    rows = np.zeros(len(atoms), dtype = INDEX_DTYPE)
    for j, (atom, tree) in enumerate(zip(atoms, trees)):
        row = rows[j]
        row['iteration'], row['atom_id'] = iteration, j
        row['offset'], row['length'] = offset, atom.image.size
        offset += atom.image.size
        for name in ('bbox', 'level', 'eccentricity', 'flag_convergence',
                     'sum_wr', 'norm_wr') + _TEXT:
            row[name] = getattr(atom, name)
        values = [('tree_bbox', getattr(tree, 'bbox', None))]
        values += [('tree_' + name, getattr(tree, name, None)) for name in _TREE_FIELDS]
        maximum = getattr(tree, 'interscale_maximum', None)
        values += [('maximum_' + name, getattr(maximum, name, None))
                   for name in _MAXIMUM_FIELDS]
        for column, value in values:
            row[column] = _missing(column) if value is None else value
        for column, value in zip(COMPUTED_COLUMNS, _computed(atom)):
            row[column] = value
    return rows


def _rebuild(f: h5py.File, rows: np.ndarray) -> list[Atom]:
    """The atoms of ``rows``, which must sit in one run of pixels, in order."""

    if not len(rows):
        return []
    start = int(rows['offset'][0])
    stop = int(rows['offset'][-1] + rows['length'][-1])
    image = f['image'][start:stop]
    det_err = f['det_err_image'][start:stop]

    atoms = []
    for row in rows:
        x0, y0, x1, y1 = (int(v) for v in row['bbox'])
        pixels = slice(int(row['offset']) - start,
                       int(row['offset'] + row['length']) - start)
        atoms.append(Atom(
            image = image[pixels].reshape(x1 - x0, y1 - y0),
            det_err_image = det_err[pixels].reshape(x1 - x0, y1 - y0),
            bbox = (x0, y0, x1, y1),
            level = int(row['level']),
            eccentricity = float(row['eccentricity']),
            filter = row['filter'].decode(),
            flag_convergence = bool(row['flag_convergence']),
            sum_wr = float(row['sum_wr']),
            norm_wr = float(row['norm_wr']),
            algorithm = row['algorithm'].decode(),
            synthesis_operator = row['synthesis_operator'].decode(),
        ))
    return atoms


class AtomStore:
    """The writer of a run's atom store, and what a resumed run reads from.

    Parameters
    ----------
    path : str or os.PathLike
        The store's file, ``<outdir>/<name>.atoms.hdf5``.
    shape : tuple of int
        The frame the atoms sit in, recorded as the ``shape`` attribute.
    resume : bool, optional
        Keep what the file holds and cut off any pixels past its last index
        row. Default False replaces the file, as a run without resume always
        has.

    Raises
    ------
    ValueError
        On resume, if the file holds a different frame than ``shape``, its
        index has a different layout than this DAWIS writes, is not a finished
        store at all, or cannot be opened as HDF5. Nothing is left open in any
        of the four, so a run can start over on it.

    Notes
    -----
    Used as a context manager; the file is closed on exit.
    """

    def __init__(self, path: str | os.PathLike, shape: tuple[int, int],
                 resume: bool = False):

        if resume and os.path.exists(path):
            # Creating the file, its `shape` and its three datasets are four
            # writes, so a run killed among them leaves a file holding some of
            # them and no atom. Named rather than left to a bare KeyError off
            # whichever one is missing first — or, when the kill landed inside
            # HDF5's own file creation, to an OSError naming a byte address.
            try:
                self._file = h5py.File(path, 'a')
            except OSError as unreadable:
                raise ValueError('cannot resume: %s is not a readable HDF5 file — the '
                                 'run that made it died while it was being created'
                                 % path) from unreadable
            if ('shape' not in self._file.attrs
                    or any(name not in self._file
                           for name in ('image', 'det_err_image', 'index'))):
                self._file.close()
                raise ValueError('cannot resume: %s is not a finished atom store — '
                                 'the run that made it died before it wrote one' % path)
            stored = tuple(int(v) for v in self._file.attrs['shape'])
            if stored != tuple(shape):
                self._file.close()
                raise ValueError('cannot resume: the store holds a %s frame, the image is %s'
                                 %(stored, tuple(shape)))
            index = self._file['index']
            # HDF5 matches a row's fields by name and quietly drops the rest,
            # so resuming a store whose index predates #125 would write atoms
            # with no flux, centroid or kurtosis and say nothing — a run half
            # measured, and no way to tell which half. Refused, like the frame.
            if index.dtype.names != INDEX_DTYPE.names:
                missing = [name for name in INDEX_DTYPE.names
                           if name not in (index.dtype.names or ())]
                self._file.close()
                raise ValueError('cannot resume: the store\'s index has a different layout '
                                 'than this DAWIS writes (%s), and the columns it does not '
                                 'share would be dropped without a word; start the run over'
                                 % ('it is missing ' + ', '.join(missing) if missing
                                    else 'its columns are in another order'))
            last = index[len(index) - 1] if len(index) else None
            end = 0 if last is None else int(last['offset'] + last['length'])
            for name in ('image', 'det_err_image'):
                self._file[name].resize((end,))
            self._file.flush()
            return

        self._file = h5py.File(path, 'w')
        self._file.attrs['shape'] = shape
        for name in ('image', 'det_err_image'):
            self._file.create_dataset(name, (0,), maxshape = (None,), dtype = 'f8',
                                      chunks = (_CHUNK,))
        self._file.create_dataset('index', (0,), maxshape = (None,),
                                  dtype = INDEX_DTYPE, chunks = (1024,))

    def __enter__(self) -> AtomStore:
        return self

    def __exit__(self, *exc) -> None:
        self._file.close()

    def iterations(self) -> set[int]:
        """The iterations whose atoms are in the store."""
        return set(int(it) for it in self._file['index']['iteration'])

    def read(self, iteration: int) -> list[Atom]:
        """One iteration's atoms, in the order they were written."""
        index = self._file['index'][()]
        return _rebuild(self._file, index[index['iteration'] == iteration])

    def append(self, iteration: int, atoms: list[Atom],
               trees: list[InterscaleTree]) -> None:
        """Write one iteration's atoms and the tree each was restored from.

        Raises
        ------
        ValueError
            If there are not as many trees as atoms. Tree ``j`` is atom
            ``j``'s, so a count that differs means the pairing is lost.
        """

        if len(atoms) != len(trees):
            raise ValueError('iteration %d has %d atoms but %d trees'
                             %(iteration, len(atoms), len(trees)))
        if not atoms:
            return

        for atom in atoms:
            x0, y0, x1, y1 = atom.bbox
            if atom.image.shape != (x1 - x0, y1 - y0):
                raise ValueError('an atom image of shape %s does not fill its bbox %s'
                                 %(atom.image.shape, atom.bbox))

        f = self._file
        offset = f['image'].shape[0]
        n = sum(atom.image.size for atom in atoms)
        for name in ('image', 'det_err_image'):
            f[name].resize((offset + n,))
            f[name][offset:] = np.concatenate([getattr(atom, name).ravel()
                                               for atom in atoms])

        rows = _index_rows(iteration, offset, atoms, trees)
        index = f['index']
        index.resize((len(index) + len(rows),))
        index[-len(rows):] = rows
        f.flush()


def read_frame_shape(outpath: str) -> tuple[int, int]:
    """The frame a run's atoms sit in, as the store recorded it."""

    with h5py.File(outpath + STORE_SUFFIX, 'r') as f:
        return tuple(int(v) for v in f.attrs['shape'])


def read_atom_index(outpath: str) -> np.ndarray:
    """A run's index rows alone, one per atom, without a pixel read.

    What a selection filters on before it reads any atom; see
    :data:`INDEX_DTYPE` for the columns.
    """

    with h5py.File(outpath + STORE_SUFFIX, 'r') as f:
        return _warn_if_written_before_the_computed_columns(f['index'][()])


def read_synthesised_image(outpath: str, rows: np.ndarray,
                           det_err: bool = False) -> np.ndarray:
    """The frame the atoms of ``rows`` add up to, read off the store (#127).

    Each atom is added inside its bounding box, so overlapping atoms
    accumulate, as :class:`~dawis.restoration.Reconstruction` accumulates them
    during the run. Only the pixels of ``rows`` are read.

    Parameters
    ----------
    outpath : str
        Path stem of the run; the store is ``<outpath>.atoms.hdf5``.
    rows : numpy.ndarray
        Index rows of this store, from :func:`read_atom_index`, in any order.
    det_err : bool, optional
        Sum the atoms' ``det_err_image`` rather than their image.

    Returns
    -------
    numpy.ndarray
        On the store's frame; zeros where no atom of ``rows`` lies.
    """

    with h5py.File(outpath + STORE_SUFFIX, 'r') as f:
        frame = np.zeros(tuple(int(v) for v in f.attrs['shape']))
        pixels = f['det_err_image' if det_err else 'image']
        # ponytail: one read per atom, as benchmarks/atom_store.py timed at
        # 1.6 s over 50 220 atoms; read spans of adjacent rows if that is slow.
        for row in rows:
            x0, y0, x1, y1 = (int(v) for v in row['bbox'])
            start = int(row['offset'])
            frame[x0:x1, y0:y1] += pixels[start:start + int(row['length'])].reshape(
                x1 - x0, y1 - y0)
    return frame


def read_image_atoms(
    outpath: str,
    iteration: int | None = None,
) -> tuple[list[Atom], np.ndarray]:
    """Read a run's atoms back off its store, with their index rows.

    Parameters
    ----------
    outpath : str
        Path stem of the run, ``<outdir>/<name>``; the store is
        ``<outpath>.atoms.hdf5``.
    iteration : int or None, optional
        Read one iteration only. Default None reads every atom, which holds
        all of a run's pixels in memory at once.

    Returns
    -------
    atoms : list of Atom
        The atoms, in the order the run wrote them.
    index : numpy.ndarray
        Their index rows, a structured array of :data:`INDEX_DTYPE`: row ``j``
        is atom ``j``'s, and carries the scalars of the tree it came from.
    """

    with h5py.File(outpath + STORE_SUFFIX, 'r') as f:
        index = _warn_if_written_before_the_computed_columns(f['index'][()])
        if iteration is not None:
            index = index[index['iteration'] == iteration]
        return _rebuild(f, index), index
