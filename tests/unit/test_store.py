"""Tier 0 — the atom store a run writes, and a run resumes from (#122).

One plain HDF5 file per run, ``<outpath>.atoms.hdf5``: every atom's pixels in
two long arrays, one index row per atom holding its scalars and its tree's.
The design is #120's; these tests pin its four promises — a round trip that
loses nothing an atom holds, an iteration that counts only once its index rows
are down, a resume that cuts off what a killed run left past them, and a
pipeline that writes this file and no per-iteration dump.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

from test_dataclasses import a_tree  # noqa: E402
from tinyrun import TINY_NAME, TINY_PARAMS, run_quiet, tiny_image  # noqa: E402

from dawis.config import Config  # noqa: E402
from dawis.io.store import AtomStore, read_frame_shape, read_image_atoms  # noqa: E402
from dawis.restoration import Atom  # noqa: E402

SHAPE = (32, 32)


def an_atom(value: float, bbox=(2, 3, 6, 8), level=2) -> Atom:
    """An atom whose pixels are all different, so a mixed-up slice shows."""
    x0, y0, x1, y1 = bbox
    n = (x1 - x0) * (y1 - y0)
    return Atom(value + np.arange(n, dtype=float).reshape(x1 - x0, y1 - y0),
                -value - np.arange(n, dtype=float).reshape(x1 - x0, y1 - y0),
                bbox, level, 0.25, 'HAAR', False, 1.5, 2.5,
                'projected_gradient', 'TRANSPOSE')


def two_iterations(path):
    with AtomStore(path, SHAPE) as store:
        store.append(1, [an_atom(0.), an_atom(100., (0, 0, 1, 5))],
                     [a_tree(), a_tree()])
        store.append(2, [an_atom(200., level=4)], [a_tree()])


def test_the_store_round_trips_every_atom(tmp_path):
    outpath = str(tmp_path / 'run')
    two_iterations(outpath + '.atoms.hdf5')

    atoms, index = read_image_atoms(outpath)
    expected = [an_atom(0.), an_atom(100., (0, 0, 1, 5)), an_atom(200., level=4)]

    assert list(index['iteration']) == [1, 1, 2]
    assert list(index['atom_id']) == [0, 1, 0]
    for back, atom in zip(atoms, expected, strict=True):
        np.testing.assert_array_equal(back.image, atom.image)
        np.testing.assert_array_equal(back.det_err_image, atom.det_err_image)
        assert back.image.shape == atom.image.shape
        for name in ('bbox', 'level', 'eccentricity', 'filter',
                     'flag_convergence', 'sum_wr', 'norm_wr', 'algorithm',
                     'synthesis_operator'):
            assert getattr(back, name) == getattr(atom, name), name
            assert type(getattr(back, name)) is type(getattr(atom, name)), name
    assert read_frame_shape(outpath) == SHAPE


def test_the_index_carries_the_tree_and_its_maximum(tmp_path):
    outpath = str(tmp_path / 'run')
    two_iterations(outpath + '.atoms.hdf5')
    tree = a_tree()

    _, index = read_image_atoms(outpath)
    row = index[0]

    assert tuple(row['tree_bbox']) == tuple(tree.bbox)
    for name in ('x_size', 'y_size', 'n_regions', 'span_levels', 'extent',
                 'eccentricity', 'lowb', 'highb', 'clip'):
        assert row['tree_' + name] == getattr(tree, name), name
    for name in ('level', 'x_max', 'y_max', 'norm_max_intensity'):
        assert row['maximum_' + name] == getattr(tree.interscale_maximum, name), name


def test_one_iteration_reads_alone(tmp_path):
    outpath = str(tmp_path / 'run')
    two_iterations(outpath + '.atoms.hdf5')

    atoms, index = read_image_atoms(outpath, iteration=2)

    assert list(index['iteration']) == [2]
    np.testing.assert_array_equal(atoms[0].image, an_atom(200.).image)


def test_a_tree_for_every_atom_or_nothing(tmp_path):
    with AtomStore(tmp_path / 'a.atoms.hdf5', SHAPE) as store:
        with pytest.raises(ValueError, match='trees'):
            store.append(1, [an_atom(0.), an_atom(1.)], [a_tree()])
        assert store.iterations() == set()


def test_resume_cuts_off_what_a_killed_run_left(tmp_path):
    """Pixels written past the last index row are an iteration never committed."""
    path = tmp_path / 'run.atoms.hdf5'
    two_iterations(path)
    with h5py.File(path, 'a') as f:
        for name in ('image', 'det_err_image'):
            f[name].resize((f[name].shape[0] + 7,))

    with AtomStore(path, SHAPE, resume=True) as store:
        assert store.iterations() == {1, 2}
        [atom] = store.read(2)
        np.testing.assert_array_equal(atom.image, an_atom(200.).image)
        store.append(3, [an_atom(300.)], [a_tree()])

    atoms, index = read_image_atoms(str(tmp_path / 'run'), iteration=3)
    np.testing.assert_array_equal(atoms[0].image, an_atom(300.).image)
    with h5py.File(path, 'r') as f:
        end = index['offset'][-1] + index['length'][-1]
        assert f['image'].shape == f['det_err_image'].shape == (end,)


def test_resume_refuses_a_different_frame(tmp_path):
    path = tmp_path / 'run.atoms.hdf5'
    two_iterations(path)

    with pytest.raises(ValueError, match='frame'):
        AtomStore(path, (64, 64), resume=True)


def test_resume_names_the_store_it_cannot_read(tmp_path):
    """A store killed before its datasets existed is not a store (#124).

    Creating the file, writing the ``shape`` attribute and creating the three
    datasets are four writes, not one, so a kill in between leaves a file that
    exists and holds none of them. Reading ``attrs['shape']`` off it raised a
    bare ``KeyError`` naming the attribute, and left the file open — so the
    run could not even start over on it.
    """
    path = tmp_path / 'run.atoms.hdf5'
    h5py.File(path, 'w').close()

    with pytest.raises(ValueError, match='run.atoms.hdf5'):
        AtomStore(path, SHAPE, resume=True)

    with AtomStore(path, SHAPE) as store:
        assert store.iterations() == set()


def test_resume_names_a_store_hdf5_cannot_open_at_all(tmp_path):
    """The kill can land inside HDF5's own file creation, not only after it.

    ``h5py.File(path, 'a')`` then raises before any guard here can look inside,
    with an ``OSError`` naming a byte address and not the store.
    """
    path = tmp_path / 'run.atoms.hdf5'
    path.write_bytes(b'\x89HDF\r\n\x1a\n' + bytes(512))

    with pytest.raises(ValueError, match='run.atoms.hdf5'):
        AtomStore(path, SHAPE, resume=True)


def test_the_goldens_read_one_iteration_at_a_time(tmp_path, monkeypatch):
    """``GoldenRun.atoms`` streams rather than reading a whole run (#124).

    ``read_image_atoms`` with no ``iteration`` holds every pixel of the run in
    memory at once — its own docstring warns so — and
    ``atom_catalogue_with_digests`` walked the atoms twice, so a Tier 2
    comparison paid for that twice. The pickle path it replaced read one
    iteration at a time. The spy pins both halves: one call per iteration,
    ``iteration`` always given, and one walk rather than two.
    """
    import dawis
    import goldenrun

    outpath = str(tmp_path / 'run')
    two_iterations(outpath + '.atoms.hdf5')

    calls = []
    read = dawis.read_image_atoms

    def spy(path, iteration=None):
        calls.append(iteration)
        return read(path, iteration=iteration)

    monkeypatch.setattr(dawis, 'read_image_atoms', spy)

    rows = goldenrun.atom_catalogue_with_digests(
        dawis, goldenrun.GoldenRun(Path(outpath)))

    assert calls == [1, 2]
    assert [(row['iteration'], row['index']) for row in rows] == [(1, 0), (1, 1), (2, 0)]
    assert all(row['image_sha256'] for row in rows)


def test_without_resume_the_file_is_replaced(tmp_path):
    path = tmp_path / 'run.atoms.hdf5'
    two_iterations(path)

    with AtomStore(path, SHAPE) as store:
        assert store.iterations() == set()


def test_a_run_writes_the_store_and_no_dumps(tmp_path):
    result = run_quiet(tiny_image(), Config(**TINY_PARAMS), outdir=str(tmp_path),
                       name=TINY_NAME)

    assert not list(tmp_path.glob('*.ol.it*')) and not list(tmp_path.glob('*.itl.it*'))
    atoms, index = read_image_atoms(result.outpath)
    assert len(atoms) == result.n_atoms > 0
    assert set(index['iteration']) == set(range(1, result.n_iterations + 1))
    restored = np.zeros(tiny_image().shape)
    for atom in atoms:
        x0, y0, x1, y1 = atom.bbox
        restored[x0:x1, y0:y1] += atom.image
    np.testing.assert_allclose(restored, result.restored, rtol=0, atol=1e-9)


def test_a_resumed_run_reads_its_iterations_back(tmp_path):
    first = run_quiet(tiny_image(), Config(**TINY_PARAMS), outdir=str(tmp_path),
                      name=TINY_NAME)
    _, before = read_image_atoms(first.outpath)

    again = run_quiet(tiny_image(), Config(**dict(TINY_PARAMS, resume=True)),
                      outdir=str(tmp_path), name=TINY_NAME)

    _, after = read_image_atoms(again.outpath)
    np.testing.assert_array_equal(after, before)
    np.testing.assert_array_equal(again.restored, first.restored)
