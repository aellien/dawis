"""The two builds give the same frame, on a tiny 1.0 run (#119, #123).

Run with ``pytest benchmarks/test_atom_store.py``; it is outside the default
suite because the code it checks is a measuring instrument, not the package.

What it pins is the harness's one promise: the frame built from the 1.0 store
and the frame built from the pre-1.0 twin ``unpack`` writes are equal *exactly*,
not to a tolerance. Float addition is not associative, so that only holds while
both builds add atoms in the run's write order — which is what the twelve-atom
iteration below is for, since it makes group ``o10`` sort before ``o2``.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'tests'))
sys.path.insert(0, str(Path(__file__).parents[1] / 'tests' / 'unit'))

import atom_store  # noqa: E402
from test_dataclasses import a_tree  # noqa: E402

from dawis.io.legacy import write_ol_to_hdf5  # noqa: E402
from dawis.io.store import AtomStore, read_frame_shape  # noqa: E402
from dawis.restoration import Atom  # noqa: E402

#: Deliberately not square: a harness that transposed the frame it read off the
#: store would agree with itself on a square one.
SHAPE = (24, 30)


def _atom(rng, bbox, level):
    shape = (bbox[2] - bbox[0], bbox[3] - bbox[1])
    return Atom(rng.normal(size=shape), rng.normal(size=shape), bbox, level,
                0.1, 'BSPL', True, 0.0, 0.0)


def _tiny_run(tmp_path):
    """A two-iteration 1.0 store, and the atoms it holds in write order."""
    rng = np.random.default_rng(0)
    outpath = str(tmp_path / 'run')
    # 12 atoms in iteration 1, so the twin's group 'o10' sorts before 'o2'.
    it1 = [_atom(rng, (i, i + 1, i + 5, i + 4), i % 7) for i in range(12)]
    it2 = [_atom(rng, (3, 2, 20, 9), 6), _atom(rng, (0, 0, 1, 1), 2),
           _atom(rng, (2, 3, 7, 4), 6)]
    with AtomStore(outpath + '.atoms.hdf5', SHAPE) as store:
        store.append(1, it1, [a_tree() for _ in it1])
        store.append(2, it2, [a_tree() for _ in it2])
    return outpath, it1 + it2


def test_builds_agree_exactly_on_both_selections(tmp_path):
    outpath, atoms = _tiny_run(tmp_path)
    nfp = str(tmp_path / 'twin')
    atom_store.unpack(outpath, nfp)
    # The shape the harness will use, not one handed in: a wrong one has to be
    # able to fail here.
    shape = read_frame_shape(outpath)
    assert shape == SHAPE
    for min_level in (0, 5):
        expected = np.zeros(shape)
        for o in atoms:
            if o.level >= min_level:
                x0, y0, x1, y1 = o.bbox
                expected[x0:x1, y0:y1] += o.image
        assert expected.any(), 'a selection that picks nothing proves nothing'
        pre = atom_store.build_pre(nfp, shape, min_level)
        long = atom_store.build_long(outpath, shape, min_level)
        # Exactly, not to a tolerance: same atoms added in the same order.
        np.testing.assert_array_equal(pre, expected)
        np.testing.assert_array_equal(long, expected)


def test_unpack_writes_one_file_per_iteration(tmp_path):
    outpath, atoms = _tiny_run(tmp_path)
    nfp = str(tmp_path / 'twin')
    atom_store.unpack(outpath, nfp)
    files = atom_store._iteration_files(nfp)
    assert [Path(p).name for p in files] == ['twin.ol.it001.hdf5',
                                             'twin.ol.it002.hdf5']
    assert sum(len(atom_store._atoms_in_write_order(p)) for p in files) == len(atoms)


def test_unpack_refuses_to_add_to_an_existing_twin(tmp_path):
    outpath, _ = _tiny_run(tmp_path)
    nfp = str(tmp_path / 'twin')
    atom_store.unpack(outpath, nfp)
    with pytest.raises(FileExistsError):
        atom_store.unpack(outpath, nfp)


def test_check_reports_a_difference_of_exactly_zero(tmp_path):
    outpath, _ = _tiny_run(tmp_path)
    nfp = str(tmp_path / 'twin')
    atom_store.unpack(outpath, nfp)
    # check() finds the shape itself, so this also pins that path.
    assert atom_store.check(outpath, nfp, 0) == 0.0
    assert atom_store.check(outpath, nfp, 5) == 0.0


def test_check_reports_a_real_difference(tmp_path):
    """The zero above has to be capable of being non-zero."""
    outpath, _ = _tiny_run(tmp_path)
    nfp = str(tmp_path / 'twin')
    atom_store.unpack(outpath, nfp)
    # Drop one atom from the twin, which the store still holds.
    victim = atom_store._iteration_files(nfp)[1]
    kept = atom_store._atoms_in_write_order(victim)[1:]
    Path(victim).unlink()
    write_ol_to_hdf5(kept, victim)
    assert atom_store.check(outpath, nfp, 0) > 0.0


def test_time_logs_one_line_per_build(tmp_path):
    outpath, _ = _tiny_run(tmp_path)
    nfp = str(tmp_path / 'twin')
    log = str(tmp_path / 'log.jsonl')
    atom_store.unpack(outpath, nfp)
    atom_store.time_builds(outpath, nfp, log, reps=2, min_levels=(0, 5))
    lines = [json.loads(line) for line in open(log)]
    assert len(lines) == 2 * 2 * 2 * 2
    assert [x['store'] for x in lines[:2]] == ['pre', 'pre']
    assert [x['store'] for x in lines[8:10]] == ['long', 'long']
    for min_level in (0, 5):
        sums = {x['frame_sum'] for x in lines if x['min_level'] == min_level}
        assert len(sums) == 1
