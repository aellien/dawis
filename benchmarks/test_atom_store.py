"""The long-array build gives the pre-1.0 build's frame, on a tiny fake run.

Run with ``pytest benchmarks/test_atom_store.py``; it is outside the default
suite because the code it checks is a throwaway (#119).
"""

import numpy as np

import atom_store
from dawis.io.store import write_ol_to_hdf5
from dawis.restoration import Atom


def _atom(rng, bbox, level):
    shape = (bbox[2] - bbox[0], bbox[3] - bbox[1])
    return Atom(rng.normal(size=shape), rng.normal(size=shape), bbox, level,
                0.1, 'BSPL', True, 0.0, 0.0)


def _fake_run(tmp_path):
    rng = np.random.default_rng(0)
    nfp = str(tmp_path / 'run')
    # 12 atoms in iteration 1 so group 'o10' sorts before 'o2'.
    it1 = [_atom(rng, (i, i + 1, i + 5, i + 4), i % 7) for i in range(12)]
    it2 = [_atom(rng, (3, 2, 20, 9), 6), _atom(rng, (0, 0, 1, 1), 2),
           _atom(rng, (2, 3, 7, 4), 6)]
    write_ol_to_hdf5(it1, nfp + '.ol.it001.hdf5')
    write_ol_to_hdf5(it2, nfp + '.ol.it002.hdf5')
    return nfp, it1 + it2


def test_builds_agree_on_both_selections(tmp_path):
    nfp, atoms = _fake_run(tmp_path)
    store = str(tmp_path / 'long.hdf5')
    atom_store.write_long_store(nfp, store, chunk=16)
    shape = (24, 24)
    for min_level in (0, 5):
        expected = np.zeros(shape)
        for o in atoms:
            if o.level >= min_level:
                x0, y0, x1, y1 = o.bbox
                expected[x0:x1, y0:y1] += o.image
        pre = atom_store.build_pre(nfp, shape, min_level)
        long = atom_store.build_long(store, shape, min_level)
        np.testing.assert_allclose(pre, expected, rtol=0, atol=1e-12)
        np.testing.assert_allclose(long, pre, rtol=0, atol=1e-12)


def test_index_rows(tmp_path):
    nfp, atoms = _fake_run(tmp_path)
    store = str(tmp_path / 'long.hdf5')
    atom_store.write_long_store(nfp, store, chunk=16)
    index = atom_store.read_index(store)
    assert len(index) == len(atoms)
    assert index['length'].sum() == sum(o.image.size for o in atoms)
    assert np.all(index['offset'][1:] == np.cumsum(index['length'])[:-1])
    assert set(index['iteration']) == {1, 2}


def test_time_logs_one_line_per_build(tmp_path):
    import json
    nfp, _ = _fake_run(tmp_path)
    store = str(tmp_path / 'long.hdf5')
    log = str(tmp_path / 'log.jsonl')
    atom_store.write_long_store(nfp, store, chunk=16)
    atom_store.time_builds(nfp, store, log, reps=2, min_levels=(0, 5))
    lines = [json.loads(line) for line in open(log)]
    assert len(lines) == 2 * 2 * 2 * 2
    assert [x['store'] for x in lines[:2]] == ['pre', 'pre']
    assert [x['store'] for x in lines[8:10]] == ['long', 'long']
    for min_level in (0, 5):
        sums = {x['frame_sum'] for x in lines if x['min_level'] == min_level}
        assert len(sums) == 1
