"""A throwaway long-array atom store, and a harness that times it (#119).

Measures one thing for #118: does a selection build faster from one long
pixel array per run than from the pre-1.0 per-iteration HDF5 files? It is not
the pipeline writer that ships in 1.0.

The long-array store is one HDF5 file holding

- ``image`` and ``det_err_image``: every atom's pixels, raveled in C order and
  concatenated into one 1-D float64 array each;
- ``index``: one row per atom — iteration, atom id (the ``N`` of the
  per-iteration group ``oN``), level, bbox, and the offset and length of the
  atom's pixels in both arrays.

A *build* adds the selected atoms' ``image`` into one full frame. Both builds
visit atoms in the same order (files sorted as strings, groups as h5py yields
them), so on the same selection they give the same frame bit for bit.

Steps on a pre-1.0 ``.pkl`` run, with ``NFP`` the path prefix before
``.ol.itNNN``::

    python benchmarks/atom_store.py convert NFP
    python benchmarks/atom_store.py write NFP STORE
    python benchmarks/atom_store.py check NFP STORE --min-level 0
    python benchmarks/atom_store.py time NFP STORE --log LOG --reps 3

``time`` runs every build in a fresh process and appends one JSON line per
build to ``LOG``. A cold build first drops the page cache of the store's own
files with ``posix_fadvise(DONTNEED)``, which needs no root; the warm build is
the same build run again straight after it. Store order alternates between
repetitions. Peak memory is the build process's ``ru_maxrss``, interpreter
and imports included.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import re
import resource
import subprocess
import sys
import time

import h5py
import numpy as np

# dawis is not installed; run it from this checkout.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dawis.io.legacy import _load_hdf5_attribute, pkl_to_hdf5, read_ol_from_hdf5  # noqa: E402

INDEX_DTYPE = np.dtype([
    ('iteration', 'i4'), ('atom_id', 'i4'), ('level', 'i4'),
    ('x_min', 'i4'), ('y_min', 'i4'), ('x_max', 'i4'), ('y_max', 'i4'),
    ('offset', 'i8'), ('length', 'i8'),
])

# Frame of hunt-euclid-2698cut5, numpy order (FITS NAXIS1 = 2650).
SHAPE = (2637, 2650)


def _iteration_files(nfp: str) -> list[str]:
    """The run's per-iteration atom files, sorted as read_image_atoms sorts them."""
    return sorted(glob.glob(nfp + '*ol.it*.hdf5'))


def convert(nfp: str) -> None:
    """Write the HDF5 twin of every ``.pkl`` iteration that lacks one."""
    for pkl_ol in sorted(glob.glob(nfp + '*ol.it*.pkl')):
        if not os.path.exists(pkl_ol[:-3] + 'hdf5'):
            pkl_to_hdf5(pkl_ol, pkl_ol.replace('.ol.it', '.itl.it'))


def write_long_store(nfp: str, out: str, chunk: int = 1 << 20) -> None:
    """Write the run's atoms into one long-array store at ``out``."""
    rows = []
    offset = 0
    with h5py.File(out, 'w') as f:
        arrays = [f.create_dataset(name, (0,), maxshape=(None,), dtype='f8',
                                   chunks=(chunk,))
                  for name in ('image', 'det_err_image')]
        for path in _iteration_files(nfp):
            iteration = int(re.search(r'ol\.it(\d+)\.hdf5$', path)[1])
            with h5py.File(path, 'r') as g:
                for name in g:
                    grp = g[name]
                    n = grp['image'].size
                    for a in arrays:
                        a.resize((offset + n,))
                        a[offset:] = grp[a.name[1:]][()].ravel()
                    bbox = _load_hdf5_attribute(grp['bbox'])
                    rows.append((iteration, int(name[1:]),
                                 _load_hdf5_attribute(grp['level']),
                                 *bbox, offset, n))
                    offset += n
        f.create_dataset('index', data=np.array(rows, dtype=INDEX_DTYPE))


def read_index(store: str) -> np.ndarray:
    with h5py.File(store, 'r') as f:
        return f['index'][()]


def build_pre(nfp: str, shape: tuple[int, int], min_level: int) -> np.ndarray:
    """Build from the per-iteration files, streaming one file at a time."""
    frame = np.zeros(shape)
    for path in _iteration_files(nfp):
        for o in read_ol_from_hdf5(path):
            if o.level >= min_level:
                x0, y0, x1, y1 = o.bbox
                # The store squeezes a one-pixel-wide image to 1-D.
                frame[x0:x1, y0:y1] += o.image.reshape(x1 - x0, y1 - y0)
    return frame


def build_long(store: str, shape: tuple[int, int], min_level: int) -> np.ndarray:
    """Build from the long-array store, reading only the selected slices."""
    frame = np.zeros(shape)
    with h5py.File(store, 'r') as f:
        index = f['index'][()]
        image = f['image']
        # ponytail: one h5py read per atom; merge contiguous slices if per-read
        # overhead dominates the full-field build.
        for r in index[index['level'] >= min_level]:
            x0, y0, x1, y1 = r['x_min'], r['y_min'], r['x_max'], r['y_max']
            pixels = image[r['offset']:r['offset'] + r['length']]
            frame[x0:x1, y0:y1] += pixels.reshape(x1 - x0, y1 - y0)
    return frame


def _store_files(kind: str, source: str) -> list[str]:
    return _iteration_files(source) if kind == 'pre' else [source]


def _drop_cache(paths: list[str]) -> None:
    for p in paths:
        fd = os.open(p, os.O_RDONLY)
        try:
            # DONTNEED drops clean pages only; a store just written is dirty.
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def _build_once(kind: str, source: str, min_level: int) -> None:
    """Child side of ``time``: one build, one JSON line on stdout."""
    build = build_pre if kind == 'pre' else build_long
    t0 = time.perf_counter()
    frame = build(source, SHAPE, min_level)
    wall = time.perf_counter() - t0
    print(json.dumps({
        'wall_s': wall,
        'peak_rss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        'frame_sum': float(frame.sum()),
    }))


def time_builds(nfp: str, store: str, log: str, reps: int,
                min_levels: tuple[int, ...]) -> None:
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                            text=True, cwd=os.path.dirname(__file__)).stdout.strip()
    sources = {'pre': nfp, 'long': store}
    for rep in range(reps):
        order = ['pre', 'long'] if rep % 2 == 0 else ['long', 'pre']
        for min_level in min_levels:
            for kind in order:
                for cache in ('cold', 'warm'):
                    if cache == 'cold':
                        _drop_cache(_store_files(kind, sources[kind]))
                    out = subprocess.run(
                        [sys.executable, __file__, 'build', kind, sources[kind],
                         '--min-level', str(min_level)],
                        capture_output=True, text=True, check=True).stdout
                    line = {'utc': datetime.datetime.now(datetime.UTC).isoformat(),
                            'commit': commit, 'rep': rep, 'store': kind,
                            'min_level': min_level, 'cache': cache,
                            **json.loads(out.splitlines()[-1])}
                    with open(log, 'a') as f:
                        f.write(json.dumps(line) + '\n')
                    print(json.dumps(line), flush=True)


def check(nfp: str, store: str, min_level: int) -> None:
    pre = build_pre(nfp, SHAPE, min_level)
    long = build_long(store, SHAPE, min_level)
    print(json.dumps({'min_level': min_level,
                      'max_abs_diff': float(np.abs(pre - long).max()),
                      'max_abs_frame': float(np.abs(pre).max())}))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('convert').add_argument('nfp')
    w = sub.add_parser('write')
    w.add_argument('nfp')
    w.add_argument('store')
    b = sub.add_parser('build')
    b.add_argument('kind', choices=['pre', 'long'])
    b.add_argument('source')
    b.add_argument('--min-level', type=int, default=0)
    c = sub.add_parser('check')
    c.add_argument('nfp')
    c.add_argument('store')
    c.add_argument('--min-level', type=int, default=0)
    t = sub.add_parser('time')
    t.add_argument('nfp')
    t.add_argument('store')
    t.add_argument('--log', required=True)
    t.add_argument('--reps', type=int, default=3)
    t.add_argument('--min-levels', type=int, nargs='+', default=[0, 5])
    a = p.parse_args()
    if a.cmd == 'convert':
        convert(a.nfp)
    elif a.cmd == 'write':
        write_long_store(a.nfp, a.store)
    elif a.cmd == 'build':
        _build_once(a.kind, a.source, a.min_level)
    elif a.cmd == 'check':
        check(a.nfp, a.store, a.min_level)
    else:
        time_builds(a.nfp, a.store, a.log, a.reps, tuple(a.min_levels))


if __name__ == '__main__':
    main()
