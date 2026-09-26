"""A build-time harness for the atom store (#119, #123).

Measures one thing, for #118 and then #121: does a selection build faster from
the 1.0 atom store — one long pixel array per run with an index — than from the
pre-1.0 per-iteration HDF5 files it replaced?

The 1.0 store is what the pipeline writes, and :mod:`dawis.io.store` owns it.
The pre-1.0 side does not exist on a 1.0 run, so the harness makes it: ``unpack``
writes the twin, one ``<nfp>.ol.itNNN.hdf5`` per iteration, through the same
:func:`dawis.io.legacy.write_ol_to_hdf5` a pre-1.0 run wrote with.

A *build* adds the selected atoms' ``image`` into one full frame. Both builds
visit atoms in the order the run wrote them, so on the same selection they give
the same frame bit for bit — see :func:`_atoms_in_write_order` for the one place
that costs anything.

Steps on a 1.0 run, with ``OUTPATH`` the run's path stem (its store is
``OUTPATH.atoms.hdf5``) and ``NFP`` the stem the twin is written under::

    python benchmarks/atom_store.py unpack OUTPATH NFP
    python benchmarks/atom_store.py check OUTPATH NFP --min-level 0
    python benchmarks/atom_store.py time OUTPATH NFP --log LOG --reps 3

``check`` exits non-zero unless the two frames are equal exactly, so the
promise the harness rests on is enforced rather than merely printed.

``time`` runs every build in a fresh process and appends one JSON line per
build to ``LOG``. A cold build drops the page cache of its own files with
``posix_fadvise(DONTNEED)``, which needs no root; the warm build is the same
build run again straight after it. The drop happens **in the build process,
after it has read the frame's shape and just before the clock starts** — done
in the parent it would leave the long store's metadata pages resident, which
the ``pre`` side gets no equivalent of, and the bias would favour the store.
Store order alternates between repetitions. Peak memory is the build process's
``ru_maxrss``, interpreter and imports included. The frame's shape comes off
the store, so the harness is not pinned to one field.

One bias is left in, knowingly: :func:`_atoms_in_write_order` opens each
iteration file a second time for its group names, inside the timed region, and
a pre-1.0 run never paid that. Measured at 0.63 % of the ``pre`` side's read on
a 19-file twin — an order of magnitude under run-to-run variance and three
under #118's 21–29× margin. Removing it would mean reimplementing
:func:`~dawis.io.legacy.read_ol_from_hdf5` here, and #105 is the record of what
this repo's harnesses cost when they reimplement what the package already owns.

#119's throwaway writer, which made a long-array store out of a pre-1.0 ``.pkl``
run, is gone: the pipeline writes the real store now, and the two layouts differ
(the 1.0 index carries one ``bbox`` column where the throwaway had four). Its
measurement, #118, reproduces from commit ``3b4f4d6``.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import resource
import subprocess
import sys
import time

import h5py
import numpy as np

# dawis is not installed; run it from this checkout.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dawis.io.legacy import read_ol_from_hdf5, write_ol_to_hdf5  # noqa: E402
from dawis.io.store import (  # noqa: E402
    STORE_SUFFIX,
    read_atom_index,
    read_frame_shape,
    read_image_atoms,
)


def _iteration_files(nfp: str) -> list[str]:
    """The twin's per-iteration atom files, sorted as read_legacy_run sorts them."""
    return sorted(glob.glob(nfp + '*ol.it*.hdf5'))


def unpack(outpath: str, nfp: str) -> None:
    """Write the pre-1.0 twin of a 1.0 run: one atom file per iteration.

    The iteration number is zero-padded to a common width, so that sorting the
    file names as strings — which is how :func:`_iteration_files` reaches them —
    puts them in iteration order past ``it999`` as well.

    The twin is **atoms only**: no ``.itl.itNNN.hdf5`` tree file is written,
    because the thing being timed adds atom pixels into a frame and never looks
    at a tree. So the twin is not a run
    :func:`~dawis.io.legacy.read_legacy_run` can read — it globs atoms and trees
    and zips them, and against an empty tree list that returns nothing at all,
    silently. It is input for :func:`build_pre` and for nothing else.

    Raises
    ------
    FileExistsError
        If a twin is already there. Left over from a different run it would be
        read as part of this one, and the pre-1.0 build would be timed over
        atoms the store does not hold.
    """

    if _iteration_files(nfp):
        raise FileExistsError('a twin is already at %s*ol.it*.hdf5; delete it first'
                              %(nfp,))

    iterations = sorted(set(int(it) for it in read_atom_index(outpath)['iteration']))
    width = max(3, len(str(iterations[-1])) if iterations else 3)
    for iteration in iterations:
        atoms, _ = read_image_atoms(outpath, iteration = iteration)
        write_ol_to_hdf5(atoms, '%s.ol.it%0*d.hdf5' %(nfp, width, iteration))


def _atoms_in_write_order(path: str) -> list:
    """One iteration's atoms, in the order the run wrote them.

    :func:`~dawis.io.legacy.read_ol_from_hdf5` returns them in h5py's group
    order, which is alphabetical — ``o10`` before ``o2`` — while the store's
    index is in write order. Both builds have to add atoms in the same order or
    their frames differ in the last bits, and ``check`` is there to report a
    difference of exactly zero.

    The read itself stays the pre-1.0 call, so what ``time`` measures on this
    side is what a pre-1.0 run paid; the second open reads group names only.
    """

    atoms = read_ol_from_hdf5(path)
    with h5py.File(path, 'r') as f:
        names = list(f)
    return [atoms[i] for i in np.argsort([int(name[1:]) for name in names])]


def build_pre(nfp: str, shape: tuple[int, int], min_level: int) -> np.ndarray:
    """Build from the per-iteration files, streaming one file at a time."""
    frame = np.zeros(shape)
    for path in _iteration_files(nfp):
        for o in _atoms_in_write_order(path):
            if o.level >= min_level:
                x0, y0, x1, y1 = o.bbox
                # The legacy format squeezes a one-pixel-wide image to 1-D.
                frame[x0:x1, y0:y1] += o.image.reshape(x1 - x0, y1 - y0)
    return frame


def build_long(outpath: str, shape: tuple[int, int], min_level: int) -> np.ndarray:
    """Build from the 1.0 atom store, reading only the selected slices."""
    frame = np.zeros(shape)
    with h5py.File(outpath + STORE_SUFFIX, 'r') as f:
        index = f['index'][()]
        image = f['image']
        # ponytail: one h5py read per atom; merge contiguous slices if per-read
        # overhead dominates the full-field build.
        for r in index[index['level'] >= min_level]:
            x0, y0, x1, y1 = r['bbox']
            pixels = image[r['offset']:r['offset'] + r['length']]
            frame[x0:x1, y0:y1] += pixels.reshape(x1 - x0, y1 - y0)
    return frame


def _store_files(kind: str, outpath: str, nfp: str) -> list[str]:
    return _iteration_files(nfp) if kind == 'pre' else [outpath + STORE_SUFFIX]


def _drop_cache(paths: list[str]) -> None:
    for p in paths:
        fd = os.open(p, os.O_RDONLY)
        try:
            # DONTNEED drops clean pages only; a store just written is dirty.
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def _build_once(kind: str, outpath: str, nfp: str, min_level: int,
                cold: bool) -> None:
    """Child side of ``time``: one build, one JSON line on stdout.

    The shape is read first and the cache dropped second, so that reading it
    does not leave the store's metadata pages resident in a build billed as
    cold.
    """
    shape = read_frame_shape(outpath)
    if cold:
        _drop_cache(_store_files(kind, outpath, nfp))
    t0 = time.perf_counter()
    frame = (build_pre(nfp, shape, min_level) if kind == 'pre'
             else build_long(outpath, shape, min_level))
    wall = time.perf_counter() - t0
    print(json.dumps({
        'wall_s': wall,
        'peak_rss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        'frame_sum': float(frame.sum()),
    }))


def time_builds(outpath: str, nfp: str, log: str, reps: int,
                min_levels: tuple[int, ...]) -> None:
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                            text=True, cwd=os.path.dirname(__file__)).stdout.strip()
    for rep in range(reps):
        order = ['pre', 'long'] if rep % 2 == 0 else ['long', 'pre']
        for min_level in min_levels:
            for kind in order:
                for cache in ('cold', 'warm'):
                    out = subprocess.run(
                        [sys.executable, __file__, 'build', kind, outpath, nfp,
                         '--min-level', str(min_level)]
                        + (['--cold'] if cache == 'cold' else []),
                        capture_output=True, text=True, check=True).stdout
                    line = {'utc': datetime.datetime.now(datetime.UTC).isoformat(),
                            'commit': commit, 'rep': rep, 'store': kind,
                            'min_level': min_level, 'cache': cache,
                            **json.loads(out.splitlines()[-1])}
                    with open(log, 'a') as f:
                        f.write(json.dumps(line) + '\n')
                    print(json.dumps(line), flush=True)


def check(outpath: str, nfp: str, min_level: int) -> float:
    """Print how far the two builds differ on one selection, and return it.

    ``max_abs_frame`` is printed with it because a difference of zero over an
    empty selection says nothing: a level cut past the run's highest level
    compares two zero frames and passes. Read the two numbers together.

    The caller — :func:`main` — turns a non-zero difference into a non-zero
    exit status. Nothing downstream of this harness is worth running once the
    two builds disagree.
    """

    shape = read_frame_shape(outpath)
    pre = build_pre(nfp, shape, min_level)
    long = build_long(outpath, shape, min_level)
    diff = float(np.abs(pre - long).max())
    print(json.dumps({'min_level': min_level, 'max_abs_diff': diff,
                      'max_abs_frame': float(np.abs(pre).max())}))
    return diff


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    for name in ('unpack', 'check', 'build', 'time'):
        s = sub.add_parser(name)
        if name == 'build':
            s.add_argument('kind', choices=['pre', 'long'])
        s.add_argument('outpath')
        s.add_argument('nfp')
        if name in ('check', 'build'):
            s.add_argument('--min-level', type=int, default=0)
        if name == 'build':
            s.add_argument('--cold', action='store_true')
        if name == 'time':
            s.add_argument('--log', required=True)
            s.add_argument('--reps', type=int, default=3)
            s.add_argument('--min-levels', type=int, nargs='+', default=[0, 5])
    a = p.parse_args()
    if a.cmd == 'unpack':
        unpack(a.outpath, a.nfp)
    elif a.cmd == 'build':
        _build_once(a.kind, a.outpath, a.nfp, a.min_level, a.cold)
    elif a.cmd == 'check':
        if check(a.outpath, a.nfp, a.min_level):
            sys.exit(1)
    else:
        time_builds(a.outpath, a.nfp, a.log, a.reps, tuple(a.min_levels))


if __name__ == '__main__':
    main()
