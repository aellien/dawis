#!/usr/bin/env python
"""Print every environment fact the provenance audit treats as an input (#62).

Hand-run, like the two recording scripts beside it. Not a test module, not a
pytest tier, not run in CI, and imported by nothing.

It exists so that ``provenance-audit.md`` can be re-run at the next release
rather than re-invented: the document says what each fact is and whether it
moves output, and this script is what reads the facts off a machine.

Every fact printed here is readable from inside a run. That is the point of
the list rather than an accident of it: the audit's standard is that an input
counts as recorded only if a run's own outputs state it, so a candidate that
could not be read at run time would be a different and much harder problem.
None of these are.

The one fact deliberately absent is the maths-library thread count. It is
already recorded -- ``[run].n_threads`` beside ``[machine].effective_n_threads``
-- and #61 is the ticket that put it there. This script lists what is *not*
recorded yet.

Usage
-----
Run it on the machine whose provenance is in question::

    python tests/goldens/audit_environment.py

Probe the floating-point state under a run's own thread pin, not before it::

    python tests/goldens/audit_environment.py --pin-threads 1

The pin matters. A maths library can set flush-to-zero when it initialises,
and the audit's question is what the pipeline computed under, not what the
interpreter started under. ``--pin-threads`` needs ``dawis`` importable; the
rest of the script does not.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info


def section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def loaded_maths_libraries() -> list[dict]:
    """Every threaded maths library loaded, with its build and thread count.

    Named for what it returns rather than for BLAS alone: ``threadpool_info``
    reports OpenMP runtimes too, and on the audited machine ``libiomp`` shows
    up beside ``libmkl_rt``. Both are listed, because either can reassociate a
    sum.

    Golden provenance reads ``[machine].blas_vendor`` out of this same call
    and discards each entry's ``version`` and ``threading_layer``. Those two
    are the audit's finding, so they are printed here in full rather than
    summarised.
    """
    return threadpool_info()


def numpy_build() -> dict:
    """How numpy was compiled, and which SIMD kernels it can dispatch to.

    Two different questions, deliberately answered together, because reading
    either one alone misleads.

    ``show_config('dicts')`` is a public numpy API and reports *build* time:
    the compiler and its flags, the BLAS the wheel was linked against, and
    which SIMD extensions were compiled in. A ``-ffast-math`` in there would
    be a finding on its own; on the audited machine there is none.

    ``__cpu_features__`` reports *run* time: what this CPU actually offers. A
    wheel compiled with AVX-512 kernels runs SSE ones on a machine without
    AVX-512, and only the intersection of the two is the code that executed.
    That is the fact F1 measures, and no public API exposes it -- hence the
    private import, guarded so a numpy that renames it again degrades to a
    missing field rather than to a crash.
    """
    config = np.show_config("dicts")
    compilers = config.get("Compilers", {})

    try:
        from numpy._core._multiarray_umath import __cpu_features__
        cpu_offers = sorted(name for name, present in __cpu_features__.items() if present)
    except ImportError:
        cpu_offers = None

    return {
        "simd_compiled_in": config.get("SIMD Extensions"),
        "cpu_offers_at_runtime": cpu_offers,
        "blas_linked_against": config.get("Build Dependencies", {}).get("blas"),
        "c_compiler": {
            key: compilers.get("c", {}).get(key) for key in ("name", "version", "args")
        },
    }


#: Every third-party package a run can import that has a say in what it
#: computes. ``matplotlib`` and ``psutil`` are absent on purpose: #46 moved
#: the drawing out of the loop, and psutil only measures. ``ray`` is in the
#: list not because it computes anything itself but because it is the backend
#: that hands out the work -- an unverified candidate (F9), and one this
#: script should report rather than assume absent.
#:
#: Keep this list and the ``[dependencies]`` table proposed in
#: ``provenance-audit.md`` identical. Two hand-maintained copies of one list
#: are one copy too many, and the audit is the thing that would go stale.
RUN_INFLUENCING_PACKAGES = (
    "numpy",
    "scipy",
    "astropy",
    "photutils",
    "skimage",
    "h5py",
    "threadpoolctl",
    "ray",
)


def package_versions() -> dict:
    """The installed version of every package a run's result can depend on.

    ``None`` means not installed, which is a real answer: ``ray`` is absent on
    the audited machine and the serial backend is what ran. HDF5's own
    library version rides along with ``h5py``, because the two move
    independently.
    """
    out = {}
    for name in RUN_INFLUENCING_PACKAGES:
        try:
            out[name] = __import__(name).__version__
        except ImportError:
            out[name] = None
    try:
        import h5py
        out["hdf5"] = h5py.version.hdf5_version
    except ImportError:
        out["hdf5"] = None
    return out


def floating_point_state() -> dict:
    """The platform's floating-point defaults, as behaviour rather than flags.

    MXCSR is not readable portably from Python, so each default is probed by
    the arithmetic it changes: a subnormal that survives a multiply means
    flush-to-zero is off, and ``1 + 2**-53 == 1`` means round-to-nearest-even
    is in force. ``geterr`` is numpy's own error policy, which is a separate
    thing from the hardware's and is recorded beside them for the same reason.
    """
    subnormal = np.array([1e-40], dtype=np.float32)
    return {
        "subnormals_survive": bool((subnormal * np.float32(1.0))[0] != 0.0),
        "round_to_nearest_even": float(np.float64(1.0) + np.float64(2.0**-53)) == 1.0,
        "numpy_errstate": np.geterr(),
    }


def thread_environment() -> dict:
    """Maths-library and dispatch variables present in the shell.

    Diagnosis only, for the thread counts: ``threadpoolctl`` overrides those
    outright, so a value set in a shell no longer decides anything -- which is
    exactly the claim ``tests/conftest.py`` asserts rather than assumes.

    The last two are different. ``MKL_ENABLE_INSTRUCTIONS`` and
    ``NPY_DISABLE_CPU_FEATURES`` are the levers F1 and F3 pull, and nothing in
    DAWIS overrides either. A value in one of those is a run computing
    something other than what the machine would otherwise give it.
    """
    names = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "MKL_THREADING_LAYER",
        "MKL_ENABLE_INSTRUCTIONS",
        "NPY_DISABLE_CPU_FEATURES",
    )
    return {name: os.environ.get(name) for name in names}


@contextlib.contextmanager
def pinned(n_threads: int | None):
    """Hold DAWIS's own thread pin open, or do nothing if none was asked for.

    The floating-point probe is the only reason this exists, and the reason it
    is opt-in: importing ``dawis`` is not free, and every other fact here is
    the same pinned or not.
    """
    if n_threads is None:
        yield None
        return
    # The recording scripts beside this one take a --source and import the
    # package out of it. This one has no algorithm to pin down, so it settles
    # for the checkout it lives in -- which is also the only checkout whose
    # environment the audit is asking about.
    repo = str(Path(__file__).resolve().parent.parent.parent)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from dawis.parallel import parallel_backend
    with parallel_backend("serial", n_threads=n_threads) as backend:
        # Give the library something to initialise on, so a flush-to-zero it
        # sets on first use is set before the probe reads it.
        np.random.rand(200, 200) @ np.random.rand(200, 200)
        yield backend


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--pin-threads",
        type=int,
        default=None,
        metavar="N",
        help="probe the floating-point state inside a DAWIS thread pin of N "
             "threads, which is the state a run actually computes under "
             "(default: probe the bare interpreter)",
    )
    args = parser.parse_args(argv)

    print("DAWIS environment audit (#62)")
    print("=" * 29)

    section("Platform")
    print(json.dumps({
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "n_cpus_machine": os.cpu_count(),
    }, indent=2))

    section("Versions of every package a run's result can depend on")
    print(json.dumps(package_versions(), indent=2))

    section("Loaded maths libraries (vendor, version, threading layer)")
    print(json.dumps(loaded_maths_libraries(), indent=2))

    section("numpy build and CPU dispatch")
    print(json.dumps(numpy_build(), indent=2))

    section("Floating-point defaults (pinned: %s)" % (args.pin_threads,))
    with pinned(args.pin_threads):
        print(json.dumps(floating_point_state(), indent=2))

    section("Environment variables (thread counts diagnostic; dispatch is not)")
    print(json.dumps(thread_environment(), indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
