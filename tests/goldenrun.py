"""What a golden run *is*, shared by every tier that records one.

Two tiers record goldens from the same pipeline and compare them the same way:
Tier 1 on a 256x256 cutout every commit (#36), Tier 2 on the full 900x900
gallery image at the shipping solver setting (#37). What differs between them
is the input, the pinned solver knob and the budget — nothing about how a run
is driven or how its output is read back off disk. All of that lives here, so
the two tiers cannot drift apart in the machinery and only differ where they
mean to.

Three callers import this module and must agree exactly: the two comparison
tests under ``tests/regression/``, and the two hand-run recording scripts under
``tests/goldens/``. A golden therefore cannot be recorded under one convention
and compared under another.

Nothing here imports pytest: the recording scripts load this file directly,
outside a test session.
"""

from __future__ import annotations

import csv
import inspect
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The legacy tree imports matplotlib at module scope and its pipeline plots the
# convergence trace at the end of every run. Set before anything imports dawis,
# because matplotlib reads this only once. The 1.0 tree neither imports nor
# draws (#46), but a golden recorded on `legacy-seeded` is compared against a
# run of this tree, so both must be drivable from a headless session.
os.environ.setdefault("MPLBACKEND", "Agg")

#: Reference image shipped inside the package, used by Tier 1 and Tier 2.
GALLERY_IMAGE = "tidal_group_f814w_sci.fits"

#: The one seed every seeded test uses. Explicit seeds are for tests only;
#: the science default stays ``seed=None`` (#3).
TEST_SEED = 20260823

#: Cap on the greedy outer loop — an unrelated knob that shares its name with
#: the conjugate-gradient solver's. The published default; both tiers converge
#: well inside it, so both run to convergence rather than to a cap.
PIPELINE_MAX_ITER = 500

#: The published pipeline configuration both tiers run: ``test.py``'s parameter
#: set — the same one #5 profiled and #33 benchmarked — with the run made
#: deterministic and silent. Everything absent here takes the entry point's own
#: default, and each tier's ``provenance.toml`` records all of them resolved.
GOLDEN_PARAMS = {
    "n_workers": 1,
    "n_threads": 1,
    "n_levels": 9,
    "starting_level": 2,
    "tau": 0.1,
    "gamma": 0.5,
    "ceps": 1e-4,
    "min_span": 2,
    "max_span": 3,
    "lvl_sep_big": 5,
    "max_iter": PIPELINE_MAX_ITER,
    "data_dump": False,
    # Dropped from the entry point by #46, which took the animation off the
    # loop: kept here because `legacy-seeded` still has it, and a recording
    # made there must not spend half its time drawing frames. Filtered out
    # against the signature by :func:`golden_kwargs`.
    "gif": False,
    "resume": False,
}

#: The conjugate-gradient solver's iteration cap as *shipped* — the published
#: Ellien+2021 value that `7c5ba8e` cut to 3 and #32 reverted. Tier 2 runs it
#: because Tier 2 is the science check; Tier 1 pins a cheaper value because its
#: job is catching structural breakage, which it does equally well at any
#: iteration count. See :func:`pin_cg_max_iter`.
SHIPPING_CG_MAX_ITER = 200

#: Maths-library threads every golden is recorded and compared at (#61).
#:
#: One, and stated here rather than only inside ``GOLDEN_PARAMS`` because two
#: things need it that do not build a ``Config``: the legacy recording path,
#: which has no such knob and is pinned from the harness instead, and
#: ``conftest``'s assertion that the pin actually reaches the loaded library.
#:
#: Not a speed knob and not scheduling. numpy splits a single array operation
#: across threads on its own, and a sum split over four threads is added in a
#: different association order from one added on a single thread. On the Tier 1
#: cutout that moves an atom's flux in its last digit — 30.25294122005867
#: unpinned against 30.252941220058663 at one thread, from the same serial run.
#: Recording and comparing at one thread is what leaves the *input* as the only
#: thing a golden depends on. Before #61 this had to be an environment variable
#: set above the numpy import; now it is ``Config.n_threads`` and DAWIS applies
#: it itself, through the same code path a user runs.
GOLDEN_N_THREADS = 1


def resolve_n_threads(n_threads: int | None) -> int:
    """The thread count a golden run uses: the override, or the published one."""
    return GOLDEN_N_THREADS if n_threads is None else n_threads


@contextmanager
def pin_threads(n_threads: int):
    """Hold the maths libraries at ``n_threads`` for the duration of a block.

    For the *legacy* trees only. They predate ``Config.n_threads`` (#61), so
    nothing inside them can pin a thread count and the harness has to do it —
    which is exactly what makes a recording comparable against a 1.0 run that
    pinned the same number through its own configuration.

    The 1.0 tree must **not** be driven through this. What #61 has to prove is
    that a user setting ``n_threads`` gets it, and a harness that pinned the
    threads itself would prove that of the harness instead.
    """
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=n_threads):
        yield


def observed_threads_at(n_threads: int) -> tuple[int | None, str | None]:
    """What the loaded maths library actually threads at when asked for this.

    The *observation*, as against the request — and the two can disagree
    silently, which is the whole lesson of #61. Provenance records both, and
    this is the half that cannot be read off a configuration.

    Implemented against ``threadpoolctl`` here rather than borrowed from
    :func:`dawis.parallel.observed_threads` on purpose: the recording scripts
    ask this of a ``legacy-seeded`` checkout, which predates the function.
    A recording must be able to say what its own run threaded at.

    **The two must answer alike, and nothing enforces it.** A provenance file
    recorded through this one is compared against runs that report through the
    other, so a rule changed on one side — which library is preferred, what
    happens when several are loaded — silently makes the recorded observation
    describe something else. Change both, and
    ``test_the_two_thread_observers_answer_alike`` in
    ``tests/unit/test_parallel.py`` is what says so when only one is changed.

    Returns
    -------
    tuple of (int or None, str or None)
        The effective thread count and the BLAS vendor ``threadpoolctl``
        found. ``(None, None)`` where no threaded library is loaded at all,
        which is a real answer: a numpy built against a reference BLAS has no
        thread count to report.
    """
    from threadpoolctl import threadpool_info, threadpool_limits

    with threadpool_limits(limits=n_threads):
        pools = threadpool_info()
    if not pools:
        return None, None
    blas = [pool for pool in pools if pool.get("user_api") == "blas"]
    pool = (blas or pools)[0]
    threads = pool.get("num_threads")
    return (None if threads is None else int(threads)), pool.get("internal_api")


@contextmanager
def pin_cg_max_iter(dawis, max_iter: int):
    """Force the CG solver's iteration cap for the duration of a run.

    Both tiers force this rather than inheriting it, which removes from the
    comparison a default that differs per branch: 200 on ``legacy-seeded``, 3
    on the drifted tree the refactor branched from, and 200 again on the 1.0
    tree since #53 restored it as ``Config.cg_max_iter``.

    Patching the bound method beats editing the tree under test: the same
    pinned value then applies to whichever branch is running, which is what
    makes a golden recorded on ``legacy-seeded`` comparable at all. It also
    still wins where the caller passes the argument — ``restore_object`` does
    since #53 — because the wrapper sets the keyword after the caller has.
    """
    interscale_tree = dawis.interscale_tree
    original = interscale_tree.CG_minimization

    def patched(self, *args, **kwargs):
        kwargs["max_iter"] = max_iter
        return original(self, *args, **kwargs)

    interscale_tree.CG_minimization = patched
    try:
        yield
    finally:
        interscale_tree.CG_minimization = original


def effective_config(entry, passed: dict) -> dict:
    """Every pipeline parameter with the value a run actually used.

    Most knobs take the entry point's defaults, and several of those are live
    — ``deconv`` gates a Richardson-Lucy call inside the CG loop — so
    provenance records the resolved set rather than only what was passed.
    """
    config = {}
    for name, param in inspect.signature(entry).parameters.items():
        if name in passed:
            config[name] = passed[name]
        elif param.default is not inspect.Parameter.empty:
            config[name] = param.default
        else:
            config[name] = None
    return config


def is_array_in(dawis) -> bool:
    """Whether this tree has the array-in entry point of #54.

    Two APIs are driven from this file and both must keep working. The 1.0
    tree takes an array and a ``Config``; ``legacy`` and ``legacy-seeded`` —
    which every golden is *recorded* from — take fifteen keywords and a path.
    Branching on the capability rather than on a branch name means a
    recording made on either tree is still comparable against this one.

    The name alone is not the capability. ``legacy-seeded``'s package is built
    from star imports, and one of the modules it re-exports does
    ``from subprocess import *``, so ``dawis.run`` exists there and is
    :func:`subprocess.run`. A bare ``hasattr`` therefore reads the legacy tree
    as array-in and the recording dies one line later on a ``Config`` it has
    no name for. Asking where the function came from is what separates the
    entry point from the collision.
    """
    entry = getattr(dawis, "run", None)
    if not callable(entry):
        return False
    return getattr(entry, "__module__", "").split(".")[0] == "dawis"


def golden_config(dawis, cg_max_iter: int, seed: int = TEST_SEED,
                  n_workers: int | None = None, n_threads: int | None = None):
    """The published configuration as a ``Config``, for the array-in entry point.

    The same ``GOLDEN_PARAMS`` the keyword entry point is handed, filtered
    against the dataclass the way :func:`golden_kwargs` filters against a
    signature, so neither tier restates a knob.

    Two knobs may be replaced, and ``None`` keeps the published value of
    either. They are named rather than left to a general override because
    only these two may safely be replaced, and they are replaceable for
    opposite reasons:

    ``n_workers`` is scheduling since #56, so a run that changes it is the
    same science down to the bit — which is what makes the parallel half of
    #57 a comparison rather than two experiments.

    ``n_threads`` is *not* scheduling (#61). Changing it reassociates the
    floating-point sums and moves an atom's flux in its last digits, so two
    runs that differ in it are compared under the golden tolerances rather
    than exactly. What must not move is a decision — an atom count, a level, a
    bounding box, a convergence flag.
    """
    Config = dawis.Config
    knobs = {
        name: value
        for name, value in GOLDEN_PARAMS.items()
        if name in Config.__dataclass_fields__
    }
    if n_workers is not None:
        knobs["n_workers"] = n_workers
    if n_threads is not None:
        knobs["n_threads"] = n_threads
    # Forced rather than inherited, for the reason :func:`pin_cg_max_iter`
    # gives: the default differs per branch, so inheriting it would compare a
    # golden recorded at one value against a run at another.
    return Config(seed=seed, cg_max_iter=cg_max_iter, **knobs)


def golden_kwargs(dawis, work_dir: Path, infile: str,
                  seed: int = TEST_SEED, n_workers: int | None = None) -> dict:
    """The exact keyword arguments a golden run passes to the entry point.

    The keyword entry point, that is — the one ``legacy-seeded`` records
    from. The array-in tree builds a :func:`golden_config` instead.

    Kept separate from :func:`run_golden` so the recording scripts can write
    the resolved set into provenance without restating any of it.
    """
    accepted = inspect.signature(dawis.synthesis_by_analysis).parameters
    # The two tiers drive two branches whose entry points differ by a
    # parameter or two — `gif` exists on `legacy-seeded` and was removed by
    # #46, and the worker count is `n_workers` here and was spelt otherwise
    # before #60 — so the shared set is filtered against the signature in hand
    # rather than restated per branch. A knob the branch in hand has no name
    # for falls to that branch's default.
    kwargs = {
        name: value for name, value in GOLDEN_PARAMS.items() if name in accepted
    }
    kwargs.update(
        indir=str(work_dir),
        infile=infile,
        outdir=str(work_dir),
    )
    if n_workers is not None:
        # Injected rather than filtered, unlike everything above. Replacing
        # the worker count is an explicit request, so a branch whose entry
        # point has no such parameter has to say so — running the published
        # single worker and being compared as though it had not is the one
        # outcome that would make #57's bit-identity claim meaningless.
        if "n_workers" not in accepted:
            raise TypeError(
                "this tree's keyword entry point has no n_workers -- it "
                "predates the rename (#60), so its worker count cannot be "
                "replaced from here"
            )
        kwargs["n_workers"] = n_workers
    # `seed` is a parameter on `legacy-seeded` and landed on the 1.0 tree with
    # #53. Where it is absent the caller pins the global generator instead —
    # see :func:`run_golden`. Both branches produce the same draws: the 1.0
    # tree threads a `numpy.random.RandomState` built from this seed, which is
    # the same Mersenne Twister stream `numpy.random.seed` pins.
    if "seed" in accepted:
        kwargs["seed"] = seed
    return kwargs


@dataclass(frozen=True)
class GoldenRun:
    """One finished golden run, addressed by the files it wrote.

    The pipeline hangs every output off one path stem, so this names them
    rather than leaving callers to concatenate suffixes.
    """

    stem: Path

    @property
    def restored(self) -> Path:
        """The sum of every atom the run restored."""
        return Path(f"{self.stem}.restored.fits")

    @property
    def residuals(self) -> Path:
        """What the greedy loop left behind."""
        return Path(f"{self.stem}.residuals.fits")

    @property
    def convergence(self) -> Path:
        """The per-iteration convergence trace."""
        return Path(f"{self.stem}.cpar.txt")

    @property
    def atom_dumps(self) -> list[Path]:
        """The per-iteration atom pickles, in iteration order."""
        return sorted(self.stem.parent.glob(f"{self.stem.name}.ol.it*.pkl"))


def run_golden(dawis, image: np.ndarray, work_dir: Path, infile: str,
               cg_max_iter: int, seed: int = TEST_SEED,
               n_workers: int | None = None,
               n_threads: int | None = None) -> GoldenRun:
    """Run the published configuration on ``image`` and return what it wrote.

    On the 1.0 tree the image goes straight in as an array (#54), which is the
    whole point of the array-in entry point: the harness needs no filesystem
    to drive a run. ``infile`` then only names the run, so the output stem —
    and therefore every golden path — is unchanged.

    ``legacy`` and ``legacy-seeded`` read their input from a FITS file, so
    there ``image`` is written into ``work_dir`` first.

    ``n_workers`` replaces the published single worker; ``None`` keeps it. Every
    golden is recorded at one worker, so a run that passes anything else is no
    longer comparable against a recording — which is the point at the only
    call site that does, the parallel-consistency check of #57, where the two
    runs being compared are each other.

    ``n_threads`` replaces the maths-library thread count; ``None`` keeps
    :data:`GOLDEN_N_THREADS`. It reaches the run by whichever route the tree in
    hand has. The 1.0 tree owns the knob since #61, so it goes into the
    ``Config`` and DAWIS applies it — deliberately, because what the tests have
    to prove is that a *user* setting it gets it, and a harness that pinned the
    threads itself would prove that of the harness instead. The legacy trees
    have no such knob, so there the harness pins them with
    :func:`pin_threads`; that is what makes a recording comparable against a
    1.0 run at all.

    One thing to know when workers are in play: :func:`pin_cg_max_iter`
    patches the solver in *this* process only, so a backend that runs patches
    on workers leaves them unpatched. On the 1.0 tree that is harmless, because
    ``restore_object`` passes ``cg_max_iter`` down from the ``Config`` and
    :func:`golden_config` puts the pinned value there too — a worker reaches
    the same number by the other route. Both are set from the one argument
    here, which is what keeps them equal. The thread count reaches a worker by
    that same second route, through the ``Config``.
    """
    name = infile[:-5]

    if is_array_in(dawis):
        config = golden_config(dawis, cg_max_iter, seed, n_workers, n_threads)
        with pin_cg_max_iter(dawis, cg_max_iter):
            dawis.run(image, config, outdir=str(work_dir), name=name)
        return GoldenRun(work_dir / name)

    from astropy.io import fits

    fits.PrimaryHDU(np.asarray(image)).writeto(
        work_dir / infile, overwrite=True
    )

    kwargs = golden_kwargs(dawis, work_dir, infile, seed, n_workers)
    if "seed" not in kwargs:
        # One call, before any random draw — exactly what the pipeline does
        # with the argument on the branch that has it.
        np.random.seed(seed)

    with pin_threads(resolve_n_threads(n_threads)), \
            pin_cg_max_iter(dawis, cg_max_iter):
        dawis.synthesis_by_analysis(**kwargs)

    return GoldenRun(work_dir / name)


#: Per-atom columns of the golden catalogue. ``flux`` and ``peak`` are derived
#: from the atom image; everything else is stored on the atom itself.
ATOM_COLUMNS = (
    "iteration",
    "index",
    "level",
    "bbox_0",
    "bbox_1",
    "bbox_2",
    "bbox_3",
    "eccentricity",
    "filter",
    "flag_convergence",
    "sum_wr",
    "norm_wr",
    "flux",
    "peak",
    "det_err_flux",
)


def atom_catalogue(dawis, run: GoldenRun) -> list[dict]:
    """Read every atom the run wrote, in the order the run wrote them.

    Atoms are read back off disk through the package's own reader rather than
    kept in memory, so this asserts what a run *produces* — which is what
    survives the carve — rather than what one function returned.
    """
    rows = []
    for dump in run.atom_dumps:
        iteration = int(dump.name.split(".ol.it")[1][:3])
        for index, atom in enumerate(dawis.read_ol_from_pickle(str(dump))):
            rows.append({
                "iteration": iteration,
                "index": index,
                "level": int(atom.level),
                "bbox_0": int(atom.bbox[0]),
                "bbox_1": int(atom.bbox[1]),
                "bbox_2": int(atom.bbox[2]),
                "bbox_3": int(atom.bbox[3]),
                "eccentricity": float(atom.eccentricity),
                "filter": str(atom.filter),
                "flag_convergence": int(bool(atom.flag_convergence)),
                "sum_wr": float(atom.sum_wr),
                "norm_wr": float(atom.norm_wr),
                "flux": float(np.sum(atom.image)),
                "peak": float(np.max(atom.image)),
                "det_err_flux": float(np.sum(atom.det_err_image)),
            })
    return rows


def _digest_of_array(array) -> str:
    """Digest an atom image's bytes, so two of them can be compared exactly.

    ``tobytes`` on a contiguous copy, so that two arrays holding the same
    numbers digest the same however each was strided when it was built.
    """
    import hashlib

    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(
        b"%s|%s|" % (str(contiguous.dtype).encode(),
                     str(contiguous.shape).encode())
        + contiguous.tobytes()
    ).hexdigest()


#: The two columns :func:`atom_catalogue_with_digests` adds to a catalogue row.
ATOM_DIGEST_COLUMNS = ("image_sha256", "det_err_image_sha256")


def atom_catalogue_with_digests(dawis, run: GoldenRun) -> list[dict]:
    """:func:`atom_catalogue`, plus a digest of each atom's two images.

    The catalogue describes an atom by measurements taken off its image —
    ``flux`` is a sum and ``peak`` is a maximum — which is the right
    description for a golden, where a tolerance is applied to them anyway.
    It is the wrong description for #57, which claims two runs produce the same
    *atom*: two different images can share a sum, a maximum and a bounding box,
    and would pass a comparison that only reads those.

    Digesting the bytes closes that gap without touching the golden tables,
    whose columns are recorded on disk and cannot gain one.

    Both this and :func:`atom_catalogue` walk ``run.atom_dumps`` in the same
    order, which is what lets the rows be zipped rather than joined.
    """
    rows = atom_catalogue(dawis, run)

    digests = []
    for dump in run.atom_dumps:
        for atom in dawis.read_ol_from_pickle(str(dump)):
            digests.append({
                "image_sha256": _digest_of_array(atom.image),
                "det_err_image_sha256": _digest_of_array(atom.det_err_image),
            })

    return [dict(row, **digest)
            for row, digest in zip(rows, digests, strict=True)]


#: Per-iteration columns of the golden convergence table, in the order the
#: pipeline writes them into ``<stem>.cpar.txt``. The file's own header line
#: names only seven of the nine fields it writes, so this is the authority.
CONVERGENCE_COLUMNS = (
    "level",
    "iteration",
    "window_normeps",
    "normeps",
    "eps",
    "flux_rec",
    "flux_res",
    "n_atoms",
)

#: The one field of ``cpar.txt`` this does not record: the iteration's wall
#: time, the only number in the file that is not a function of the input.
CONVERGENCE_FIELDS = len(CONVERGENCE_COLUMNS) + 1

#: Convergence columns compared by tolerance rather than exactly.
CONVERGENCE_APPROXIMATE = (
    "window_normeps", "normeps", "eps", "flux_rec", "flux_res",
)


def convergence_table(run: GoldenRun) -> list[dict]:
    """Read the run's convergence trace, one row per greedy iteration."""
    rows = []
    with open(run.convergence) as handle:
        for number, line in enumerate(handle, start=1):
            if line.startswith("#") or not line.strip():
                continue
            fields = line.split()
            if len(fields) != CONVERGENCE_FIELDS:
                # Silently zipping a changed column count would misalign every
                # row instead of failing, and the goldens would follow.
                raise ValueError(
                    "%s line %d: expected %d fields, found %d — the pipeline's "
                    "convergence columns have changed"
                    % (run.convergence, number, CONVERGENCE_FIELDS, len(fields))
                )
            row = dict(zip(CONVERGENCE_COLUMNS, fields))
            row["level"] = int(row["level"])
            row["iteration"] = int(row["iteration"])
            row["n_atoms"] = int(row["n_atoms"])
            for name in CONVERGENCE_APPROXIMATE:
                row[name] = float(row[name])
            rows.append(row)
    return rows


#: Golden-table columns whose values are compared exactly, never by tolerance.
INTEGER_COLUMNS = frozenset({
    "iteration", "index", "level", "n_atoms", "flag_convergence",
    "bbox_0", "bbox_1", "bbox_2", "bbox_3",
})

#: Golden-table columns holding text rather than a number.
TEXT_COLUMNS = frozenset({"filter"})


def write_table(path: Path, columns, rows) -> None:
    """Write a golden table as CSV with a fixed column order."""
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def read_table(path: Path, columns) -> list[dict]:
    """Read a golden table back, restoring each column's type."""
    rows = []
    with open(path, newline="") as handle:
        for raw in csv.DictReader(handle):
            row = {}
            for name in columns:
                value = raw[name]
                if name in INTEGER_COLUMNS:
                    row[name] = int(value)
                elif name in TEXT_COLUMNS:
                    row[name] = value
                else:
                    row[name] = float(value)
            rows.append(row)
    return rows


def robust_sigma(values) -> float:
    """Half the 16-84 percentile spread — the noise scale #3's atol rests on."""
    low, high = np.percentile(values, [16, 84])
    return float((high - low) / 2.0)


def sha256_of(path: Path) -> str:
    """Digest a file, so provenance can name the exact input it was fed."""
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
