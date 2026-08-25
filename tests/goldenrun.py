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
    "n_cpus": 1,
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
    """
    return hasattr(dawis, "run")


def golden_config(dawis, cg_max_iter: int, seed: int = TEST_SEED):
    """The published configuration as a ``Config``, for the array-in entry point.

    The same ``GOLDEN_PARAMS`` the keyword entry point is handed, filtered
    against the dataclass the way :func:`golden_kwargs` filters against a
    signature, so neither tier restates a knob.
    """
    Config = dawis.Config
    knobs = {
        name: value
        for name, value in GOLDEN_PARAMS.items()
        if name in Config.__dataclass_fields__
    }
    # Forced rather than inherited, for the reason :func:`pin_cg_max_iter`
    # gives: the default differs per branch, so inheriting it would compare a
    # golden recorded at one value against a run at another.
    return Config(seed=seed, cg_max_iter=cg_max_iter, **knobs)


def golden_kwargs(dawis, work_dir: Path, infile: str,
                  seed: int = TEST_SEED) -> dict:
    """The exact keyword arguments a golden run passes to the entry point.

    The keyword entry point, that is — the one ``legacy-seeded`` records
    from. The array-in tree builds a :func:`golden_config` instead.

    Kept separate from :func:`run_golden` so the recording scripts can write
    the resolved set into provenance without restating any of it.
    """
    accepted = inspect.signature(dawis.synthesis_by_analysis).parameters
    # The two tiers drive two branches whose entry points differ by a
    # parameter or two — `gif` exists on `legacy-seeded` and was removed by
    # #46 — so the shared set is filtered against the signature in hand
    # rather than restated per branch.
    kwargs = {
        name: value for name, value in GOLDEN_PARAMS.items() if name in accepted
    }
    kwargs.update(
        indir=str(work_dir),
        infile=infile,
        outdir=str(work_dir),
    )
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
               cg_max_iter: int, seed: int = TEST_SEED) -> GoldenRun:
    """Run the published configuration on ``image`` and return what it wrote.

    On the 1.0 tree the image goes straight in as an array (#54), which is the
    whole point of the array-in entry point: the harness needs no filesystem
    to drive a run. ``infile`` then only names the run, so the output stem —
    and therefore every golden path — is unchanged.

    ``legacy`` and ``legacy-seeded`` read their input from a FITS file, so
    there ``image`` is written into ``work_dir`` first.
    """
    name = infile[:-5]

    if is_array_in(dawis):
        config = golden_config(dawis, cg_max_iter, seed)
        with pin_cg_max_iter(dawis, cg_max_iter):
            dawis.run(image, config, outdir=str(work_dir), name=name)
        return GoldenRun(work_dir / name)

    from astropy.io import fits

    fits.PrimaryHDU(np.asarray(image)).writeto(
        work_dir / infile, overwrite=True
    )

    kwargs = golden_kwargs(dawis, work_dir, infile, seed)
    if "seed" not in kwargs:
        # One call, before any random draw — exactly what the pipeline does
        # with the argument on the branch that has it.
        np.random.seed(seed)

    with pin_cg_max_iter(dawis, cg_max_iter):
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
