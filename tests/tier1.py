"""The Tier 1 run definition (#36).

One seeded run of the published configuration on a 256x256 patch of the
gallery image. Everything about that run lives here — the cutout bounds, the
seed, the pinned solver knob, the pipeline parameters, and how the run's
output is read back off disk.

Two things drive a Tier 1 run and they must agree exactly: the comparison test
in ``tests/regression/test_golden_cutout.py``, and the hand-run recording
script ``tests/goldens/record_cutout_golden.py``. Both import this module, so
a golden cannot be recorded under one configuration and compared under
another.

Nothing here imports pytest: the recording script loads this file directly,
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

# The legacy tree imports matplotlib at module scope and the pipeline plots its
# convergence trace at the end of every run. Set before anything imports dawis,
# because matplotlib reads this only once.
os.environ.setdefault("MPLBACKEND", "Agg")

#: Reference image shipped inside the package, used by Tier 1 and Tier 2.
GALLERY_IMAGE = "tidal_group_f814w_sci.fits"

#: Centre of the Tier 1 cutout, as ``(x, y)``, supplied by Amaël on #3.
CUTOUT_CENTRE = (222, 242)

#: Side of the square Tier 1 cutout, in pixels.
CUTOUT_SIZE = 256

#: The resulting bounds, ``((y0, y1), (x0, x1))`` == ``((114, 370), (94, 350))``.
#:
#: Centre, size and bounds are all stated because the FITS array is indexed
#: ``(ny, nx)`` while the coordinates are ``(x, y)`` — so every misreading of
#: those two numbers still lands on a valid source-rich patch, and produces a
#: different golden *silently* rather than failing. The recorded
#: ``input_cutout.fits`` is compared for the same reason.
CUTOUT_SLICE = (
    (CUTOUT_CENTRE[1] - CUTOUT_SIZE // 2, CUTOUT_CENTRE[1] + CUTOUT_SIZE // 2),
    (CUTOUT_CENTRE[0] - CUTOUT_SIZE // 2, CUTOUT_CENTRE[0] + CUTOUT_SIZE // 2),
)

#: The one seed every seeded test uses. Explicit seeds are for tests only;
#: the science default stays ``seed=None`` (#3).
TEST_SEED = 20260823

#: Iteration cap of the conjugate-gradient solver inside
#: ``interscale_tree.CG_minimization``.
#:
#: **A speed knob, not a science claim.** Tier 1's job is catching structural
#: breakage during the carve, which it does equally well at any iteration
#: count; the shipping value of 200 costs about six times more per atom (#33)
#: and would push this tier past its three-minute budget. Tier 2 (#37) runs
#: the shipping value and is the science check.
#:
#: ``restore_object`` never passes this argument, so the value in effect is
#: whatever the method's default happens to be — 200 on ``legacy-seeded``, 3
#: on the drifted tree the refactor branched from. Tier 1 therefore forces it
#: rather than inheriting it, which also removes that drift from the
#: comparison. See :func:`pin_cg_max_iter`.
CG_MAX_ITER = 3

#: Cap on the greedy outer loop — an unrelated knob that shares the name. The
#: published default; the cutout converges well inside it, so Tier 1 runs to
#: convergence rather than to a cap.
PIPELINE_MAX_ITER = 500

#: Name of the FITS file the cutout is written to before every Tier 1 run.
TIER1_INFILE = "tier1_cutout.fits"

#: The Tier 1 pipeline configuration: ``test.py``'s published parameter set —
#: the same one #5 profiled and #33 benchmarked — with the run made
#: deterministic and silent. Everything absent here takes the entry point's
#: own default, and ``provenance.toml`` records all of them resolved.
TIER1_PARAMS = {
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
    "gif": False,
    "resume": False,
}


def cutout_of(image: np.ndarray) -> np.ndarray:
    """Slice the agreed Tier 1 cutout out of the full gallery image."""
    (y0, y1), (x0, x1) = CUTOUT_SLICE
    return image[y0:y1, x0:x1]


@contextmanager
def pin_cg_max_iter(dawis, max_iter: int = CG_MAX_ITER):
    """Force the CG solver's iteration cap for the duration of a run.

    Patching the bound method beats editing the tree under test: the same
    pinned value then applies to whichever branch is running, which is what
    makes a golden recorded on ``legacy-seeded`` comparable at all.
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


def tier1_kwargs(dawis, work_dir: Path, seed: int = TEST_SEED) -> dict:
    """The exact keyword arguments a Tier 1 run passes to the entry point.

    Kept separate from :func:`run_tier1` so the recording script can write the
    resolved set into provenance without restating any of it.
    """
    kwargs = dict(
        TIER1_PARAMS,
        indir=str(work_dir),
        infile=TIER1_INFILE,
        outdir=str(work_dir),
    )
    # `seed` is a parameter on `legacy-seeded` and lands on the 1.0 tree in
    # #53. Where it is absent the caller pins the global generator instead —
    # see :func:`run_tier1`.
    if "seed" in inspect.signature(dawis.synthesis_by_analysis).parameters:
        kwargs["seed"] = seed
    return kwargs


@dataclass(frozen=True)
class Tier1Run:
    """One finished Tier 1 run, addressed by the files it wrote.

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


def run_tier1(dawis, image: np.ndarray, work_dir: Path, seed: int = TEST_SEED,
              cg_max_iter: int = CG_MAX_ITER) -> Tier1Run:
    """Run the Tier 1 configuration on ``image`` and return what it wrote.

    The pipeline entry point reads its input from a FITS file, so the cutout
    is written into ``work_dir`` first.
    """
    from astropy.io import fits

    fits.PrimaryHDU(np.asarray(image)).writeto(
        work_dir / TIER1_INFILE, overwrite=True
    )

    kwargs = tier1_kwargs(dawis, work_dir, seed)
    if "seed" not in kwargs:
        # One call, before any random draw — exactly what the pipeline does
        # with the argument on the branch that has it.
        np.random.seed(seed)

    with pin_cg_max_iter(dawis, cg_max_iter):
        dawis.synthesis_by_analysis(**kwargs)

    return Tier1Run(work_dir / TIER1_INFILE[:-5])


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


def atom_catalogue(dawis, run: Tier1Run) -> list[dict]:
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


def convergence_table(run: Tier1Run) -> list[dict]:
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
            for name in ("window_normeps", "normeps", "eps", "flux_rec", "flux_res"):
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
