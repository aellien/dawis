#!/usr/bin/env python
"""Record the Tier 1 cutout goldens (#36).

Hand-run tool, not a pytest tier. One invocation runs the Tier 1
configuration on the agreed 256x256 cutout of the gallery image, copies the
outputs into ``tests/goldens/tier1_cutout/`` and writes the ``provenance.toml``
beside them.

The goldens must come from the *published* algorithm, so the run is driven
against a ``legacy-seeded`` checkout (#32) rather than against whatever tree
this script happens to sit in::

    python tests/goldens/record_cutout_golden.py \\
        --source ~/dawis-legacy-seeded \\
        --reason "initial Tier 1 recording for #36"

The run definition itself — cutout bounds, seed, pinned CG ``max_iter``,
pipeline parameters — lives in ``tests/tier1.py`` and is imported from there,
so the recording and the comparison can never drift apart. Everything the two
tiers' recorders share lives in ``tests/goldens/recording.py``.

Goldens are a ratchet, not a shrine. Re-run this only in a commit whose
message says why the golden moved.
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import recording
from recording import HERE

TICKET = "#36"
TICKET_URL = "https://github.com/aellien/dawis-dev/issues/36"


def main():
    parser = recording.build_parser(__doc__, TICKET, HERE / "tier1_cutout")
    args = parser.parse_args()

    # Importing the tier module also pins a headless matplotlib backend, which
    # the legacy tree needs before anything imports dawis.
    tier1 = recording.load_tier("tier1")
    import goldenrun

    source, state = recording.resolve_source(args, TICKET)
    stubbed = recording.stub_missing()
    dawis = recording.import_dawis(source)

    import numpy as np
    from astropy.io import fits

    image_path = Path(dawis.__file__).parent / "gallery" / goldenrun.GALLERY_IMAGE
    full = fits.getdata(image_path)
    cut = np.array(tier1.cutout_of(full))

    work_dir = recording.work_dir_for(args, "dawis-tier1-")

    print("recording Tier 1 goldens")
    print("  source     %s  (%s)" % (source, state["describe"]))
    print("  image      %s %s" % (image_path, list(full.shape)))
    print("  cutout     y %s  x %s" % tier1.CUTOUT_SLICE)
    print("  seed       %d" % goldenrun.TEST_SEED)
    print("  CG max_iter %d (pinned speed knob)" % tier1.CG_MAX_ITER)
    print("  work dir   %s" % work_dir)

    started = time.time()
    run = tier1.run_tier1(dawis, cut, work_dir)
    wall = time.time() - started

    atoms = goldenrun.atom_catalogue(dawis, run)
    convergence = goldenrun.convergence_table(run)
    sigma = goldenrun.robust_sigma(fits.getdata(run.residuals))

    goldens = Path(args.goldens)
    goldens.mkdir(parents=True, exist_ok=True)

    fits.PrimaryHDU(cut).writeto(goldens / "input_cutout.fits", overwrite=True)
    shutil.copyfile(run.restored, goldens / "restored.fits")
    shutil.copyfile(run.residuals, goldens / "residuals.fits")
    goldenrun.write_table(goldens / "atoms.csv", goldenrun.ATOM_COLUMNS, atoms)
    goldenrun.write_table(
        goldens / "convergence.csv", goldenrun.CONVERGENCE_COLUMNS, convergence
    )

    # #3 ties the image tolerance to the noise of the golden run's own
    # residual, not to the raw input: 1e-3 of a sigma is far below anything a
    # reader could see and far above float64 reassociation noise.
    image_atol = 1e-3 * sigma
    # The relative half of the pair, and the per-atom tolerance, are #3's
    # stated starting points, kept verbatim. Dropping rtol would leave the
    # brightest pixels guarded at about 3e-6 relative, which is exactly where
    # BLAS reassociation on another machine lives -- and #3 chose float
    # tolerance over bit-for-bit precisely to survive that.
    image_rtol = 1e-5
    atom_rtol = 1e-4

    lines = [
        "# Provenance of the Tier 1 cutout goldens (#36).",
        "#",
        "# These pin the published Ellien+2021 behaviour on a 256x256 patch and",
        "# gate every commit of the 1.0 carve. Regenerate them only in a commit",
        "# whose message says why the golden moved (#3).",
        "",
    ]
    lines += recording.golden_table(
        1, recording.author_of(args), args.reason, TICKET_URL, wall
    )
    lines += recording.source_table(state, stubbed)
    lines += recording.toml_table("input", {
        "image": "dawis/gallery/%s" % goldenrun.GALLERY_IMAGE,
        "image_shape": list(full.shape),
        "image_dtype": str(full.dtype),
        "cutout_centre_xy": list(tier1.CUTOUT_CENTRE),
        "cutout_size": tier1.CUTOUT_SIZE,
        "cutout_half_width": tier1.CUTOUT_SIZE // 2,
        "cutout_slice_y": list(tier1.CUTOUT_SLICE[0]),
        "cutout_slice_x": list(tier1.CUTOUT_SLICE[1]),
        "cutout_expression": "image[114:370, 94:350]",
        "cutout_dtype": str(cut.dtype),
    }, comment=(
        "Amaël supplied (222, 242) on #3 as the CENTRE of the cutout, in (x, y).\n"
        "The FITS array is (ny, nx), so x is the SECOND axis. Centre, half-width\n"
        "and the resulting bounds are all recorded because every misreading of\n"
        "those two numbers still lands on a valid source-rich patch, and so\n"
        "produces a different golden silently rather than failing. This is the\n"
        "one full statement of the convention; the code points here."
    ))
    lines += recording.toml_table("run", {
        "seed": goldenrun.TEST_SEED,
        "n_workers": 1,
        "n_threads": goldenrun.GOLDEN_N_THREADS,
        "cg_max_iter": tier1.CG_MAX_ITER,
        "cg_max_iter_default_in_source": recording.cg_max_iter_default(dawis),
        "pipeline_max_iter": goldenrun.PIPELINE_MAX_ITER,
        "n_iterations": len(convergence),
        "n_atoms": len(atoms),
    }, comment=(
        "cg_max_iter is the conjugate-gradient solver's cap inside\n"
        "interscale_tree.CG_minimization, and it is A SPEED KNOB, NOT A SCIENCE\n"
        "CLAIM. Tier 1 catches structural breakage during the carve, which it\n"
        "does equally well at any iteration count; the shipping 200 costs about\n"
        "six times more per atom (#33) and would break this tier's three-minute\n"
        "budget. Tier 2 (#37) runs the shipping value and is the science check.\n"
        "pipeline_max_iter is the unrelated cap on the greedy outer loop; the\n"
        "cutout converges well inside it, so this run converged rather than\n"
        "hitting the cap."
        "\n"
        "n_threads is what this run asked the maths libraries to split a single\n"
        "array operation across, per process -- a different knob from n_workers,\n"
        "which hands whole trees to other processes (#61). It is recorded here\n"
        "because a golden compared at another thread count differs in its last\n"
        "digits for a reason that has nothing to do with the code under test.\n"
        "What the loaded library ACTUALLY gave is in [machine]; the request and\n"
        "the observation are separate facts."
    ))
    lines += recording.config_table(dawis, work_dir, tier1.TIER1_INFILE, tier1.CG_MAX_ITER)
    lines += recording.toml_table("tolerances", {
        "residual_sigma": sigma,
        "image_atol": image_atol,
        "image_rtol": image_rtol,
        "atom_rtol": atom_rtol,
        "atom_atol": 0.0,
        "convergence_rtol": atom_rtol,
        "convergence_atol": 0.0,
        "exact": ["n_atoms", "n_iterations", "level", "bbox", "flag_convergence",
                  "filter"],
    }, comment=(
        "Per #3: float tolerance on images and per-atom quantities, exact on\n"
        "atom and iteration counts. The values are #3's stated starting points\n"
        "-- rtol=1e-5 with atol=1e-3*sigma on images, rtol=1e-4 per atom -- and\n"
        "the sigma is the robust sigma of THIS run's residual, recorded above,\n"
        "not of the raw input. Never bit-for-bit: BLAS, OS, numpy versions and\n"
        "the maths-library thread count all move the last bits, which is why\n"
        "[run].n_threads is recorded above. Tier 2's worker-count check (#57)\n"
        "does compare two runs bit for bit, but that is a narrower claim than\n"
        "it reads: one machine, one BLAS build, one thread count, with the\n"
        "worker count as the only variable (#61). It is not a property of the\n"
        "goldens and no golden comparison rests on it."
    ))
    lines += recording.dependencies_table()
    lines += recording.machine_table(goldenrun.GOLDEN_N_THREADS)

    (goldens / "provenance.toml").write_text("\n".join(lines))

    print("\nrecorded into %s" % goldens)
    print("  iterations %d, atoms %d, wall %.1f s" % (
        len(convergence), len(atoms), wall))
    print("  residual sigma %.6g -> image_atol %.6g" % (sigma, image_atol))
    if args.work_dir is None:
        print("  scratch run left in %s" % work_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
