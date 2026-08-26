#!/usr/bin/env python
"""Record the Tier 2 full-image goldens (#37).

Hand-run tool, not a pytest tier. One invocation runs the published
configuration on the whole 900x900 gallery image at the shipping CG
``max_iter=200``, copies the outputs into ``tests/goldens/tier2_gallery/`` and
writes the ``provenance.toml`` beside them.

This is the science check, so the goldens must come from the *published*
algorithm: the run is driven against a ``legacy-seeded`` checkout (#32) rather
than against whatever tree this script happens to sit in::

    python tests/goldens/record_gallery_golden.py \\
        --source ~/dawis-legacy-seeded \\
        --reason "initial Tier 2 recording for #37"

Budget a full half hour on a 2017 laptop — 1641 s measured. The figure lands
in ``provenance.toml`` under ``[budget]`` so checkpoint scheduling can be
planned from a number rather than a guess.

The run definition — the shipping CG ``max_iter``, the seed, the pipeline
parameters — lives in ``tests/tier2.py`` and is imported from there, so the
recording and the comparison can never drift apart. Everything the two tiers'
recorders share lives in ``tests/goldens/recording.py``.

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

TICKET = "#37"
TICKET_URL = "https://github.com/aellien/dawis-dev/issues/37"


def main():
    parser = recording.build_parser(__doc__, TICKET, HERE / "tier2_gallery")
    args = parser.parse_args()

    # Importing the tier module also pins a headless matplotlib backend, which
    # the legacy tree needs before anything imports dawis.
    tier2 = recording.load_tier("tier2")
    import goldenrun

    source, state = recording.resolve_source(args, TICKET)
    stubbed = recording.stub_missing()
    dawis = recording.import_dawis(source)

    from astropy.io import fits

    image_path = Path(dawis.__file__).parent / "gallery" / goldenrun.GALLERY_IMAGE
    image = fits.getdata(image_path)
    # The image ships inside the package and is already under version control,
    # so it is fingerprinted rather than copied — a second 3 MB of FITS in the
    # goldens directory would pin nothing the digest does not.
    image_sha256 = goldenrun.sha256_of(image_path)

    work_dir = recording.work_dir_for(args, "dawis-tier2-")

    print("recording Tier 2 goldens")
    print("  source     %s  (%s)" % (source, state["describe"]))
    print("  image      %s %s" % (image_path, list(image.shape)))
    print("  sha256     %s" % image_sha256)
    print("  seed       %d" % goldenrun.TEST_SEED)
    print("  CG max_iter %d (the shipping value — this is the science check)"
          % tier2.CG_MAX_ITER)
    print("  work dir   %s" % work_dir)
    print("  expect about half an hour (1641 s measured on a 2017 laptop)")

    started = time.time()
    run = tier2.run_tier2(dawis, image, work_dir)
    wall = time.time() - started

    atoms = goldenrun.atom_catalogue(dawis, run)
    convergence = goldenrun.convergence_table(run)
    sigma = goldenrun.robust_sigma(fits.getdata(run.residuals))

    goldens = Path(args.goldens)
    goldens.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(run.restored, goldens / "restored.fits")
    shutil.copyfile(run.residuals, goldens / "residuals.fits")
    goldenrun.write_table(goldens / "atoms.csv", goldenrun.ATOM_COLUMNS, atoms)
    goldenrun.write_table(
        goldens / "convergence.csv", goldenrun.CONVERGENCE_COLUMNS, convergence
    )

    # Same rule as Tier 1, per #3: the image tolerance is tied to the noise of
    # this run's own residual, not to the raw input. The absolute half is
    # therefore a different number here than at Tier 1 — the full image and the
    # patch do not have the same residual noise — while the rule producing it
    # is identical.
    image_atol = 1e-3 * sigma
    image_rtol = 1e-5
    atom_rtol = 1e-4

    lines = [
        "# Provenance of the Tier 2 full-image goldens (#37).",
        "#",
        "# These pin the published Ellien+2021 behaviour on the whole gallery",
        "# image at the shipping CG max_iter. This is the science check: it runs",
        "# at named carve checkpoints and nightly in CI (#19), not per commit.",
        "# Regenerate only in a commit whose message says why the golden moved",
        "# (#3).",
        "",
    ]
    lines += recording.golden_table(
        2, recording.author_of(args), args.reason, TICKET_URL, wall
    )
    lines += recording.source_table(state, stubbed)
    lines += recording.toml_table("input", {
        "image": "dawis/gallery/%s" % goldenrun.GALLERY_IMAGE,
        "image_shape": list(image.shape),
        "image_dtype": str(image.dtype),
        "image_sha256": image_sha256,
        "cutout": False,
    }, comment=(
        "The whole gallery image, unsliced — there is no cutout convention to\n"
        "misread here, which is why Tier 2 records a digest of the packaged\n"
        "file instead of a second copy of it. The comparison checks the digest\n"
        "first, so a changed input fails loudly rather than shifting the golden\n"
        "silently."
    ))
    lines += recording.toml_table("run", {
        "seed": goldenrun.TEST_SEED,
        "n_workers": 1,
        "cg_max_iter": tier2.CG_MAX_ITER,
        "cg_max_iter_default_in_source": recording.cg_max_iter_default(dawis),
        "pipeline_max_iter": goldenrun.PIPELINE_MAX_ITER,
        "n_iterations": len(convergence),
        "n_atoms": len(atoms),
    }, comment=(
        "cg_max_iter is the conjugate-gradient solver's cap inside\n"
        "interscale_tree.CG_minimization, and here it is A SCIENCE CLAIM, NOT A\n"
        "SPEED KNOB. 200 is the published Ellien+2021 value; 7c5ba8e cut it to 3\n"
        "in 2024 and #32 reverted that drift. At 200 the solver's sigma_flux\n"
        "convergence test is live rather than vestigial, and atom fluxes sit\n"
        "25-29% higher than at 3 (#29). Tier 1 (#36) pins 3 because its job is\n"
        "catching structural breakage on a three-minute budget; this tier is the\n"
        "one that says the science is unchanged, so it is never lowered to fit a\n"
        "budget. It is still FORCED rather than inherited: the default is 200 on\n"
        "legacy-seeded but 3 on the tree the refactor branched from.\n"
        "pipeline_max_iter is the unrelated cap on the greedy outer loop."
    ))
    lines += recording.config_table(dawis, work_dir, tier2.TIER2_INFILE, tier2.CG_MAX_ITER)
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
        "The same rule as Tier 1, per #3: float tolerance on images and per-atom\n"
        "quantities, exact on atom and iteration counts, rtol=1e-5 with\n"
        "atol=1e-3*sigma on images and rtol=1e-4 per atom. The sigma is the\n"
        "robust sigma of THIS run's residual, so the absolute tolerance is a\n"
        "different number from Tier 1's while the rule producing it is the same.\n"
        "Never bit-for-bit: BLAS, OS and numpy versions move the last bits, and\n"
        "the one comparison that IS bit-identical by design is the\n"
        "parallel-consistency check (#57)."
    ))
    lines += recording.toml_table("budget", {
        "wall_s": round(wall, 1),
        "benchmark_wall_s": 1167.6,
        "benchmark_source": "benchmarks/results/2026-08-24-xps13-9360/"
                            "cg-max-iter-200/benchmark.json",
        "marker": "slow",
        "runs_on": "carve checkpoints and nightly CI (#19)",
    }, comment=(
        "What this tier costs to run, recorded so checkpoint scheduling is\n"
        "planned from a measurement rather than a guess. #37 expected it to\n"
        "exceed #5's 3539 s baseline; it does not. #5 measured the drifted Wolfe\n"
        "restoration path with animation frames written inside the loop, and #33\n"
        "measured this configuration -- published CG path, gif off -- at 1167.6 s\n"
        "on this machine. wall_s above sits higher than that for two reasons: #33\n"
        "ran seed 0 and restored 394 atoms against this run's 463, and the\n"
        "recording additionally reads every atom pickle back off disk. Budget a\n"
        "full half hour for a checkpoint run."
    ))
    lines += recording.machine_table()

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
