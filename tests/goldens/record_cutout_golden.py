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
so the recording and the comparison can never drift apart.

Goldens are a ratchet, not a shrine. Re-run this only in a commit whose
message says why the golden moved.
"""

from __future__ import annotations

import argparse
import importlib.util
import platform
import shutil
import subprocess
import sys
import time
import types
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

#: Modules legacy DAWIS imports at module scope but never calls on the serial
#: path. Stubbing a *missing* one keeps the recording runnable on a clean
#: environment; whatever gets stubbed is recorded in provenance.
STUBBABLE = ("ray", "ultranest")


def load_tier1():
    """Import ``tests/tier1.py`` as a plain module, by path.

    It holds the Tier 1 run definition the comparison test also uses. This
    script runs by hand, outside pytest, so it loads the file directly rather
    than relying on pytest's collection.
    """
    spec = importlib.util.spec_from_file_location(
        "dawis_tier1", REPO / "tests" / "tier1.py"
    )
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: `dataclass` resolves a class's module out
    # of sys.modules and fails on a module that is not there yet.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def stub_missing(names):
    """Install do-nothing stand-ins for missing module-scope-only imports."""
    stubbed = []
    for name in names:
        try:
            __import__(name)
            continue
        except ImportError:
            pass
        module = types.ModuleType(name)
        if name == "ray":
            # Only `@ray.remote` runs at import time; nothing else is reached
            # on the serial path this recording uses.
            module.remote = lambda fn: fn
            module.is_initialized = lambda: False
        sys.modules[name] = module
        stubbed.append(name)
    return stubbed


def git_state(path):
    """Identify the checkout the measured package was imported from."""

    def git(*args):
        try:
            return subprocess.check_output(
                ("git", "-C", str(path)) + args, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except (subprocess.CalledProcessError, OSError):
            return None

    commit = git("rev-parse", "HEAD")
    if commit is None:
        # `git` swallows its own failures, so without this a non-repository
        # --source would report a clean checkout on branch None.
        raise SystemExit("%s is not a git checkout" % path)
    return {
        "commit": commit,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "describe": git("describe", "--tags", "--always"),
        "dirty": bool(git("status", "--porcelain")),
    }


def robust_sigma(values):
    """Half the 16-84 percentile spread — the noise scale #3's atol rests on."""
    import numpy as np

    low, high = np.percentile(values, [16, 84])
    return float((high - low) / 2.0)


def toml_value(value):
    """Render a Python scalar as TOML. Enough for what provenance holds."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if value is None:
        return '""'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    return '"%s"' % str(value).replace('"', '\\"')


def toml_table(name, mapping, comment=None):
    lines = []
    if comment:
        lines.extend("# " + line for line in comment.strip().splitlines())
    lines.append("[%s]" % name)
    for key, value in mapping.items():
        lines.append("%s = %s" % (key, toml_value(value)))
    lines.append("")
    return lines


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source",
        required=True,
        help="checkout holding the `dawis` package to record from "
        "(must be a legacy-seeded checkout, per #36)",
    )
    parser.add_argument(
        "--goldens",
        default=str(HERE / "tier1_cutout"),
        help="directory the goldens are written into",
    )
    parser.add_argument(
        "--work-dir",
        help="scratch directory for the run itself (default: a temporary one)",
    )
    parser.add_argument("--author", help="who recorded these (default: git user.name)")
    parser.add_argument(
        "--reason", required=True, help="why these goldens were (re)recorded"
    )
    parser.add_argument(
        "--allow-any-branch",
        action="store_true",
        help="record from a checkout that is not on legacy-seeded",
    )
    args = parser.parse_args()

    # Importing tier1 also pins a headless matplotlib backend, which the
    # legacy tree needs before anything imports dawis.
    tier1 = load_tier1()

    source = Path(args.source).resolve()
    state = git_state(source)
    if state["branch"] != "legacy-seeded" and not args.allow_any_branch:
        raise SystemExit(
            "#36 records Tier 1 goldens from legacy-seeded; %s is on %s.\n"
            "Pass --allow-any-branch only if you mean it." % (source, state["branch"])
        )
    if state["dirty"]:
        raise SystemExit(
            "%s has uncommitted changes; a golden's provenance commit would "
            "not describe what produced it." % source
        )

    stubbed = stub_missing(STUBBABLE)
    sys.path.insert(0, str(source))
    import dawis

    import numpy as np
    from astropy.io import fits

    package_dir = Path(dawis.__file__).parent
    if package_dir.parent != source:
        raise SystemExit(
            "imported dawis from %s, not from --source %s" % (package_dir, source)
        )

    image_path = package_dir / "gallery" / tier1.GALLERY_IMAGE
    full = fits.getdata(image_path)
    cut = np.array(tier1.cutout_of(full))

    work_dir = Path(args.work_dir) if args.work_dir else None
    if work_dir is None:
        import tempfile

        work_dir = Path(tempfile.mkdtemp(prefix="dawis-tier1-"))
    work_dir.mkdir(parents=True, exist_ok=True)

    print("recording Tier 1 goldens")
    print("  source     %s  (%s)" % (source, state["describe"]))
    print("  image      %s %s" % (image_path, list(full.shape)))
    print("  cutout     y %s  x %s" % tier1.CUTOUT_SLICE)
    print("  seed       %d" % tier1.TEST_SEED)
    print("  CG max_iter %d (pinned speed knob)" % tier1.CG_MAX_ITER)
    print("  work dir   %s" % work_dir)

    started = time.time()
    run = tier1.run_tier1(dawis, cut, work_dir)
    wall = time.time() - started

    atoms = tier1.atom_catalogue(dawis, run)
    convergence = tier1.convergence_table(run)
    residual = fits.getdata(run.residuals)
    sigma = robust_sigma(residual)

    goldens = Path(args.goldens)
    goldens.mkdir(parents=True, exist_ok=True)

    fits.PrimaryHDU(cut).writeto(goldens / "input_cutout.fits", overwrite=True)
    shutil.copyfile(run.restored, goldens / "restored.fits")
    shutil.copyfile(run.residuals, goldens / "residuals.fits")
    tier1.write_table(goldens / "atoms.csv", tier1.ATOM_COLUMNS, atoms)
    tier1.write_table(
        goldens / "convergence.csv", tier1.CONVERGENCE_COLUMNS, convergence
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

    entry_config = tier1.effective_config(
        dawis.synthesis_by_analysis, tier1.tier1_kwargs(dawis, work_dir)
    )
    # Paths identify this machine, not the run's behaviour.
    for name in ("indir", "outdir", "infile"):
        entry_config.pop(name, None)

    author = args.author or subprocess.run(
        ["git", "config", "user.name"], capture_output=True, text=True
    ).stdout.strip() or "unknown"

    lines = [
        "# Provenance of the Tier 1 cutout goldens (#36).",
        "#",
        "# These pin the published Ellien+2021 behaviour on a 256x256 patch and",
        "# gate every commit of the 1.0 carve. Regenerate them only in a commit",
        "# whose message says why the golden moved (#3).",
        "",
    ]
    lines += toml_table("golden", {
        "tier": 1,
        "recorded_by": author,
        "recorded_on": date.today().isoformat(),
        "reason": args.reason,
        "ticket": "https://github.com/aellien/dawis-dev/issues/36",
        "wall_s": round(wall, 1),
    })
    lines += toml_table("source", {
        "repo": "dawis-dev",
        "branch": state["branch"],
        "commit": state["commit"],
        "describe": state["describe"],
        "dirty": state["dirty"],
        "stubbed_modules": stubbed,
    }, comment=(
        "The branch the goldens were recorded from: the published algorithm\n"
        "with the RNG pinned and the post-publication drifts reverted (#32)."
    ))
    lines += toml_table("input", {
        "image": "dawis/gallery/%s" % tier1.GALLERY_IMAGE,
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
    lines += toml_table("run", {
        "seed": tier1.TEST_SEED,
        "n_cpus": 1,
        "cg_max_iter": tier1.CG_MAX_ITER,
        "cg_max_iter_default_in_source": 200,
        "pipeline_max_iter": tier1.PIPELINE_MAX_ITER,
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
    ))
    lines += toml_table("config", entry_config, comment=(
        "Every pipeline parameter with the value this run actually used,\n"
        "defaults resolved."
    ))
    lines += toml_table("tolerances", {
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
        "not of the raw input. Never bit-for-bit: BLAS, OS and numpy versions\n"
        "move the last bits, and the one comparison that IS bit-identical by\n"
        "design is Tier 2's parallel-consistency check (#57)."
    ))
    lines += toml_table("machine", {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }, comment=(
        "Recorded for diagnosis only. The goldens are expected to reproduce\n"
        "across machines within the tolerances above."
    ))

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
