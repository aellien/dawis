#!/usr/bin/env python
"""Hand-run wall-time benchmark for DAWIS (ticket #33).

This is a hand-run script. It is not a pytest tier and it is not run in CI.
It exists so that every speed number measured on the 1.0 tree stays connected
to the published 3539 s baseline recorded on #5, even though the shipping
configuration has moved (the CG solver's ``max_iter`` returns to the published
200 -- RFC #29).

One invocation measures one configuration: a full synthesis-by-analysis
run on the 900x900 reference image, and writes ``benchmark.json`` into the
output directory with wall time overall, wall time per pipeline phase (the
restoration stage in particular), peak RSS, atom and iteration counts, and the
full configuration plus machine identification needed to reproduce it.

Two ``max_iter`` knobs exist and they are unrelated. ``--pipeline-max-iter``
caps the number of greedy outer iterations (``Config.max_iter`` on the 1.0
tree). ``--cg-max-iter`` is the conjugate-gradient solver's iteration cap
inside ``InterscaleTree.CG_minimization``, which is the knob #29 restores
from 3 to 200 and the one this benchmark sweeps.

Usage
-----
Run one configuration::

    python benchmarks/run_benchmark.py \\
        --source ~/dawis-legacy-seeded \\
        --outdir ~/bench/legacy-seeded-cg200 \\
        --cg-max-iter 200 --label "legacy-seeded, CG max_iter=200"

Aggregate finished runs into a markdown table::

    python benchmarks/run_benchmark.py --report ~/bench
"""
import argparse
import dataclasses
import glob
import inspect
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
import types
from datetime import datetime

try:
    import psutil
except ImportError:  # pragma: no cover - a clear message beats a silent 0.0 MB
    raise SystemExit(
        "run_benchmark.py needs psutil for the peak-RSS number.\n"
        "Install it with:  pip install 'dawis[bench]'   (or: pip install psutil)"
    )

# --------------------------------------------------------------------------
# Reference image and parameter set
# --------------------------------------------------------------------------

REFERENCE_IMAGE = "tidal_group_f814w_sci.fits"

# test.py's parameter set, verbatim -- the same one #5 profiled, so the numbers
# stay comparable. Only the knobs the benchmark sweeps are left out here.
REFERENCE_PARAMS = dict(
    n_levels=9,
    tau=0.1,
    gamma=0.5,
    ceps=1e-4,
    min_span=2,
    max_span=3,
    lvl_sep_big=5,
    starting_level=2,
)

# Modules that legacy DAWIS imports at module scope but never calls on the
# serial path. Stubbing a *missing* one keeps the benchmark runnable on a
# clean environment without installing a heavy dependency that no measured
# code path touches. Whatever gets stubbed is recorded in benchmark.json.
STUBBABLE = ("ray", "ultranest")


# --------------------------------------------------------------------------
# Log parsing -- phase timings from DAWIS's own log markers
# --------------------------------------------------------------------------

TS = r"\[ (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+) \]"
MARKS = [
    ("iter_start", re.compile(TS + r" Level = (\d+) Iteration = (\d+)")),
    ("wavelet", re.compile(TS + r" Start wavelet transform and detection")),
    ("trees", re.compile(TS + r" Start interscale trees")),
    ("restore", re.compile(TS + r" Start object restoration")),
    ("atoms", re.compile(TS + r" Add atoms to restored images")),
    ("dump", re.compile(TS + r" Dumping data in")),
    ("iter_end", re.compile(TS + r"End of iteration \(time: (\S+)\)")),
]
NOBJ = re.compile(r"Number Objects = (\d+)")

PHASES = [
    ("inpaint+resume", "t0", "wavelet"),
    ("wavelet+detect", "wavelet", "trees"),
    ("trees", "trees", "restore"),
    ("restore", "restore", "atoms"),
    ("atoms+converge", "atoms", "dump"),
    # Named for the legacy loop, which drew the animation frames inside this
    # phase. On the 1.0 tree it is the dump alone: #46 moved the drawing out.
    ("dump+gif", "dump", "end"),
]
PHASES_NODUMP = [
    ("inpaint+resume", "t0", "wavelet"),
    ("wavelet+detect", "wavelet", "trees"),
    ("trees", "trees", "restore"),
    ("restore", "restore", "atoms"),
    ("atoms+converge+write", "atoms", "end"),
]


def parse_log(path):
    """Return the ordered list of (mark_name, timestamp, extra) in a run log."""
    events = []
    with open(path, errors="replace") as fh:
        for line in fh:
            for name, rx in MARKS:
                m = rx.search(line)
                if m:
                    t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f")
                    events.append((name, t, m.groups()[1:]))
                    break
            else:
                m = NOBJ.search(line)
                if m:
                    events.append(("nobj", None, (m.group(1),)))
    return events


def iterations(events):
    """Yield one dict per greedy iteration, carrying its phase timestamps."""
    cur = None
    for name, t, extra in events:
        if name == "iter_start":
            if cur:
                yield cur
            cur = dict(level=int(extra[0]), it=int(extra[1]), t0=t, marks={}, nobj=None)
        elif cur is None:
            continue
        elif name == "nobj":
            cur["nobj"] = int(extra[0])
        elif name == "iter_end":
            cur["marks"]["end"] = t
            yield cur
            cur = None
        else:
            cur["marks"][name] = t
    if cur:
        yield cur


def phase_totals(log_path):
    """Sum each pipeline phase over every complete iteration in a run log."""
    its = list(iterations(parse_log(log_path)))
    has_dump = any("dump" in i["marks"] for i in its)
    phases = PHASES if has_dump else PHASES_NODUMP
    totals = {name: 0.0 for name, _, _ in phases}
    n_complete = 0
    n_atoms = 0
    for i in its:
        marks = dict(i["marks"])
        marks["t0"] = i["t0"]
        if "end" not in marks:
            continue
        n_complete += 1
        if i["nobj"] is not None:
            n_atoms += i["nobj"]
        for name, a, b in phases:
            ta, tb = marks.get(a), marks.get(b)
            if ta and tb:
                totals[name] += (tb - ta).total_seconds()
    return dict(
        phases_s=totals,
        loop_s=sum(totals.values()),
        n_iterations=n_complete,
        n_atoms=n_atoms,
    )


# --------------------------------------------------------------------------
# Environment capture
# --------------------------------------------------------------------------


def git_state(path):
    """Identify the checkout a package was imported from."""

    def git(*args):
        try:
            return subprocess.check_output(
                ("git", "-C", path) + args, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except (subprocess.CalledProcessError, OSError):
            return None

    commit = git("rev-parse", "HEAD")
    if commit is None:
        return dict(commit=None, branch=None, dirty=None)
    return dict(
        commit=commit,
        branch=git("rev-parse", "--abbrev-ref", "HEAD"),
        describe=git("describe", "--tags", "--always"),
        dirty=bool(git("status", "--porcelain")),
    )


def cpu_model():
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def total_ram_gb():
    return round(psutil.virtual_memory().total / 2**30, 1)


# pyproject.toml's hard dependencies, then the optional ones a run can touch.
VERSIONED = (
    "numpy",
    "scipy",
    "astropy",
    "h5py",
    "photutils",
    "skimage",
    "matplotlib",
    "psutil",
    "ray",
)


def package_versions():
    out = {}
    for name in VERSIONED:
        try:
            out[name] = __import__(name).__version__
        except Exception:
            out[name] = None
    return out


def machine_info():
    return dict(
        hostname=platform.node(),
        cpu_model=cpu_model(),
        n_cpus_machine=os.cpu_count(),
        ram_total_gb=total_ram_gb(),
        platform=platform.platform(),
        python=platform.python_version(),
        packages=package_versions(),
    )


# --------------------------------------------------------------------------
# Memory sampling
# --------------------------------------------------------------------------


def sample_memory(stop_event, out_path, interval=0.5):
    """Sample RSS of this process and its children until stop_event is set."""
    proc = psutil.Process()
    t0 = time.time()
    with open(out_path, "w") as fh:
        fh.write("t_s,rss_total_mb,rss_driver_mb,n_procs\n")
        while not stop_event.is_set():
            try:
                driver = proc.memory_info().rss
                total = driver
                kids = proc.children(recursive=True)
                for k in kids:
                    try:
                        total += k.memory_info().rss
                    except psutil.Error:
                        pass
                fh.write(
                    "%.2f,%.1f,%.1f,%d\n"
                    % (time.time() - t0, total / 2**20, driver / 2**20, 1 + len(kids))
                )
                fh.flush()
            except psutil.Error:
                pass
            stop_event.wait(interval)


def peak_memory(path):
    peak_total = peak_driver = 0.0
    max_procs = 0
    try:
        with open(path) as fh:
            next(fh, None)
            for line in fh:
                parts = line.strip().split(",")
                if len(parts) != 4:
                    continue
                peak_total = max(peak_total, float(parts[1]))
                peak_driver = max(peak_driver, float(parts[2]))
                max_procs = max(max_procs, int(parts[3]))
    except OSError:
        pass
    return dict(
        peak_rss_total_mb=round(peak_total, 1),
        peak_rss_driver_mb=round(peak_driver, 1),
        max_procs=max_procs,
    )


def dir_size_mb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return round(total / 2**20, 1)


# --------------------------------------------------------------------------
# Import and instrumentation of the measured package
# --------------------------------------------------------------------------


def stub_missing(names):
    """Install do-nothing stand-ins for missing module-scope-only imports."""
    stubbed = []
    for name in names:
        try:
            __import__(name)
            continue
        except ImportError:
            pass
        mod = types.ModuleType(name)
        if name == "ray":
            # Only `@ray.remote` runs at import time; nothing else is reached
            # on the serial path this benchmark measures.
            mod.remote = lambda fn: fn
            mod.is_initialized = lambda: False
        sys.modules[name] = mod
        stubbed.append(name)
    return stubbed


def import_dawis(source):
    if source:
        sys.path.insert(0, os.path.abspath(source))
    import dawis

    return dawis


def effective_config(entry, passed):
    """Every pipeline parameter with the value this run actually used.

    The benchmark passes only the reference parameter set, so most knobs take
    the entry point's defaults. A results file that recorded only what was
    passed would omit live knobs -- `deconv` gates a Richardson-Lucy call
    inside the CG loop, `conditions` picks the boundary treatment -- so the
    defaults are resolved here and recorded with the rest.
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


def force_cg_max_iter(max_iter):
    """Pin the CG solver's iteration cap and count how often it is called.

    The benchmark sweeps that cap, so it is forced here rather than edited
    into the tree under measurement. Patching the bound method still beats
    passing the argument, and for the same reason it always did: the same
    forced value then applies to whichever branch is running, including
    ``legacy-seeded``, whose ``restore_object`` passes nothing at all. Since
    #53 the 1.0 tree does pass it — ``Config.cg_max_iter`` reaches the solver
    through ``restore_object`` — and the wrapper overrides that too, because it
    sets the keyword after the caller has. Returns (original_default,
    call_counter).
    """
    from dawis.trees import InterscaleTree

    original = InterscaleTree.CG_minimization
    default = inspect.signature(original).parameters["max_iter"].default
    counter = {"calls": 0}

    def patched(self, *args, **kwargs):
        counter["calls"] += 1
        kwargs["max_iter"] = max_iter
        return original(self, *args, **kwargs)

    # A wrapper that forces a value equal to the source default is a no-op; the
    # results file needs to be able to say which of the two runs was patched.

    InterscaleTree.CG_minimization = patched
    return default, counter


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------


def run(args):
    os.makedirs(args.outdir, exist_ok=True)

    stubbed = stub_missing(STUBBABLE) if args.stub_missing else []
    dawis = import_dawis(args.source)
    package_dir = os.path.dirname(dawis.__file__)

    indir = args.indir or os.path.join(package_dir, "gallery")
    infile = args.infile
    image_path = os.path.join(indir, infile)
    if not os.path.exists(image_path):
        raise SystemExit("reference image not found: %s" % image_path)

    from astropy.io import fits

    with fits.open(image_path) as hdu:
        image_shape = list(hdu[0].data.shape)

    cg_default, cg_counter = force_cg_max_iter(args.cg_max_iter)

    knobs = dict(
        REFERENCE_PARAMS,
        n_workers=args.n_workers,
        max_iter=args.pipeline_max_iter,
        # Pinned off rather than dropped when #66 deleted the flag. The 1.0
        # tree has no such knob and filters this out against `Config`, but
        # `legacy` and `legacy-seeded` still take it and default it to True --
        # so dropping it here would silently put the per-iteration region
        # pickle and three FITS writes back into exactly the runs this script
        # times against the 1.0 tree. The measurement is CPU work, on all
        # three trees or on none.
        data_dump=False,
        resume=False,
    )

    # Two entry points are measured by this one script. The 1.0 tree takes a
    # path, a `Config` and an output directory (#54); `legacy` and
    # `legacy-seeded` take fifteen keywords. Branching on the capability
    # rather than on a branch name keeps one benchmark comparable across all
    # three, which is the whole reason the script exists.
    is_array_in = hasattr(dawis, "run_fits")

    if is_array_in:
        fields = dawis.Config.__dataclass_fields__
        config_passed = {
            name: value for name, value in knobs.items() if name in fields
        }
        config_passed["seed"] = args.seed
        # Recorded, not obeyed: `force_cg_max_iter` patches the solver itself,
        # so this is what the configuration *says* the run used.
        config_passed["cg_max_iter"] = args.cg_max_iter
        config_object = dawis.Config(**config_passed)
        supports_seed = True
        supports_gif = False

        def call():
            dawis.run_fits(image_path, config_object, outdir=args.outdir)

        resolved_config = dataclasses.asdict(config_object)
        # Knobs the 1.0 tree validates and no run reads yet (#104 empties the
        # tuple). `benchmark.json` reports what the run did, so a knob
        # nothing honours has no value to report.
        for name in getattr(dawis.config, "UNWIRED_KNOBS", ()):
            resolved_config.pop(name, None)
    else:
        entry = dawis.synthesis_by_analysis
        entry_params = inspect.signature(entry).parameters
        supports_seed = "seed" in entry_params
        # `gif` is a parameter on `legacy` and `legacy-seeded`; #46 took the
        # animation off the loop and removed it from the 1.0 tree, where the
        # frames are rebuilt afterwards by `dawis.viz.rebuild_animation`.
        supports_gif = "gif" in entry_params

        # Filtered against the signature in hand rather than passed whole:
        # the entry points differ by a knob or two, and #60 renamed the
        # worker count, which the older trees still spell their own way. A
        # knob this branch has no name for falls to that branch's default.
        config_passed = {
            name: value for name, value in knobs.items()
            if name in entry_params
        }
        config_passed.update(indir=indir, infile=infile, outdir=args.outdir)

        # The worker count is the one knob that may not quietly fall back.
        # This script exists to compare timings across the three trees, and a
        # run that silently used one worker when four were asked for would be
        # compared against one that used four.
        if "n_workers" not in entry_params and args.n_workers != 1:
            raise SystemExit(
                "--n-workers: this tree's entry point has no such parameter "
                "-- it predates the rename (#60), so it can only be "
                "benchmarked at its own default worker count"
            )

        # `seed` exists on legacy-seeded, not on legacy.
        if supports_seed:
            config_passed["seed"] = args.seed
        if supports_gif:
            config_passed["gif"] = args.gif

        def call():
            entry(**config_passed)

        resolved_config = effective_config(entry, config_passed)

    if args.gif and not supports_gif:
        raise SystemExit(
            "--gif: this tree draws no frames during a run (#46); rebuild them "
            "afterwards with dawis.viz.rebuild_animation"
        )

    stop = threading.Event()
    mem_path = os.path.join(args.outdir, "memory.csv")
    sampler = threading.Thread(
        target=sample_memory, args=(stop, mem_path), daemon=True
    )
    sampler.start()

    started = datetime.now()
    t0 = time.time()
    status, err = "ok", None
    try:
        call()
    except BaseException as exc:  # noqa: BLE001 - a benchmark records anything
        status = "error"
        err = "%s: %s" % (type(exc).__name__, exc)
    finally:
        wall = time.time() - t0
        stop.set()
        sampler.join(timeout=3)

    log_path = os.path.join(args.outdir, infile[:-5] + ".log.txt")
    timings = phase_totals(log_path) if os.path.exists(log_path) else {}

    result = dict(
        label=args.label,
        status=status,
        error=err,
        started=started.isoformat(timespec="seconds"),
        wall_s=round(wall, 1),
        image=dict(path=image_path, shape=image_shape),
        cg_max_iter=args.cg_max_iter,
        cg_max_iter_default_in_source=cg_default,
        cg_max_iter_forced=args.cg_max_iter != cg_default,
        cg_calls=cg_counter["calls"],
        seed=args.seed if supports_seed else None,
        seed_supported=supports_seed,
        config=resolved_config,
        config_passed=config_passed,
        source=dict(
            package_dir=package_dir,
            git=git_state(os.path.dirname(package_dir)),
            stubbed_modules=stubbed,
        ),
        machine=machine_info(),
        outdir_mb=dir_size_mb(args.outdir),
    )
    result.update(timings)
    result.update(peak_memory(mem_path))
    if timings.get("loop_s"):
        result["restore_share"] = round(
            timings["phases_s"].get("restore", 0.0) / timings["loop_s"], 4
        )

    with open(os.path.join(args.outdir, "benchmark.json"), "w") as fh:
        json.dump(result, fh, indent=2)

    print(json.dumps(result, indent=2))
    return 0 if status == "ok" else 1


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def report(root):
    paths = sorted(glob.glob(os.path.join(root, "**", "benchmark.json"), recursive=True))
    runs = []
    for path in paths:
        with open(path) as fh:
            run_data = json.load(fh)
        run_data.setdefault("_dirname", os.path.basename(os.path.dirname(path)))
        runs.append(run_data)
    if not runs:
        raise SystemExit("no benchmark.json found under %s" % root)

    rows = [
        "| Run | CG `max_iter` | Wall (s) | Loop (s) | Restoration (s) | Restoration share | Iterations | Atoms | Peak RSS (MB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in runs:
        restore = r.get("phases_s", {}).get("restore")
        rows.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                r.get("label") or r["_dirname"],
                r.get("cg_max_iter"),
                r.get("wall_s"),
                round(r["loop_s"], 1) if r.get("loop_s") else "-",
                round(restore, 1) if restore else "-",
                "%.1f%%" % (100 * r["restore_share"]) if r.get("restore_share") else "-",
                r.get("n_iterations", "-"),
                r.get("n_atoms", "-"),
                r.get("peak_rss_total_mb", "-"),
            )
        )
    print("\n".join(rows))
    return 0


# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--report", metavar="DIR", help="aggregate finished runs and exit")
    ap.add_argument("--outdir", help="directory for this run's outputs")
    ap.add_argument(
        "--source",
        help="checkout holding the `dawis` package to measure "
        "(default: whatever is importable)",
    )
    ap.add_argument("--indir", help="directory holding the reference image")
    ap.add_argument("--infile", default=REFERENCE_IMAGE)
    ap.add_argument(
        "--cg-max-iter",
        type=int,
        help="CG solver iteration cap, forced onto InterscaleTree.CG_minimization "
        "(the knob #29 restores from 3 to 200)",
    )
    ap.add_argument(
        "--pipeline-max-iter",
        type=int,
        default=500,
        help="cap on greedy outer iterations -- NOT the CG knob (default: 500)",
    )
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="")
    ap.add_argument(
        "--gif",
        action="store_true",
        help="write animation frames -- only on a tree whose entry point still "
             "takes `gif`; #46 removed it from 1.0",
    )
    ap.add_argument(
        "--no-stub-missing",
        dest="stub_missing",
        action="store_false",
        help="fail instead of stubbing missing module-scope-only imports",
    )
    args = ap.parse_args()

    if args.report:
        return report(args.report)
    if not args.outdir or args.cg_max_iter is None:
        ap.error("--outdir and --cg-max-iter are required for a run")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
