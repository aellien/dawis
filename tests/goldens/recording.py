"""Machinery the golden recording scripts share.

Recording a golden is the same job at every tier: prove which checkout the
measured code came from, make that checkout importable, run one pipeline
configuration, and write a ``provenance.toml`` beside the outputs. Only the
run itself and the provenance tables differ, so everything else lives here and
``record_cutout_golden.py`` and ``record_gallery_golden.py`` stay short enough
to read in one screen.

Not a test module. These scripts run by hand, outside pytest.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import types
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
TESTS = REPO / "tests"

#: Modules legacy DAWIS imports at module scope but never calls on the serial
#: path. Stubbing a *missing* one keeps the recording runnable on a clean
#: environment; whatever gets stubbed is recorded in provenance.
STUBBABLE = ("ray", "ultranest")

#: The branch every golden must be recorded from: the published Ellien+2021
#: algorithm with the RNG pinned and the post-publication drift reverted (#32).
GOLDEN_BRANCH = "legacy-seeded"

#: Knobs the legacy trees spell differently, mapped to the name 1.0 uses.
#: Provenance records the 1.0 spelling, so a re-recording cannot walk a rename
#: back. Values are untouched — these are the same knobs, renamed and not
#: redefined.
LEGACY_KNOB_NAMES = {"n_cpus": "n_workers"}


def load_tier(name: str):
    """Import a tier definition from ``tests/`` by name.

    The recording scripts run outside pytest, so ``tests/`` is not on the path
    the way collection would put it there. Importing the tier module also pins
    a headless matplotlib backend, which the legacy tree needs before anything
    imports dawis — so call this before touching ``--source``.
    """
    if str(TESTS) not in sys.path:
        sys.path.insert(0, str(TESTS))
    return __import__(name)


def stub_missing(names=STUBBABLE):
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


def git_state(path: Path) -> dict:
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


def cg_max_iter_default(dawis) -> int:
    """The CG solver's iteration cap as the recorded checkout ships it.

    Read rather than hardcoded: it is 200 on ``legacy-seeded`` and 3 on the
    tree the refactor branched from, and a recording made with
    ``--allow-any-branch`` must say which one it actually ran against.
    """
    import inspect

    parameter = inspect.signature(
        dawis.interscale_tree.CG_minimization
    ).parameters["max_iter"]
    return int(parameter.default)


def build_parser(description: str, ticket: str, goldens_dir: Path):
    """The argument parser every recording script takes."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        required=True,
        help="checkout holding the `dawis` package to record from "
        "(must be a %s checkout, per %s)" % (GOLDEN_BRANCH, ticket),
    )
    parser.add_argument(
        "--goldens",
        default=str(goldens_dir),
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
        help="record from a checkout that is not on %s" % GOLDEN_BRANCH,
    )
    return parser


def resolve_source(args, ticket: str) -> tuple[Path, dict]:
    """Check ``--source`` is a clean golden-branch checkout and describe it."""
    source = Path(args.source).resolve()
    state = git_state(source)
    if state["branch"] != GOLDEN_BRANCH and not args.allow_any_branch:
        raise SystemExit(
            "%s records goldens from %s; %s is on %s.\n"
            "Pass --allow-any-branch only if you mean it."
            % (ticket, GOLDEN_BRANCH, source, state["branch"])
        )
    if state["dirty"]:
        raise SystemExit(
            "%s has uncommitted changes; a golden's provenance commit would "
            "not describe what produced it." % source
        )
    return source, state


def import_dawis(source: Path):
    """Import the ``dawis`` package out of ``source``, and prove it came from there."""
    sys.path.insert(0, str(source))
    import dawis

    package_dir = Path(dawis.__file__).parent
    if package_dir.parent != source:
        raise SystemExit(
            "imported dawis from %s, not from --source %s" % (package_dir, source)
        )
    return dawis


def work_dir_for(args, prefix: str) -> Path:
    """The scratch directory the run writes into."""
    import tempfile

    path = (
        Path(args.work_dir)
        if args.work_dir
        else Path(tempfile.mkdtemp(prefix=prefix))
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def author_of(args) -> str:
    """Who recorded this golden: ``--author``, else the local git identity."""
    return (
        args.author
        or subprocess.run(
            ["git", "config", "user.name"], capture_output=True, text=True
        ).stdout.strip()
        or "unknown"
    )


def toml_value(value) -> str:
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
    # `--reason` and `--author` are free operator text. JSON's string escaping
    # is a subset of TOML's basic string escaping, so this is valid TOML for
    # backslashes, newlines and control characters as well as for quotes —
    # where hand-escaping only the quote would emit a file `tomllib` refuses
    # to load, and the refusal would surface a recording run too late.
    #
    # `ensure_ascii=False`, because TOML files are UTF-8 by definition and the
    # author of every golden here has a diaeresis in their name. Escaping it
    # loads back identically and reads back as `Ama\u00ebl`, which is a
    # provenance file that has stopped saying who recorded it.
    return json.dumps(str(value), ensure_ascii=False)


def toml_table(name: str, mapping: dict, comment: str | None = None) -> list[str]:
    """One provenance table, as lines."""
    lines = []
    if comment:
        lines.extend(("# " + line).rstrip()
                     for line in comment.strip().splitlines())
    lines.append("[%s]" % name)
    for key, value in mapping.items():
        lines.append("%s = %s" % (key, toml_value(value)))
    lines.append("")
    return lines


def source_table(state: dict, stubbed) -> list[str]:
    """The ``[source]`` table, identical at every tier."""
    return toml_table("source", {
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


def golden_table(tier: int, author: str, reason: str, ticket_url: str,
                 wall: float) -> list[str]:
    """The ``[golden]`` table, identical at every tier."""
    return toml_table("golden", {
        "tier": tier,
        "recorded_by": author,
        "recorded_on": date.today().isoformat(),
        "reason": reason,
        "ticket": ticket_url,
        "wall_s": round(wall, 1),
    })


def machine_table(n_threads: int) -> list[str]:
    """The ``[machine]`` table, identical at every tier.

    ``n_threads`` is what the run *asked* the maths libraries for; it is
    recorded under ``[run]`` beside the seed, with the other knobs. What lands
    here is the *observation* — the effective thread count the loaded library
    reports under that request, and which library it turned out to be. The two
    are recorded separately because they can disagree, and a knob that
    silently failed is the bug #61 exists to make visible.
    """
    import numpy as np

    import goldenrun

    effective, vendor = goldenrun.observed_threads_at(n_threads)

    return toml_table("machine", {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "blas_vendor": vendor or "none",
        "effective_n_threads": -1 if effective is None else effective,
    }, comment=(
        "Recorded for diagnosis only, EXCEPT the two thread fields. The\n"
        "goldens are expected to reproduce across machines within the\n"
        "tolerances above.\n"
        "\n"
        "effective_n_threads is what the loaded BLAS reported it would\n"
        "actually use when this run asked it for [run].n_threads, and\n"
        "blas_vendor is which library that was. The request and the\n"
        "observation are separate facts: until #61 nothing in DAWIS chose a\n"
        "thread count and nothing recorded one, so a run compared at one\n"
        "thread against a golden recorded at four differed in its last\n"
        "digits for a reason no file said. -1 means no threaded library was\n"
        "loaded at all, which is a real answer and not a failure."
    ))


def config_table(dawis, work_dir: Path, infile: str,
                 cg_max_iter: int) -> list[str]:
    """The ``[config]`` table: every pipeline parameter, defaults resolved.

    Resolved out of whichever entry point the recorded tree has. The 1.0 tree
    takes a ``Config`` (#54), whose fields *are* the resolved set; ``legacy``
    and ``legacy-seeded`` take keywords, so there the defaults are read off
    the signature. Either way provenance records every knob and not only the
    ones the tier names.
    """
    import dataclasses

    import goldenrun

    if goldenrun.is_array_in(dawis):
        config = dataclasses.asdict(goldenrun.golden_config(dawis, cg_max_iter))
    else:
        config = goldenrun.effective_config(
            dawis.synthesis_by_analysis,
            goldenrun.golden_kwargs(dawis, work_dir, infile),
        )
        # Paths identify this machine, not the run's behaviour.
        for name in ("indir", "outdir", "infile"):
            config.pop(name, None)
        # The legacy trees spell the worker count `n_cpus`; #60 renamed it.
        # It is the same knob holding the same value, so it is recorded under
        # the name the repository uses -- otherwise a re-recording would
        # silently walk that rename back, and two goldens would disagree on
        # what to call one number.
        for old_name, new_name in LEGACY_KNOB_NAMES.items():
            if old_name in config:
                config[new_name] = config.pop(old_name)

    return toml_table("config", config, comment=(
        "Every pipeline parameter with the value this run actually used,\n"
        "defaults resolved."
    ))
