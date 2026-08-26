"""Tier 0 — the three dependency rules of #4, made executable (#58).

The target architecture states three edges the import graph may not have.
Each was written down in prose, and each was pinned separately — one in
``test_viz.py``, one in ``test_parallel.py``, one in
``test_pipelines_layout.py`` — beside the code that moved. This module is the
one walk that owns all three, so a rule cannot be true in the docstring of a
package nobody reads.

**Nothing outside ``dawis.viz`` imports matplotlib.** Drawing is the optional
``[viz]`` extra. A solver that imports it is a solver that will not start on a
headless install.

**Nothing outside ``dawis.parallel`` imports ray.** ray is the optional
``[ray]`` extra and the seam owns every line of it, so a second backend can
replace the first without touching science code.

**No block imports from ``dawis.pipelines``.** The dependency edge runs one
way: a pipeline may import any block, and a block that imported the pipeline
back could not be run without the whole solver behind it.

matplotlib crept into the core once already, which is what the whole viz carve
(#46) was about. This walk fails the moment someone reaches for it in a
solver, and it costs a few tens of milliseconds to keep.

The rules are read off the *source*, never off imported modules. A module that
imports matplotlib inside a function, or only on some platform, still breaks
the rule, and a run-time check would miss it.
"""

from __future__ import annotations

import ast
import importlib.util
import time
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# ``find_spec`` rather than ``import dawis``: importing would execute every
# module the rules are about, so a violation bad enough to break the import —
# ``import ray`` at the top of a solver, on a machine without the [ray] extra
# — would collapse this file rather than be reported by it. ``find_spec``
# reads the same install and executes nothing.
#
# This buys the module its own independence, not the whole session's: the
# suite's ``conftest`` imports dawis in a session-scoped autouse fixture, so
# in a full pytest run such a violation still surfaces first as an error at
# setup. Running this file alone, or importing it outside pytest, then names
# the rule.
_SPEC = importlib.util.find_spec('dawis')

#: The package directory, found without importing the package.
PACKAGE_ROOT = Path(list(_SPEC.submodule_search_locations)[0]).resolve()

#: The name the source tree is rooted at, so relative imports can be resolved
#: to the absolute names the rules are written in.
PACKAGE_NAME = PACKAGE_ROOT.name


def module_names(path: Path) -> tuple[str, str]:
    """The dotted name of a source file, and of the package containing it.

    ``dawis/detection/noise.py`` is ``dawis.detection.noise`` inside
    ``dawis.detection``; ``dawis/detection/__init__.py`` *is*
    ``dawis.detection``, and is contained by it too — which is what ``from
    . import x`` means in an ``__init__``.
    """
    parts = [PACKAGE_NAME, *path.relative_to(PACKAGE_ROOT).with_suffix('').parts]
    if parts[-1] == '__init__':
        parts.pop()
        return '.'.join(parts), '.'.join(parts)
    return '.'.join(parts), '.'.join(parts[:-1])


@lru_cache(maxsize=None)
def imported_modules(path: Path) -> frozenset[str]:
    """Every module name a source file imports, as an absolute dotted name.

    Three forms all count as importing ``dawis.pipelines``, and all three are
    returned here: ``import dawis.pipelines``, ``from dawis.pipelines import
    run``, and ``from dawis import pipelines``. Relative imports are resolved
    against the containing package, because ``from ..pipelines import run``
    breaks the rule exactly as loudly as the absolute spelling does.
    """
    _, package = module_names(path)
    with warnings.catch_warnings():
        # Legacy docstrings in the package carry stray backslashes. Parsing
        # them is not the subject of this test.
        warnings.simplefilter('ignore', SyntaxWarning)
        tree = ast.parse(path.read_text(), filename=str(path))

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = node.module or ''
            else:
                # ``level`` counts dots: one means the containing package,
                # each further dot climbs one level out of it.
                climb = package.split('.')[: len(package.split('.')) - node.level + 1]
                base = '.'.join([*climb, node.module] if node.module else climb)
            if not base:
                continue
            names.add(base)
            # ``from dawis import pipelines`` imports ``dawis.pipelines``,
            # even though the module named in the statement is ``dawis``.
            names.update('%s.%s' % (base, alias.name) for alias in node.names)
    return frozenset(names)


def package_modules() -> list[Path]:
    """Every source file in the installed package.

    The *installed* one, which under the editable install contributors and CI
    use is the working tree. A non-editable install would put site-packages
    here, and the rules would then be checked against what was installed
    rather than against what is being edited.
    """
    return sorted(PACKAGE_ROOT.rglob('*.py'))


def imports(names: frozenset[str] | set[str], module: str) -> bool:
    """Whether ``module`` or anything under it appears in ``names``."""
    return any(name == module or name.startswith(module + '.') for name in names)


@dataclass(frozen=True)
class Rule:
    """One forbidden edge: what may not be imported, and by whom.

    ``exempt`` holds the paths the rule does not reach, relative to the
    package root — the module that owns the dependency, and nothing else.
    """

    forbidden: str
    exempt: frozenset[str]
    statement: str

    def owns(self, path: Path) -> bool:
        """Whether the rule exempts ``path``.

        An entry ending in ``/`` exempts a directory and everything under it;
        any other entry is one file, matched whole. Matching on a bare string
        prefix instead would let ``parallel.py`` exempt ``parallel.pyi`` too.
        """
        name = path.relative_to(PACKAGE_ROOT).as_posix()
        return any(
            name.startswith(entry) if entry.endswith('/') else name == entry
            for entry in self.exempt
        )

    def offenders(self) -> list[str]:
        """Every module that breaks this rule, by its path in the package."""
        return [
            path.relative_to(PACKAGE_ROOT).as_posix()
            for path in package_modules()
            if not self.owns(path)
            and imports(imported_modules(path), self.forbidden)
        ]


#: The three rules, as data. Adding a fourth is one entry, not a fourth walk.
RULES = {
    'matplotlib-belongs-to-viz': Rule(
        forbidden='matplotlib',
        exempt=frozenset({'viz/'}),
        statement=(
            'matplotlib is the optional [viz] extra and belongs only to '
            'dawis.viz, so that dawis imports on a headless install'
        ),
    ),
    'ray-belongs-to-parallel': Rule(
        forbidden='ray',
        exempt=frozenset({'parallel.py'}),
        statement=(
            'ray is the optional [ray] extra and belongs only to '
            'dawis.parallel, which owns every line of it behind pmap'
        ),
    ),
    'no-block-imports-a-pipeline': Rule(
        forbidden='dawis.pipelines',
        # The package __init__ re-exports the public entry point, which is
        # the package speaking for itself rather than a block reaching for a
        # pipeline; pipelines may of course import each other.
        exempt=frozenset({'__init__.py', 'pipelines/'}),
        statement=(
            'the dependency edge runs one way only: a pipeline may import any '
            'block, and no block may import a pipeline'
        ),
    ),
}


def test_the_import_graph_breaks_no_dependency_rule():
    """All three rules, over every module, in one walk.

    One test rather than three, because the walk is the expensive half and
    the failure has to name both halves anyway: which module, and which rule.
    """
    offenders = {name: rule.offenders() for name, rule in RULES.items()}
    broken = {name: found for name, found in offenders.items() if found}

    assert broken == {}, '\n'.join(
        '%s: %s -- %s' % (name, ', '.join(offenders), RULES[name].statement)
        for name, offenders in sorted(broken.items())
    )


def test_the_walk_reaches_every_module_in_the_package():
    """The rules are only worth as much as the set of files they run over.

    A walk that silently stopped at the top level would pass all three rules
    forever. This pins that it descends into the blocks, and that parsing
    every file works — an unparseable module would otherwise raise from
    inside the rule test and read as a rule failure.
    """
    walked = {path.relative_to(PACKAGE_ROOT).as_posix() for path in package_modules()}

    assert '__init__.py' in walked
    assert 'parallel.py' in walked
    for block in ('detection', 'io', 'pipelines', 'restoration', 'selection',
                  'trees', 'viz', 'wavelets'):
        assert any(name.startswith(block + '/') for name in walked), block

    for path in package_modules():
        assert isinstance(imported_modules(path), frozenset)


def test_a_rule_matches_exactly_what_it_names():
    """A rule that matches too little, or too much, passes for a wrong reason.

    Both halves are fed their inputs directly, because the package cannot be
    given a bad import to find. Matching too little is a typo in ``forbidden``
    — ``matplotlab`` — which is a green test forever. Matching too much is a
    string prefix: ``ray`` flagging ``rays``, or an exemption of ``io`` also
    exempting ``iolib.py``, which quietly switches the rule off for files it
    should cover.
    """
    for name, rule in RULES.items():
        assert imports({rule.forbidden}, rule.forbidden), name
        assert imports({rule.forbidden + '.pyplot'}, rule.forbidden), name
        assert not imports({rule.forbidden + 'x'}, rule.forbidden), name

    directory = Rule(forbidden='x', exempt=frozenset({'viz/'}), statement='')
    assert directory.owns(PACKAGE_ROOT / 'viz' / 'cubes.py')
    assert not directory.owns(PACKAGE_ROOT / 'cubes.py')

    one_file = Rule(forbidden='x', exempt=frozenset({'parallel.py'}), statement='')
    assert one_file.owns(PACKAGE_ROOT / 'parallel.py')
    assert not one_file.owns(PACKAGE_ROOT / 'parallel.pyi')


def test_the_rules_name_dependencies_the_package_really_has():
    """Each forbidden import exists somewhere, in the module that owns it.

    A rule over a library nobody uses is a rule that cannot break. When the
    viz block stops importing matplotlib, or the parallel seam stops owning
    ray, this fails and the rule should be reconsidered rather than kept.
    """
    for name, rule in RULES.items():
        owners = [
            path for path in package_modules()
            if rule.owns(path) and imports(imported_modules(path), rule.forbidden)
        ]
        assert owners != [], (
            '%s forbids %s, which no module the rule exempts imports'
            % (name, rule.forbidden)
        )


def test_the_walk_runs_well_under_a_second():
    """The budget that makes the whole walk worth keeping in every run.

    Parsing the package takes a few tens of milliseconds, so the budget below
    is more than an order of magnitude above what the walk costs. It is not a
    benchmark: it fails when someone makes the walk import the package, shell
    out, or read something off the network, which is the only way it grows
    that far.

    The parse cache is cleared before each attempt, so what is timed is the
    cold walk a fresh test session pays for and not three cache hits. The best
    of three attempts is taken: a loaded machine can lose a scheduling slice
    in any one of them, and this test is not there to measure the machine.
    """
    attempts = []
    for _ in range(3):
        imported_modules.cache_clear()
        started = time.perf_counter()
        for rule in RULES.values():
            rule.offenders()
        attempts.append(time.perf_counter() - started)

    assert min(attempts) < 1.0, (
        'the import-graph walk took %.2f s' % min(attempts)
    )
