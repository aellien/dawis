"""Tier 0 — how DAWIS talks, and what it never does to the caller's logging (#55).

Three rules, all of them previously prose and none of them executable.

**Every module that speaks does it through a module-level logger.** One
``log = logging.getLogger(__name__)`` at the top of the file, named for the
module it sits in, so a caller can silence one block without silencing the
package and a log line says which file it came from. A logger built inside a
function is the same logger, but it hides the module's voice from anything
that reads the source, and it re-does the lookup on every call.

The rule is scoped to modules that *speak*, which is narrower than #55's
"every module uses a module-level logger". A module that emits nothing needs
no logger, and half the package emits nothing — the invariant worth holding
is that no module reaches past its own logger, which is what the third rule
below enforces on every file whether it speaks or not.

**Nothing prints.** ``print`` writes to stdout unconditionally, cannot be
filtered, cannot be redirected to the run's own file, and cannot be turned off
by a caller who is running DAWIS inside something larger.

**The library never configures the root logger.** Configuration belongs to
whoever wrote ``main``, and a library that calls
:func:`logging.basicConfig` steals it. The one deliberate exception is the
top-level pipeline entry, tested below: it attaches its handlers to the
``dawis`` logger, and only when the caller has configured nothing at all.
"""

from __future__ import annotations

import ast
import logging
import warnings
from contextlib import contextmanager
from pathlib import Path

import pytest

import dawis
from dawis.logs import LOG_EXTENSION, PACKAGE_LOGGER, run_logging

PACKAGE_ROOT = Path(dawis.__file__).resolve().parent


def package_modules() -> list[Path]:
    """Every source file in the installed package."""
    return sorted(PACKAGE_ROOT.rglob('*.py'))


def parse(path: Path) -> ast.Module:
    """Parse one source file of the package."""
    with warnings.catch_warnings():
        # Legacy docstrings in the package carry stray backslashes. Parsing
        # them is not the subject of this test.
        warnings.simplefilter('ignore', SyntaxWarning)
        warnings.simplefilter('ignore', DeprecationWarning)
        return ast.parse(path.read_text())


def attribute_calls(tree: ast.Module, module: str) -> list[ast.Call]:
    """Every ``module.something(...)`` call in a parsed source file."""
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == module
    ]


#: What a module-level logger is called, everywhere in the package.
LOGGER_NAME = 'log'

#: The one module exempt from the rule below, named by its path rather than
#: its basename so a future ``logs.py`` elsewhere in the package is not
#: silently exempt too. :mod:`dawis.logs` owns the *package* logger — the one
#: every module's logger hangs off — and reaches for it by name inside the
#: run's context manager. It emits nothing itself.
LOGGER_OWNER = Path('logs.py')

#: The methods that emit a record. A call to one of these on the ``logging``
#: module itself goes to the root logger and configures it on the way.
EMITTERS = {
    'debug', 'info', 'warning', 'warn', 'error', 'exception', 'critical', 'log',
}


# ---------------------------------------------------------------------------
# The three source rules
# ---------------------------------------------------------------------------


def test_no_module_prints():
    """``print`` is not how a library speaks."""
    offenders = []
    for path in package_modules():
        for node in ast.walk(parse(path)):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == 'print'):
                offenders.append(
                    '%s:%d' % (path.relative_to(PACKAGE_ROOT), node.lineno))

    assert offenders == [], (
        'these lines print instead of logging, so a caller cannot filter, '
        'redirect or silence them: %s' % ', '.join(offenders))


def test_every_logger_is_built_once_at_module_level():
    """One ``log = logging.getLogger(__name__)``, at the top of the file."""
    for path in package_modules():
        if path.relative_to(PACKAGE_ROOT) == LOGGER_OWNER:
            continue
        tree = parse(path)

        module_level = [
            node for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == LOGGER_NAME
        ]

        lookups = [
            node for node in attribute_calls(tree, 'logging')
            if node.func.attr == 'getLogger'
        ]

        where = path.relative_to(PACKAGE_ROOT)
        assert len(lookups) <= 1, (
            '%s builds %d loggers; a module has one voice, built once at '
            'module level' % (where, len(lookups)))

        if not lookups:
            assert not module_level, '%s assigns `log` without building one' % where
            continue

        assert len(module_level) == 1, (
            '%s builds a logger somewhere other than module level, where '
            'neither a reader nor a caller can see it' % where)

        (lookup,) = lookups
        assert [a.id for a in lookup.args if isinstance(a, ast.Name)] == ['__name__'], (
            '%s names its logger something other than __name__, so its log '
            'lines do not say where they came from' % where)


def test_the_library_never_configures_the_root_logger():
    """No ``basicConfig``, and no emitting through the ``logging`` module."""
    offenders = []
    for path in package_modules():
        for node in attribute_calls(parse(path), 'logging'):
            if node.func.attr == 'basicConfig' or node.func.attr in EMITTERS:
                offenders.append('%s:%d logging.%s' % (
                    path.relative_to(PACKAGE_ROOT), node.lineno, node.func.attr))

    assert offenders == [], (
        'these lines reach the root logger, which belongs to whoever wrote '
        'main and not to DAWIS: %s' % ', '.join(offenders))


# ---------------------------------------------------------------------------
# The one exception: the pipeline entry's own handlers
# ---------------------------------------------------------------------------


@contextmanager
def as_a_fresh_interpreter():
    """Run the block with logging as an untouched interpreter has it.

    Nothing on the ``dawis`` logger, nothing on the root logger, both at their
    default levels. Every test below turns on the part it is about and asserts
    against that baseline.

    It is a context manager entered inside the test body rather than a
    fixture, and it has to be: pytest installs its own capture handlers around
    the *call* phase, which is after fixture setup, so a fixture that cleared
    them would be undone before the first line of the test ran. Turning the
    ``dawis`` logger's ``propagate`` off instead does not help — pytest 9
    attaches to non-propagating loggers directly.
    """
    root = logging.getLogger()
    package = logging.getLogger(PACKAGE_LOGGER)
    saved_root = (list(root.handlers), root.level)
    saved_package = (list(package.handlers), package.level, package.propagate)

    root.handlers = []
    root.setLevel(logging.WARNING)
    package.handlers = []
    package.setLevel(logging.NOTSET)
    package.propagate = True

    try:
        yield package
    finally:
        root.handlers, root.level = saved_root
        package.handlers, package.level, package.propagate = saved_package


def test_the_run_keeps_its_log_file(tmp_path):
    """The run directory's log file is what it always was."""
    outpath = str(tmp_path / 'ngc1275')

    with as_a_fresh_interpreter() as package:
        with run_logging(outpath):
            package.info('the start banner')

    assert 'the start banner' in Path(outpath + LOG_EXTENSION).read_text()


def test_a_caller_who_configured_nothing_still_sees_the_run(tmp_path, capsys):
    """Default output is visible with no setup at all, notebooks included."""
    with as_a_fresh_interpreter() as package:
        with run_logging(str(tmp_path / 'run')):
            package.info('Level = 1 Iteration = 1')

    assert 'Level = 1 Iteration = 1' in capsys.readouterr().out


def test_a_caller_who_configured_logging_is_left_alone(tmp_path, capsys):
    """A caller's own handler receives the run, and nothing is doubled."""
    records = []

    with as_a_fresh_interpreter() as package:
        root = logging.getLogger()
        handler = logging.Handler()
        handler.emit = records.append
        root.addHandler(handler)
        root.setLevel(logging.INFO)

        with run_logging(str(tmp_path / 'run')):
            package.info('one line')

    assert [record.getMessage() for record in records] == ['one line']
    assert 'one line' not in capsys.readouterr().out


def test_a_handler_on_the_dawis_logger_counts_as_configured(tmp_path, capsys):
    """Configuring the package's own logger is configuring logging."""
    with as_a_fresh_interpreter() as package:
        package.addHandler(logging.NullHandler())
        package.setLevel(logging.INFO)

        with run_logging(str(tmp_path / 'run')):
            package.info('one line')

    assert 'one line' not in capsys.readouterr().out


def test_the_root_logger_is_never_touched(tmp_path):
    """Not its handlers, not its level, whatever the caller left there."""
    with as_a_fresh_interpreter():
        root = logging.getLogger()
        before = (list(root.handlers), root.level)

        with run_logging(str(tmp_path / 'run')):
            assert (list(root.handlers), root.level) == before

        assert (list(root.handlers), root.level) == before


def test_a_caller_who_set_a_level_keeps_it(tmp_path):
    """An explicit level on the ``dawis`` logger survives the run."""
    with as_a_fresh_interpreter() as package:
        package.addHandler(logging.NullHandler())
        package.setLevel(logging.WARNING)

        with run_logging(str(tmp_path / 'run')) as logger:
            assert logger.level == logging.WARNING

        assert package.level == logging.WARNING


def test_the_handlers_come_back_off(tmp_path):
    """A finished run leaves the logger exactly as it found it.

    A campaign is many runs in one process. Handlers that accumulated would
    write the second run into the first run's log file, and print every line
    of the tenth run ten times.
    """
    with as_a_fresh_interpreter() as package:
        for index in range(3):
            with run_logging(str(tmp_path / ('run%d' % index))):
                pass
            assert package.handlers == []
            assert package.level == logging.NOTSET

    assert (tmp_path / ('run0' + LOG_EXTENSION)).is_file()
    assert (tmp_path / ('run2' + LOG_EXTENSION)).is_file()


def test_the_handlers_come_back_off_through_an_exception(tmp_path):
    """A run that raises still gives the caller their logging back."""
    with as_a_fresh_interpreter() as package:
        with pytest.raises(ZeroDivisionError):
            with run_logging(str(tmp_path / 'run')):
                raise ZeroDivisionError

        assert package.handlers == []


def test_what_a_run_can_be_repeated_from_reaches_a_quietened_caller_s_log(tmp_path):
    """The lines that survive a caller's own level, and why they must.

    ``logging.basicConfig()`` with no arguments sets WARNING, which is the
    commonest configuration there is. Everything the run says at INFO is
    dropped for that caller, by their decision and #55's rule that their
    configuration wins.

    Two lines are the exception, logged at WARNING, and they are the exception
    for one reason: a run has to be repeatable by reading its own log. A
    generated seed exists nowhere but its line, so losing it makes the run
    unrepeatable and nothing in the outputs can say what it was. The
    maths-library thread count is the second, and #61 is why: it moves an
    atom's flux in its last digits, and until then nothing in DAWIS chose one
    or wrote one down. Every other banner line can be recovered from the
    configuration the caller already holds.
    """
    import dawis
    from tinyrun import TINY_NAME, TINY_PARAMS, tiny_image

    with as_a_fresh_interpreter() as package:
        root = logging.getLogger()
        root.addHandler(logging.NullHandler())
        root.setLevel(logging.WARNING)
        package.propagate = True

        result = dawis.run(tiny_image(), dawis.Config(**TINY_PARAMS),
                           outdir = str(tmp_path), name = TINY_NAME)

    log = Path('%s%s' % (result.outpath, LOG_EXTENSION)).read_text()

    assert 'Random seed = %d' % result.seed in log, (
        'a run quietened to WARNING lost the only record of the seed it '
        'drew, so it can never be repeated'
    )
    assert 'Maths library threads = 1' in log, (
        'a run quietened to WARNING lost the record of the thread count it '
        'ran at, so a rerun that differs in its last digits has nothing to '
        'say why (#61)'
    )
