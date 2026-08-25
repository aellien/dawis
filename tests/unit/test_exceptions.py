"""Tier 0 — the pruned error family and the rule behind it (#52).

Two things are pinned here.

**The family is the one specced.** Three types over one base, and the two
runtime type checks retired. ``DawisWrongType`` and ``DawisUnknownType`` re-checked
what the typed cube classes of #51 now state statically, so they were removed
rather than carried; the tests below fail if either comes back, at the module
or on the package surface.

**Nothing is swallowed.** #4's rule is that a crash costs an afternoon and a
quietly wrong flux costs a paper. The executable half of it is that no module
in the package catches everything: a bare ``except:`` swallows
:exc:`KeyboardInterrupt` and :exc:`MemoryError` along with the bug it was
written for. This is checked by reading the source tree rather than by
importing it, the same way ``test_viz.py`` checks the matplotlib rule, so a
module that only misbehaves under some other condition cannot hide.

The call sites are checked too, one per retired type, because pruning a family
is only real if what used to raise the removed type raises something sensible
now.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import dawis
from dawis.exceptions import (
    ConfigError,
    DawisDimensionError,
    DawisError,
    DawisValueError,
)

PACKAGE_ROOT = Path(dawis.__file__).resolve().parent

#: The whole family, base included. Nothing else may live in the module.
FAMILY = {
    'ConfigError',
    'DawisDimensionError',
    'DawisError',
    'DawisValueError',
}

#: Retired with #52. Both were runtime type checks; see the module docstring.
RETIRED = {'DawisUnknownType', 'DawisWrongType'}


# --- the family -----------------------------------------------------------


def test_the_module_holds_exactly_the_specced_family():
    """No error type is carried that nothing raises."""
    import dawis.exceptions as exceptions

    defined = {
        name
        for name, value in vars(exceptions).items()
        if isinstance(value, type) and issubclass(value, BaseException)
    }

    assert defined == FAMILY
    assert set(exceptions.__all__) == FAMILY


@pytest.mark.parametrize('name', sorted(RETIRED))
def test_the_retired_type_checks_are_gone(name):
    """A wrong cube is a type error now, not a runtime check."""
    import dawis.exceptions as exceptions

    assert not hasattr(exceptions, name)
    assert not hasattr(dawis, name)
    assert name not in dawis.__all__


@pytest.mark.parametrize(
    'error', [ConfigError, DawisDimensionError, DawisValueError]
)
def test_every_error_is_catchable_as_one(error):
    """``except DawisError`` catches everything the package raises on purpose."""
    assert issubclass(error, DawisError)
    assert issubclass(DawisError, Exception)


def test_the_package_exports_the_family():
    """The surface names the errors, so callers can catch them by import."""
    assert FAMILY <= set(dawis.__all__)
    assert dawis.ConfigError is ConfigError


# --- what each one carries ------------------------------------------------


def test_dimension_error_states_both_ranks():
    error = DawisDimensionError('Not a cube -->', 2, 3)

    assert error.ndim == 2
    assert error.allowed_ndim == 3
    assert str(error) == 'Not a cube --> number of dimensions is 2 instead of 3'


def test_value_error_states_the_value_and_the_allowed_ones():
    error = DawisValueError('Not an allowed condition -->', 'wrap', ['mirror'])

    assert error.value == 'wrap'
    assert error.allowed_value == ['mirror']
    assert str(error) == (
        'Not an allowed condition --> value wrap is not allowed, '
        "must be ['mirror']"
    )


def test_config_error_carries_only_its_message():
    """A cross-parameter rule has no single offending argument to name."""
    error = ConfigError('max_iter must exceed 0 when ceps is set')

    assert str(error) == 'max_iter must exceed 0 when ceps is set'


@pytest.mark.parametrize(
    'error',
    [
        ConfigError('bad'),
        DawisDimensionError('Not a cube -->', 2, 3),
        DawisValueError('Not allowed -->', 'wrap', ['mirror']),
    ],
)
def test_repr_names_the_class_it_is(error):
    """The old family repr'd three of its types as ``DawisDimensionError``."""
    assert repr(error).startswith('%s(' % type(error).__name__)


# --- nothing is swallowed -------------------------------------------------


#: Names that catch everything when written as ``except <name>:``.
CATCH_ALL = {'BaseException', 'Exception'}


def swallowing_handlers(source: Path) -> list[tuple[int, str]]:
    """Every handler in a file that swallows, with the reason it does.

    Two shapes are visible to a reader of the syntax tree, and both are here:
    a handler that names no type at all, and one that names
    :exc:`Exception` or :exc:`BaseException`. Either catches
    :exc:`KeyboardInterrupt` or :exc:`MemoryError` along with whatever it was
    written for.

    A third shape is not visible to any such reader: a handler that names a
    narrow type and then discards it. Its narrowest form — a body of nothing
    but ``pass`` — is caught here; the general form, where the handler logs
    and carries on, is a judgement about whether continuing is right, and no
    syntax tree can make it. :func:`dawis.viz.make_gif` is the one place in
    the package that deliberately does so, and its docstring argues the case.
    """
    found = []
    for node in ast.walk(ast.parse(source.read_text())):
        if not isinstance(node, ast.ExceptHandler):
            continue
        caught = node.type
        if caught is None:
            found.append((node.lineno, 'catches everything'))
        elif isinstance(caught, ast.Name) and caught.id in CATCH_ALL:
            found.append((node.lineno, 'catches %s' % caught.id))
        elif all(isinstance(stmt, ast.Pass) for stmt in node.body):
            found.append((node.lineno, 'discards what it catches'))
    return found


def test_no_module_swallows_an_error():
    """#4's rule, made executable: crashing beats being silently wrong."""
    offenders = [
        '%s:%d %s' % (path.relative_to(PACKAGE_ROOT).as_posix(), lineno, why)
        for path in sorted(PACKAGE_ROOT.rglob('*.py'))
        for lineno, why in swallowing_handlers(path)
    ]

    assert offenders == [], (
        'a run that crashes costs an afternoon and one that is quietly wrong '
        'costs a paper: %s' % ', '.join(offenders)
    )


# --- the call sites the retired types used to serve -----------------------


def test_an_untabulated_wavelet_family_is_a_value_error():
    """``DawisUnknownType`` here was a value check wearing a type's name."""
    from dawis.detection import table_noise_stds

    with pytest.raises(DawisValueError, match='Unknown wavelet type'):
        table_noise_stds(level_max=5, wavelet_type='NOSUCHWAVELET')


def test_a_non_string_boundary_condition_is_a_plain_type_error():
    """The one surviving builtin-type check raises the builtin error."""
    import numpy as np

    from dawis.wavelets import boundary_conditions

    with pytest.raises(TypeError, match='conditions'):
        boundary_conditions(np.zeros(4, dtype=int), 4, conditions=3)


def test_a_non_fits_input_is_a_config_error(tmp_path):
    """The FITS wrapper's own argument check, before it opens anything.

    The path is checked before the file is, so a mistyped extension is a
    configuration error rather than whatever astropy says about a file it
    cannot read — and here, on a path that does not exist at all, a run that
    opened first would raise ``FileNotFoundError`` instead.
    """
    with pytest.raises(ConfigError, match=r'does not end in \.fits'):
        dawis.run_fits(str(tmp_path / 'image.txt'))


def test_an_untyped_fits_file_reads_back_as_an_untyped_cube(tmp_path):
    """A file DAWIS did not write is allowed, and says so honestly."""
    import numpy as np
    from astropy.io import fits

    path = tmp_path / 'untyped.fits'
    fits.PrimaryHDU(np.zeros((4, 4, 2))).writeto(path)

    assert dawis.Cube.from_fits(str(path)).dctype == 'NONE'


def test_a_fits_file_that_lies_about_its_type_crashes(tmp_path):
    """The swallow this ticket closed, pinned by the case it used to hide.

    A header saying ``DCTYPE = 'WAVELET'`` and then carrying no ``WAVTYPE``
    is broken. Until #52 the whole dispatch sat inside one handler, so the
    cube came back silently typed ``'NONE'`` — an untyped cube built out of
    wavelet coefficients, which is exactly the quietly-wrong outcome #4's
    rule exists to prevent.
    """
    import numpy as np
    from astropy.io import fits

    hdu = fits.PrimaryHDU(np.zeros((4, 4, 2)))
    hdu.header['DCTYPE'] = 'WAVELET'
    path = tmp_path / 'lying.fits'
    hdu.writeto(path)

    with pytest.raises(KeyError, match='WAVTYPE'):
        dawis.WaveletCube.from_fits(str(path))


def test_a_wrong_rank_array_is_still_a_dimension_error():
    """Rank is data, not type, so this check stays a runtime one."""
    import numpy as np

    with pytest.raises(DawisDimensionError, match='Not a cube'):
        dawis.Cube(np.zeros((8, 8)))
