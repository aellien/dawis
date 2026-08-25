"""The errors DAWIS raises, and the ones it deliberately does not.

Three types, one base. The family is small on purpose, and the shape of it
follows one rule taken from #4: **nothing is swallowed**. A run that crashes
costs an afternoon; a run that quietly produces the wrong flux costs a paper.
So an error is either fatal here or it is not an error at all.

That rule cuts both ways, and the second half is the one that keeps the family
small. A science outcome is *not* an error. An iteration that fails to
converge, a tree too small to restore, a level that finds nothing — those are
statuses recorded on the result and read by the caller, never exceptions. The
loop is greedy and expected to run into all of them; raising there would turn
a normal run into a traceback.

What is left is three kinds of genuine mistake:

:class:`DawisDimensionError`
    An array of the wrong rank reached a function that needs a specific one —
    an image handed to a cube, a cube handed to an image routine.
:class:`DawisValueError`
    An argument is the right kind of thing but not an allowed one, such as a
    wavelet family that has no tabulated noise coefficients.
:class:`ConfigError`
    A run was configured in a way that cannot work, including cross-parameter
    rules that no single knob is wrong on its own for. Raised before the
    solver starts, which is the only cheap moment to fail.

Two type-check errors used to stand beside them and are gone with #52.
``DawisWrongType`` and ``DawisUnknownType`` re-checked at runtime what the
type system now states: since #51 the cube family is four distinct classes
rather than one class carrying a ``dctype`` string, so handing a
:class:`~dawis.cubes.SupportCube` to something that wants a
:class:`~dawis.cubes.WaveletCube` is a static type error, caught before the
code runs. The few remaining checks over plain builtin types raise the builtin
:exc:`TypeError`, which is what a caller catches for that anyway; a DAWIS-
specific spelling of it bought nothing.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    'ConfigError',
    'DawisDimensionError',
    'DawisError',
    'DawisValueError',
]


class DawisError(Exception):
    """Base class for every error DAWIS raises.

    Catch this to catch anything the package raises on purpose, and nothing
    it raises by accident.
    """

    def __repr__(self) -> str:
        return '%s(%r)' % (type(self).__name__, str(self))


class DawisDimensionError(DawisError):
    """An array has the wrong number of dimensions.

    Parameters
    ----------
    message : str
        What was being done, ending in an arrow — the allowed and actual
        ranks are appended to it.
    ndim : int
        Rank the array actually has.
    allowed_ndim : int or list of int
        Rank, or ranks, the function accepts.

    Attributes
    ----------
    ndim : int
        As passed.
    allowed_ndim : int or list of int
        As passed.

    Examples
    --------
    >>> raise DawisDimensionError('Not a cube -->', 2, 3)
    Traceback (most recent call last):
    dawis.exceptions.DawisDimensionError: Not a cube --> number of dimensions is 2 instead of 3
    """

    def __init__(
        self,
        message: str,
        ndim: int,
        allowed_ndim: int | list[int],
    ) -> None:
        self.ndim = ndim
        self.allowed_ndim = allowed_ndim
        super().__init__(
            '%s number of dimensions is %s instead of %s'
            % (message, ndim, allowed_ndim)
        )


class DawisValueError(DawisError):
    """An argument holds a value the function does not accept.

    Parameters
    ----------
    message : str
        What was being done, ending in an arrow — the offending and allowed
        values are appended to it.
    value : Any
        Value that was passed and rejected.
    allowed_value : Any
        Value, or collection of values, that would have been accepted.

    Attributes
    ----------
    value : Any
        As passed.
    allowed_value : Any
        As passed.

    Examples
    --------
    >>> raise DawisValueError('Not an allowed condition -->', 'wrap', ['mirror', 'loop'])
    Traceback (most recent call last):
    dawis.exceptions.DawisValueError: Not an allowed condition --> value wrap is not allowed, must be ['mirror', 'loop']
    """

    def __init__(
        self,
        message: str,
        value: Any,
        allowed_value: Any,
    ) -> None:
        self.value = value
        self.allowed_value = allowed_value
        super().__init__(
            '%s value %s is not allowed, must be %s'
            % (message, value, allowed_value)
        )


class ConfigError(DawisError):
    """A run is configured in a way that cannot work.

    Unlike the other two, this one carries a message and nothing else. A
    configuration mistake is as often a contradiction between two knobs as a
    single bad value, and there is no offending argument to name when the
    fault is the pair.

    Raised while a run is being set up — argument checking at a pipeline
    entry point, and the validation ``Config`` will do in ``__post_init__``
    (#53) — never once the solver is running.

    Parameters
    ----------
    message : str
        What is wrong, and what would be right.

    Examples
    --------
    >>> raise ConfigError('Incorrect file extension --> run.txt is not .fits')
    Traceback (most recent call last):
    dawis.exceptions.ConfigError: Incorrect file extension --> run.txt is not .fits
    """
