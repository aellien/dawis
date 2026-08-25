"""What a run reports: the check at its input, and the object it hands back.

The entry point of #54 takes an array and returns something. This module holds
both halves of that "something": :class:`InputDiagnostics`, measured on the
image before the solver starts, and :class:`RunResult`, the object the run ends
with.

Input diagnostics
-----------------

DAWIS is fed images from every instrument and every reduction pipeline there
is, and a few properties of an array make a run mean less than it appears to.
Blank borders read as a real zero level. A masked region left as NaN
propagates through the wavelet transform. A detector's saturated core is a
flat plateau the solver will happily restore as a source. An integer dtype
cannot hold the sky-subtracted image, so the run converts it and the numbers
stop being the ones the caller handed over.

None of that is a reason to refuse the run — the astronomer usually knows, and
a pipeline that stops on a border of zeros is a pipeline nobody can use. So
this **warns and never blocks** (#4): the numbers land on the result, the
concerning ones also go to the log, and the run proceeds either way.

A *warning* here is a line of text, delivered two ways and not three. It goes
to :mod:`logging`, which is where everything else a run says goes and which
therefore puts it in the run's own log file beside the seed; and it stays on
:attr:`InputDiagnostics.warnings`, so a caller working in memory can read it
without configuring logging at all. It is deliberately not a
:func:`warnings.warn`: that channel is for a *caller's* mistake — a
deprecated argument, a misuse — and dedupes by default, so the second run of a
campaign would be told nothing about its second image.

Run result
----------

:class:`RunResult` is what makes the entry point usable without a filesystem.
A run still writes its outputs — Phase A does not change what lands on disk —
but a caller who was handed an array back does not have to read one of those
files to see what happened. It carries the images the run produced, the
configuration it actually used with ``n_levels`` resolved, the seed it drew,
and the diagnostics above.

This module sits at the top level, beside :mod:`dawis.convergence` and
:mod:`dawis.cubes`, for the reason those do: it is vocabulary the pipeline
produces and a caller reads, and no block owns it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dawis.config import Config

__all__ = [
    'SATURATED_FRACTION_WARN',
    'SATURATED_PIXELS_WARN',
    'ZERO_FRACTION_WARN',
    'InputDiagnostics',
    'RunResult',
    'inspect_image',
]

#: Fraction of zero pixels above which the input is worth a warning.
#:
#: A tenth of an image being exactly zero is not noise — it is a blank border,
#: a masked region or a mosaic gap. Every one of them reads to the solver as a
#: real, perfectly quiet sky, and the noise the detection thresholds are quoted
#: in is measured over them.
ZERO_FRACTION_WARN = 0.1

#: Fraction of pixels sharing the brightest value above which saturation is
#: worth a warning.
#:
#: A ten-thousandth of the image pinned at one value is about eighty pixels of
#: a 900x900 frame — far more than a point source's brightest pixel and far
#: fewer than a real diffuse structure.
SATURATED_FRACTION_WARN = 1E-4

#: Pixels that must share the brightest value before a fraction means anything.
#:
#: Every image has a maximum and at least one pixel holds it, so on a small
#: enough frame that single pixel clears any fraction — a 32x32 patch makes it
#: a thousandth of the image on its own. A *plateau* is what says a detector
#: clipped, and a plateau is more than one pixel.
SATURATED_PIXELS_WARN = 1


@dataclass(frozen = True)
class InputDiagnostics:
    """What the image a run was handed looks like, measured before it starts.

    Every field is a plain number, so the whole thing survives being written
    into provenance or a log line. Nothing here stops a run — see the module
    docstring for why.

    Attributes
    ----------
    dtype : str
        The array's dtype, as it spells itself. Recorded because a run is done
        in floating point: a non-floating image cannot hold its own
        sky-subtracted values, so the run converts it first.
    shape : tuple of int
        The array's shape.
    n_pixels : int
        How many pixels that is.
    zero_fraction : float
        Fraction of pixels exactly equal to zero.
    nan_fraction : float
        Fraction of pixels that are NaN — a masked region, most often.
    infinite_fraction : float
        Fraction of pixels that are infinite. Counted apart from the NaNs
        because they arrive from different mistakes, and warned about together
        because they break the wavelet transform the same way.
    n_saturated : int
        How many finite pixels sit at the brightest finite value. One is what
        every image has; more than one is a plateau.
    saturated_fraction : float
        That count as a fraction of the finite pixels. Zero when the image
        holds no finite pixel at all.
    saturation_value : float or None
        That brightest finite value, or ``None`` when there is none.
    warnings : tuple of str
        One line per concerning property, already phrased for a reader. Empty
        when nothing stood out. The pipeline logs these; they are kept here so
        a caller who configured no logging can still see them.
    """

    dtype: str
    shape: tuple[int, ...]
    n_pixels: int
    zero_fraction: float
    nan_fraction: float
    infinite_fraction: float
    n_saturated: int
    saturated_fraction: float
    saturation_value: float | None
    warnings: tuple[str, ...] = ()

    @property
    def ndim(self) -> int:
        """How many axes the array had. Two is an image; anything else is not."""

        return len(self.shape)

    def summary(self) -> str:
        """One line naming every measurement, for a log or a provenance file.

        Built here rather than at the call site so the pipeline stays free of
        arithmetic and the phrasing cannot drift from the fields.
        """

        return ('dtype = %s, shape = %s --> zero fraction = %1.3e, '
                'NaN fraction = %1.3e, infinite fraction = %1.3e, '
                'saturated fraction = %1.3e (%d pixels at %s)'
                % (self.dtype, list(self.shape), self.zero_fraction,
                   self.nan_fraction, self.infinite_fraction,
                   self.saturated_fraction, self.n_saturated,
                   self.saturation_value))


def inspect_image(image: np.ndarray) -> InputDiagnostics:
    """Measure the properties of an input image that make a run mean less.

    Parameters
    ----------
    image : numpy.ndarray
        The image about to be run on. Read only — nothing here modifies it.

    Returns
    -------
    InputDiagnostics
        The measurements, and a warning line for each one that stood out.

    Examples
    --------
    >>> import numpy as np
    >>> diagnostics = inspect_image(np.ones((4, 4)))
    >>> diagnostics.zero_fraction, diagnostics.warnings
    (0.0, ())
    >>> inspect_image(np.zeros((4, 4))).zero_fraction
    1.0
    """

    array = np.asarray(image)
    n_pixels = int(array.size)

    finite = np.isfinite(array)
    n_finite = int(np.count_nonzero(finite))
    if n_pixels:
        nan_fraction = int(np.count_nonzero(np.isnan(array))) / n_pixels
        infinite_fraction = (n_pixels - n_finite) / n_pixels - nan_fraction
        zero_fraction = int(np.count_nonzero(array == 0)) / n_pixels
    else:
        nan_fraction = infinite_fraction = zero_fraction = 0.0

    # Saturation is a *plateau*, not a peak: the brightest value of any image
    # is held by at least one pixel, so what says a detector clipped is how
    # many pixels share it. Measured over the finite pixels alone, because a
    # NaN is not a bright pixel and `max` over an array holding one is NaN.
    if n_finite:
        saturation_value = float(np.max(array[finite]))
        n_saturated = int(np.count_nonzero(array[finite] == saturation_value))
        saturated_fraction = n_saturated / n_finite
    else:
        saturation_value = None
        n_saturated = 0
        saturated_fraction = 0.0

    warnings = []

    if not np.issubdtype(array.dtype, np.floating):
        warnings.append(
            'input dtype is %s --> a run is done in floating point, so the '
            'image is converted to float64 before the sky comes out of it; '
            'convert it yourself to choose the precision'
            % array.dtype
        )

    if nan_fraction + infinite_fraction > 0:
        warnings.append(
            '%.3f%% of the input is NaN and %.3f%% is infinite --> both spread '
            'across a wavelet plane, so structure near them is restored from '
            'values that mean nothing'
            % (100 * nan_fraction, 100 * infinite_fraction)
        )

    if zero_fraction >= ZERO_FRACTION_WARN:
        warnings.append(
            '%.1f%% of the input is exactly zero --> a blank border or a '
            'masked region reads as perfectly quiet sky, and the noise the '
            'detection thresholds are quoted in is measured over it'
            % (100 * zero_fraction)
        )

    if (n_saturated > SATURATED_PIXELS_WARN
            and saturated_fraction >= SATURATED_FRACTION_WARN):
        warnings.append(
            '%.3f%% of the input sits at its brightest value %g --> that is a '
            'plateau rather than a peak, so the run may be restoring a '
            'saturated core as though it were a source'
            % (100 * saturated_fraction, saturation_value)
        )

    return InputDiagnostics(
        dtype = str(array.dtype),
        shape = tuple(int(size) for size in array.shape),
        n_pixels = n_pixels,
        zero_fraction = zero_fraction,
        nan_fraction = nan_fraction,
        infinite_fraction = infinite_fraction,
        n_saturated = n_saturated,
        saturated_fraction = saturated_fraction,
        saturation_value = saturation_value,
        warnings = tuple(warnings),
    )


@dataclass(frozen = True)
class RunResult:
    """Everything one finished run produced, without opening a file.

    A run still writes its outputs, and :attr:`outpath` says where. This
    object is what makes reading them back optional: the entry point takes an
    array and hands one of these back, so a caller — a test, a notebook, a
    campaign script — can work entirely in memory.

    Attributes
    ----------
    config : dawis.config.Config
        The configuration the run actually used, with ``n_levels`` resolved to
        the number the image supported rather than left as ``None``.
    seed : int
        The seed the run's one generator was built from, drawn at run start
        when :attr:`~dawis.config.Config.seed` was ``None``. Pass it back as
        ``Config.seed`` to repeat the run.
    diagnostics : InputDiagnostics
        What the input image looked like — see the module docstring.
    outpath : str
        Path stem every output file hangs off, ``<outdir>/<name>``.
    n_iterations : int
        Greedy iterations the loop ran, across every scale.
    n_atoms : int
        Atoms restored, across every iteration.
    restored : numpy.ndarray
        The sum of every atom.
    residuals : numpy.ndarray
        What the loop could not explain.
    det_err : numpy.ndarray
        The accumulated detection error.
    restored_per_level : numpy.ndarray
        The restored image split by the wavelet scale each atom came from,
        shaped ``(x, y, n_levels)``.
    """

    config: Config
    seed: int
    diagnostics: InputDiagnostics
    outpath: str
    n_iterations: int
    n_atoms: int
    restored: np.ndarray = field(repr = False)
    residuals: np.ndarray = field(repr = False)
    det_err: np.ndarray = field(repr = False)
    restored_per_level: np.ndarray = field(repr = False)
