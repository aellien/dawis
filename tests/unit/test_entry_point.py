"""Tier 0 — the array-in entry point and the FITS wrapper (#54).

Four things this ticket is accountable for, all readable in seconds.

**A run takes an array and a configuration.** ``dawis.run(image, config)``.
No path reaches the solver, which is what lets the whole harness drive runs on
synthetic arrays and lets a caller work entirely in memory.

**A thin wrapper takes a FITS path and delegates.** ``dawis.run_fits`` opens
the file and hands the array on. Thin is the assertion: the same image run
both ways produces the same numbers.

**The core touches no file path.** The wrapper is the only place in the
pipeline that reads one.

**Input diagnostics warn and never block.** Zeros, NaNs, a saturated plateau
and a non-floating dtype are measured before the solver starts, recorded on
the result and logged — and a run with all four of them still finishes.

The end-to-end runs here are on a 64x64 frame capped at two iterations, which
is about half a second each and inside Tier 0's budget. What the pipeline
*produces* is Tier 1's business, not this file's.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import dawis
from tinyrun import TINY_PARAMS, run_quiet
from tinyrun import tiny_image as make_tiny_image

from dawis.config import Config
from dawis.exceptions import ConfigError, DawisDimensionError
from dawis.results import (
    SATURATED_FRACTION_WARN,
    InputDiagnostics,
    RunResult,
    inspect_image,
)

#: The one seed these runs pin, so two of them are comparable.
SEED = 20260825


@pytest.fixture
def tiny_image() -> np.ndarray:
    """A 64x64 noise frame with one source bright enough to be restored."""
    return make_tiny_image()


# ---------------------------------------------------------------------------
# The array-in entry point
# ---------------------------------------------------------------------------


def test_a_run_takes_an_array_and_a_configuration(tiny_image, tmp_path):
    """The acceptance criterion, run: an array in, a result out.

    Nothing is read from disk. The only file in ``tmp_path`` afterwards is
    what the run itself wrote.
    """
    result = run_quiet(tiny_image, Config(seed = SEED, **TINY_PARAMS),
                       outdir = str(tmp_path), name = 'tiny')

    assert isinstance(result, RunResult)
    assert result.seed == SEED
    assert result.restored.shape == tiny_image.shape
    assert result.outpath == str(tmp_path / 'tiny')
    assert Path('%s.restored.fits' % result.outpath).is_file()


def test_a_run_needs_no_configuration_at_all(tiny_image, tmp_path):
    """``Config`` defaults are the published ones, so a bare run is a real run."""
    result = run_quiet(tiny_image, outdir = str(tmp_path))

    assert result.config.cg_max_iter == Config().cg_max_iter
    assert Path(result.outpath).name == 'dawis'


def test_the_configuration_on_the_result_has_the_depth_resolved(tiny_image,
                                                                tmp_path):
    """``n_levels=None`` means *the image decides*, and the result says what it decided."""
    result = run_quiet(tiny_image, Config(seed = SEED, **TINY_PARAMS),
                       outdir = str(tmp_path))

    assert result.config.n_levels == dawis.max_levels(tiny_image.shape)


def test_a_run_does_not_modify_the_array_it_was_handed(tiny_image, tmp_path):
    """The caller still owns its image — including a read-only one."""
    before = np.copy(tiny_image)
    tiny_image.setflags(write = False)

    run_quiet(tiny_image, Config(seed = SEED, **TINY_PARAMS),
              outdir = str(tmp_path))

    np.testing.assert_array_equal(tiny_image, before)


def test_a_run_refuses_an_image_that_is_not_one(tmp_path):
    """A cube is not an image, and failing here beats failing eight calls down."""
    with pytest.raises(DawisDimensionError):
        run_quiet(np.zeros((8, 8, 2)), Config(**TINY_PARAMS),
                  outdir = str(tmp_path))


# ---------------------------------------------------------------------------
# The FITS wrapper
# ---------------------------------------------------------------------------


def test_the_wrapper_takes_a_path_and_produces_the_same_run(tiny_image,
                                                            tmp_path):
    """Thin is the assertion: the wrapper adds a file read and nothing else."""
    path = tmp_path / 'tiny.fits'
    fits.PrimaryHDU(tiny_image).writeto(path)

    array_dir = tmp_path / 'from_array'
    array_dir.mkdir()
    from_array = run_quiet(tiny_image, Config(seed = SEED, **TINY_PARAMS),
                           outdir = str(array_dir), name = 'tiny')

    wrapper_dir = tmp_path / 'from_fits'
    wrapper_dir.mkdir()
    from_fits = dawis.run_fits(str(path), Config(seed = SEED, **TINY_PARAMS),
                               outdir = str(wrapper_dir))

    assert from_fits.n_atoms == from_array.n_atoms
    np.testing.assert_array_equal(from_fits.restored, from_array.restored)


def test_the_wrapper_names_the_run_after_the_file(tiny_image, tmp_path):
    """``ngc1275.fits`` still writes ``ngc1275.restored.fits``, as it always did."""
    path = tmp_path / 'ngc1275.fits'
    fits.PrimaryHDU(tiny_image).writeto(path)

    result = dawis.run_fits(str(path), Config(seed = SEED, **TINY_PARAMS),
                            outdir = str(tmp_path))

    assert Path(result.outpath).name == 'ngc1275'
    assert (tmp_path / 'ngc1275.restored.fits').is_file()


def test_the_wrapper_checks_the_extension_before_it_opens_anything(tmp_path):
    """A run that opened first would raise ``FileNotFoundError`` instead."""
    with pytest.raises(ConfigError, match = r'does not end in \.fits'):
        dawis.run_fits(str(tmp_path / 'image.txt'))


def test_only_the_wrapper_reads_the_input():
    """The core touches no file path — the acceptance criterion, as source.

    ``run`` is handed an array and never resolves one itself. Asserted over
    the syntax tree rather than by mocking, because what matters is that no
    *future* edit puts a read back into the loop.
    """
    module = ast.parse(
        Path(dawis.pipelines.synthesis_by_analysis.__file__).read_text()
    )
    reads_a_file = {
        node.name: any(
            isinstance(child, ast.Name) and child.id == 'read_image'
            for child in ast.walk(node)
        )
        for node in module.body
        if isinstance(node, ast.FunctionDef)
    }

    assert reads_a_file['run_fits']
    assert not reads_a_file['run'], (
        'the array-in entry point reads a file, so the core knows about paths '
        'again'
    )


# ---------------------------------------------------------------------------
# Input diagnostics
# ---------------------------------------------------------------------------


def test_a_clean_image_raises_no_warning():
    """The quiet case has to stay quiet, or nobody reads the loud one."""
    rng = np.random.default_rng(3)
    diagnostics = inspect_image(rng.normal(0.0, 1.0, (32, 32)))

    assert diagnostics.warnings == ()
    assert diagnostics.zero_fraction == 0.0
    assert diagnostics.nan_fraction == 0.0
    assert diagnostics.infinite_fraction == 0.0
    assert diagnostics.dtype == 'float64'
    assert diagnostics.shape == (32, 32)
    assert diagnostics.ndim == 2
    assert diagnostics.n_pixels == 32 * 32


def test_a_blank_border_is_measured_and_warned_about():
    """A quarter of this image is exactly zero — a mosaic gap, not sky."""
    image = np.ones((32, 32))
    image[:16, :16] = 0.0

    diagnostics = inspect_image(image)

    assert diagnostics.zero_fraction == pytest.approx(0.25)
    assert any('exactly zero' in line for line in diagnostics.warnings)


def test_non_finite_pixels_are_measured_and_warned_about():
    """NaN and infinity break the transform the same way, counted apart."""
    image = np.ones((10, 10))
    image[0, 0] = np.nan
    image[0, 1] = np.inf

    diagnostics = inspect_image(image)

    assert diagnostics.nan_fraction == pytest.approx(0.01)
    assert diagnostics.infinite_fraction == pytest.approx(0.01)
    assert any('is NaN' in line for line in diagnostics.warnings)


def test_a_saturated_plateau_is_measured_and_warned_about():
    """One bright pixel is a peak; a thousand at one value is a clipped core."""
    rng = np.random.default_rng(5)
    image = rng.normal(0.0, 1.0, (100, 100))
    image[40:60, 40:60] = 65535.0

    diagnostics = inspect_image(image)

    assert diagnostics.saturation_value == 65535.0
    assert diagnostics.saturated_fraction == pytest.approx(0.04)
    assert any('brightest value' in line for line in diagnostics.warnings)


def test_a_single_brightest_pixel_is_not_saturation():
    """Every image has a maximum. Only a plateau says the detector clipped."""
    rng = np.random.default_rng(5)
    image = rng.normal(0.0, 1.0, (200, 200))
    image[0, 0] = 1E6

    diagnostics = inspect_image(image)

    assert diagnostics.saturated_fraction < SATURATED_FRACTION_WARN
    assert diagnostics.warnings == ()


def test_an_integer_dtype_is_warned_about():
    """A run is done in floating point, so an integer image is converted."""
    diagnostics = inspect_image(np.ones((8, 8), dtype = int))

    assert diagnostics.dtype.startswith('int')
    assert any('dtype' in line for line in diagnostics.warnings)


def test_an_integer_image_is_warned_about_and_still_runs(tmp_path):
    """Warn, never block — including the one diagnostic that changes the run.

    An integer array cannot hold its own sky-subtracted values, so the run
    converts it. Before #54 it raised out of the first subtraction instead,
    which made the dtype warning a warning about nothing.
    """
    image = (make_tiny_image() * 10).astype('int32')

    result = run_quiet(image, Config(seed = SEED, **TINY_PARAMS),
                       outdir = str(tmp_path))

    assert any('dtype' in line for line in result.diagnostics.warnings)
    assert Path('%s.restored.fits' % result.outpath).is_file()


def test_an_image_of_nothing_but_nan_is_still_described():
    """No finite pixel means no brightest one, and no crash saying so."""
    diagnostics = inspect_image(np.full((4, 4), np.nan))

    assert diagnostics.nan_fraction == 1.0
    assert diagnostics.infinite_fraction == 0.0
    assert diagnostics.saturation_value is None
    assert diagnostics.saturated_fraction == 0.0


def test_the_diagnostics_land_on_the_result_and_never_stop_the_run(tmp_path):
    """Warn, never block (#4) — with every warning at once.

    A blank border, a NaN, and a saturated plateau, on an image that still has
    a source in it. The run finishes and says what it saw.
    """
    rng = np.random.default_rng(11)
    image = rng.normal(100.0, 5.0, (64, 64))
    image[28:36, 28:36] += 300.0
    image[:20, :] = 0.0
    image[63, 63] = np.nan
    image[40:48, 40:48] = image[28:36, 28:36].max()

    result = run_quiet(image, Config(seed = SEED, **TINY_PARAMS),
                       outdir = str(tmp_path))

    assert isinstance(result.diagnostics, InputDiagnostics)
    assert len(result.diagnostics.warnings) == 3
    assert 'NaN fraction' in result.diagnostics.summary()
    assert Path('%s.restored.fits' % result.outpath).is_file()


def test_the_warnings_reach_the_runs_log(tiny_image, tmp_path):
    """A number on the result nobody prints is a number nobody reads.

    Read out of the run's own log file rather than out of ``caplog``: the
    pipeline replaces the root logger's handlers with its own, which is what
    #55 is about, so pytest's capturing handler is not there to see it.
    """
    image = np.array(tiny_image)
    image[:, :32] = 0.0

    result = run_quiet(image, Config(seed = SEED, **TINY_PARAMS),
                       outdir = str(tmp_path))
    log = Path('%s.log.txt' % result.outpath).read_text()

    assert 'WARNING' in log
    assert 'exactly zero' in log


def test_the_entry_point_signature_names_no_knob():
    """Thirty keywords became one ``Config`` — the readable half of #54."""
    assert list(inspect.signature(dawis.run).parameters) == [
        'image', 'config', 'outdir', 'name', 'header']
    assert list(inspect.signature(dawis.run_fits).parameters) == [
        'path', 'config', 'outdir', 'name']
