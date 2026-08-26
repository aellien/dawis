"""Tier 0 — the configuration object and the run's one generator (#53).

Three things are pinned here, one per group of acceptance criteria.

**Validation happens at construction.** Every knob has a range, a handful are
only wrong in company, and both kinds fail before a run opens its image
rather than an hour into the greedy loop. The tests name the knob and read the
message, because a validation error whose message does not say what to change
is barely better than the crash it replaced.

**One generator reaches every consumer.** A run makes one
:class:`numpy.random.RandomState` and threads it to the Anscombe fit, the
Anscombe transform of each residual and the inpainting. The three used to be
independent unseeded calls into the global :mod:`numpy.random` state, which is
why two runs of one input diverged.

**A seeded run is bit-reproducible, and an unseeded one is reproducible after
the fact.** Both are checked end to end, against a 64x64 image that runs the
whole pipeline in half a second, because the property is about what a *run*
does and not about what a function returns.

The compatibility of the generator family is checked too, and it is the least
obvious test in the file: ``RandomState`` is used rather than the newer
``Generator`` specifically so the draws match what the published algorithm and
every recorded golden were produced from. That is a science constraint wearing
the clothes of a library choice, so it is asserted rather than left in prose.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import dawis
from dawis import parallel
from tinyrun import TINY_NAME, TINY_PARAMS, run_quiet
from tinyrun import tiny_image as make_tiny_image

from dawis.config import (
    BOUNDARY_CONDITIONS,
    DEFAULT_CG_MAX_ITER,
    OUTPUT_FORMATS,
    PARALLEL_BACKENDS,
    SEED_BITS,
    Config,
    make_rng,
    resolve_seed,
)
from dawis.exceptions import ConfigError, DawisError, DawisValueError

#: Seed used wherever a test needs a pinned one. Explicit seeds are for tests;
#: the science default is ``None``.
SEED = 20260823


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_the_default_configuration_is_the_published_one():
    """A bare ``Config`` is what the pipeline has always run."""
    config = Config()

    assert config.seed is None
    assert config.cg_max_iter == DEFAULT_CG_MAX_ITER
    assert config.conditions == 'loop'
    assert config.outfile_format == 'pickle'
    assert config.n_workers == 1
    assert config.backend is None
    assert config.n_threads == 1


def test_the_solver_iteration_cap_defaults_to_the_published_200():
    """#26 owns the value; 200 is where its sweep starts.

    Checked as a number rather than against the constant, because the point of
    the criterion is *which* number, and asserting the constant equals itself
    would pass at any value.
    """
    assert DEFAULT_CG_MAX_ITER == 200
    assert Config().cg_max_iter == 200


def test_the_solver_inherits_that_cap_when_nobody_passes_one():
    """The default reaches the solver, not just the configuration object.

    ``7c5ba8e`` moved this default from 200 to 3 and no configuration file
    mentioned it either way, which is exactly how the drift went unnoticed.
    """
    import inspect

    signature = inspect.signature(dawis.interscale_tree.CG_minimization)

    assert signature.parameters['max_iter'].default == DEFAULT_CG_MAX_ITER


def test_every_knob_has_exactly_one_default():
    """One knob, one default, in one place (#54).

    ``synthesis_by_analysis`` used to restate all twenty-nine knobs as
    keywords and build a ``Config`` out of them, so every default was written
    twice with nothing in the language tying the two together. A pair that
    drifts is exactly the ``7c5ba8e`` failure — a default that moved with no
    configuration file mentioning it either way. #54 deleted the keyword wall:
    the entry point takes the ``Config`` itself, so there is no second copy
    left to drift.
    """
    import inspect

    where_a_run_reads_and_writes = {'image', 'config', 'outdir', 'name',
                                    'header'}
    signature = inspect.signature(dawis.run)

    assert set(signature.parameters) == where_a_run_reads_and_writes, (
        'the entry point takes what a run works on and where it writes, and '
        'nothing else')

    restated = set(signature.parameters) & set(Config.__dataclass_fields__)
    assert restated == set(), (
        'these knobs have a second default at the entry point: %s'
        % sorted(restated))


def test_the_configuration_cannot_be_changed_after_it_is_checked():
    """Frozen, because validation only guarantees what actually runs."""
    config = Config()

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.tau = 0.5


# ---------------------------------------------------------------------------
# One-knob validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('knob, value', [
    ('n_workers', 0),
    ('n_threads', 0),
    ('max_iter', 0),
    ('size_patch', 0),
    ('cg_max_iter', 0),
    ('starting_level', -1),
    ('lvl_sep_big', -1),
    ('lvl_deblend', -1),
    ('lvl_sep_lin', -1),
    ('lvl_sep_op', -1),
    ('n_sigmas', 0),
    ('ceps', 0),
    ('scale_lvl_eps', 0),
    ('iptd_sigma', -1),
    ('n_levels', 0),
    ('tau', 0),
    ('tau', 1.5),
    ('gamma', 0),
    ('gamma', 1.5),
    ('deblend_contrast', -0.1),
    ('deblend_contrast', 1.5),
    ('extent_sep', -0.1),
    ('extent_sep', 1.5),
    ('ecc_sep', -0.1),
    ('ecc_sep', 1.5),
    ('conditions', 'wrap'),
    ('outfile_format', 'csv'),
    ('backend', 'dask'),
    ('seed', -1),
    ('seed', 2 ** SEED_BITS),
])
def test_a_knob_outside_its_range_is_refused_and_named(knob, value):
    """Every rejection says which knob, and what it was."""
    with pytest.raises(ConfigError) as raised:
        Config(**{knob: value})

    assert knob in str(raised.value)
    assert str(value) in str(raised.value) or repr(value) in str(raised.value)


def test_a_name_that_is_not_a_knob_is_refused_with_the_names_that_are():
    """A name the configuration has no knob for fails as a configuration error.

    Two claims, and #60 needs both. It is a ``ConfigError`` and not the
    :exc:`TypeError` the generated ``__init__`` would raise, and the message
    carries the names that *do* work — which is how a caller still passing
    the pre-1.0 spelling of the worker count is told to write ``n_workers``.
    See ``Config.__new__`` for why.
    """
    with pytest.raises(ConfigError) as raised:
        Config(**{'not_a_knob': 1})

    message = str(raised.value)
    assert 'not_a_knob' in message
    assert 'n_workers' in message


@pytest.mark.parametrize('condition', BOUNDARY_CONDITIONS)
def test_every_accepted_boundary_condition_is_one_the_transform_accepts(condition):
    """The two lists of conditions must agree, and nothing imports one from
    the other.

    ``dawis.config`` is shared vocabulary that every block may import, so it
    imports no block and states its own tuple. That is a second copy of the
    list inside
    :func:`~dawis.wavelets.atrous.boundary_conditions`, and this is what stops
    the copies drifting: a value ``Config`` accepts and the transform rejects
    would be a run that validates and then dies an iteration in, which is the
    exact failure the validation exists to prevent.
    """
    assert Config(conditions = condition).conditions == condition

    dawis.boundary_conditions(np.zeros(4), 4, conditions = condition)


def test_a_condition_neither_of_them_accepts_is_refused_by_both():
    """The agreement runs both ways, so neither list may quietly widen."""
    with pytest.raises(ConfigError):
        Config(conditions = 'wrap')

    with pytest.raises(DawisValueError):
        dawis.boundary_conditions(np.zeros(4), 4, conditions = 'wrap')


@pytest.mark.parametrize('output_format', OUTPUT_FORMATS)
def test_every_supported_output_format_is_accepted(output_format):
    assert Config(outfile_format = output_format).outfile_format == output_format


@pytest.mark.parametrize('backend', PARALLEL_BACKENDS)
def test_every_accepted_backend_is_one_the_parallelism_seam_knows(backend):
    """The two lists of backends must agree, and neither imports the other.

    Same standing rule as the boundary conditions above: ``dawis.config`` is
    shared vocabulary and imports no module of the package, so it states its
    own tuple beside :data:`dawis.parallel.BACKENDS`. This is what stops the
    copies drifting — a backend ``Config`` accepts and the seam cannot build
    would be a run that validates and then dies at the first ``pmap``.
    """
    assert Config(backend = backend).backend == backend

    assert backend in parallel.BACKENDS


def test_the_seam_knows_no_backend_the_configuration_cannot_name():
    """The agreement runs both ways, so neither list may quietly widen."""
    assert set(PARALLEL_BACKENDS) == set(parallel.BACKENDS)


def test_no_backend_is_a_backend_choice_and_means_let_the_worker_count_decide():
    """``None`` is not a missing value: it is what the pipeline has always done.

    One worker selects serial and more than one the default parallel backend.
    Naming a backend overrides that, which is the whole point of the knob.
    """
    assert Config(backend = None).backend is None

    assert parallel.make_backend(None, 1).name == 'serial'
    assert parallel.make_backend(None, 4).name == parallel.DEFAULT_PARALLEL_BACKEND


def test_a_configuration_error_is_a_dawis_error():
    """One base catches everything the package raises on purpose."""
    with pytest.raises(DawisError):
        Config(n_workers = 0)


# ---------------------------------------------------------------------------
# Cross-parameter validation
# ---------------------------------------------------------------------------


def test_a_minimum_span_above_the_maximum_is_refused():
    """Neither number is out of range; together they ask for nothing."""
    with pytest.raises(ConfigError) as raised:
        Config(min_span = 4, max_span = 2)

    message = str(raised.value)
    assert 'min_span' in message and 'max_span' in message


def test_an_equal_span_pair_is_accepted():
    """A tree spanning exactly one number of scales is a real request."""
    assert Config(min_span = 3, max_span = 3).max_span == 3


def test_a_starting_level_at_or_above_the_depth_is_refused():
    """The failure this replaces was silence, not a crash.

    ``range(starting_level, n_levels)`` was simply empty, so the run wrote out
    an untouched copy of its own input and reported success.
    """
    with pytest.raises(ConfigError) as raised:
        Config(starting_level = 9, n_levels = 9)

    message = str(raised.value)
    assert 'starting_level' in message and 'n_levels' in message


def test_the_depth_rule_waits_until_the_depth_is_known():
    """``n_levels=None`` means the image decides, so there is nothing to check."""
    assert Config(starting_level = 9, n_levels = None).n_levels is None


def test_a_scale_knob_above_the_depth_is_allowed():
    """Degenerate is not the same as impossible.

    ``lvl_sep_big`` at its default of 6 is above the six scales a 64x64 image
    supports, which only means no object is ever treated as big. Refusing it
    would refuse the defaults on any small cutout.
    """
    assert Config(n_levels = 6, lvl_sep_big = 6).lvl_sep_big == 6


# ---------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------


def test_an_absent_seed_is_drawn_rather_than_defaulted():
    """``None`` means fresh entropy, which is the science default."""
    drawn = {resolve_seed(None) for _ in range(8)}

    assert len(drawn) == 8
    assert all(0 <= seed < 2 ** SEED_BITS for seed in drawn)


def test_a_given_seed_is_used_as_given():
    assert resolve_seed(SEED) == SEED
    assert resolve_seed(0) == 0


def test_one_seed_gives_one_stream():
    first = make_rng(SEED).normal(0, 1, 32)
    second = make_rng(SEED).normal(0, 1, 32)

    assert np.array_equal(first, second)


def test_different_seeds_give_different_streams():
    assert not np.array_equal(
        make_rng(SEED).normal(0, 1, 32),
        make_rng(SEED + 1).normal(0, 1, 32),
    )


def test_the_generator_reproduces_the_draws_the_published_algorithm_made():
    """The generator family is a science constraint, not a style choice.

    Before #53 every consumer called ``numpy.random`` after the pipeline had
    called ``numpy.random.seed``. ``RandomState`` continues that exact stream;
    ``numpy.random.Generator`` would draw normals by a different method and
    move every recorded golden. If this test ever fails, the goldens have to
    be re-recorded and it is a science change with its own ticket.
    """
    threaded = make_rng(SEED).normal(0, 1, 64)

    np.random.seed(SEED)
    global_state = np.random.normal(0, 1, 64)

    assert np.array_equal(threaded, global_state)


# ---------------------------------------------------------------------------
# The generator reaches the consumers
# ---------------------------------------------------------------------------


def test_the_inpainting_draws_from_the_generator_it_is_given():
    """Same generator, same fill; and it is the run's, not the global one."""
    def inpainted(rng):
        image = np.full((32, 32), -50.0)
        return dawis.inpaint_with_gaussian_noise(
            image, mean = 0.0, sigma = 3.0, iptd_sigma = 3.0, rng = rng)

    assert np.array_equal(inpainted(make_rng(SEED)), inpainted(make_rng(SEED)))
    assert not np.array_equal(inpainted(make_rng(SEED)),
                              inpainted(make_rng(SEED + 1)))


def test_the_anscombe_fill_draws_from_the_generator_it_is_given():
    """The NaN fill is a random draw, and it is part of the run's stream.

    A corner is pushed below the transform's domain, where the square root
    returns NaN, so there is something to fill. The rest is noise, and both
    halves of that matter: an image that is NaN everywhere has no clipped
    statistics to draw the fill from, and a *flat* image has a clipped spread
    of zero, so every fill would be the same number whatever generator drew
    it and the test would pass on a stub.
    """
    image = np.random.default_rng(3).normal(100.0, 5.0, (32, 32))
    image[:4, :4] = -1E6

    def stabilised(rng):
        return dawis.anscombe_transform(image, sigma = 1.0, mean = 0.0,
                                        gain = 1.0, rng = rng)

    assert np.array_equal(stabilised(make_rng(SEED)), stabilised(make_rng(SEED)))
    assert not np.array_equal(stabilised(make_rng(SEED)),
                              stabilised(make_rng(SEED + 1)))


def test_a_consumer_given_no_generator_still_works():
    """The blocks stay usable on their own, which is what a toolbox means."""
    np.random.seed(SEED)
    image = np.full((16, 16), -50.0)

    filled = dawis.inpaint_with_gaussian_noise(
        image, mean = 0.0, sigma = 3.0, iptd_sigma = 3.0)

    assert np.isfinite(filled).all()


def test_the_two_consumers_share_one_stream_rather_than_restarting_it():
    """One generator threaded to both, not one generator each.

    Two consumers each holding their own copy of the seed would draw the same
    numbers as each other, which is a correlation between the inpainting and
    the stabilisation that nothing in the algorithm intends.
    """
    rng = make_rng(SEED)
    first = dawis.inpaint_with_gaussian_noise(
        np.full((16, 16), -50.0), mean = 0.0, sigma = 3.0, iptd_sigma = 3.0,
        rng = rng)
    second = dawis.inpaint_with_gaussian_noise(
        np.full((16, 16), -50.0), mean = 0.0, sigma = 3.0, iptd_sigma = 3.0,
        rng = rng)

    assert not np.array_equal(first, second)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------
#
# A whole pipeline run on a 64x64 frame takes about half a second, which is
# inside Tier 0's budget and is the only way to assert what the criteria
# actually say: that a *run* is reproducible, and that a run that was never
# asked to be still can be.


@pytest.fixture
def tiny_image() -> np.ndarray:
    """A 64x64 noise frame with one source bright enough to be restored."""
    return make_tiny_image()


def run_tiny(image: np.ndarray, directory: Path, **overrides) -> Path:
    """Run the whole pipeline on ``image`` in ``directory``, and return its stem.

    The image goes in as an array (#54) — no FITS file is written, which is
    the property the array-in entry point exists to give the test suite.
    """
    directory.mkdir(parents = True, exist_ok = True)
    config = Config(**dict(TINY_PARAMS, **overrides))
    run_quiet(image, config, outdir = str(directory), name = TINY_NAME)

    return directory / TINY_NAME


def logged_seed(stem: Path) -> int:
    """The seed a finished run recorded in its own log."""
    for line in (stem.parent / ('%s.log.txt' % stem.name)).read_text().splitlines():
        if 'Random seed' in line:
            return int(line.split('=')[1].split('(')[0])
    raise AssertionError('no seed line in %s.log.txt' % stem)


def test_a_seeded_run_is_bit_reproducible(tiny_image, tmp_path):
    """Same seed, same input, identical output — not merely close."""
    first = run_tiny(tiny_image, tmp_path / 'a', seed = SEED)
    second = run_tiny(tiny_image, tmp_path / 'b', seed = SEED)

    for name in ('restored', 'residuals'):
        assert np.array_equal(
            fits.getdata('%s.%s.fits' % (first, name)),
            fits.getdata('%s.%s.fits' % (second, name)),
        ), '%s differs between two runs of the same seed' % name


def test_two_unseeded_runs_do_not_repeat_each_other(tiny_image, tmp_path):
    """The science default is fresh entropy, so two runs must differ.

    This is the property a forgotten pinned seed destroys, and it is invisible
    in any single run's output.
    """
    first = run_tiny(tiny_image, tmp_path / 'a')
    second = run_tiny(tiny_image, tmp_path / 'b')

    assert logged_seed(first) != logged_seed(second)


def test_an_unseeded_run_logs_the_seed_it_drew(tiny_image, tmp_path):
    """The run directory carries what is needed to repeat the run."""
    stem = run_tiny(tiny_image, tmp_path / 'a')
    log = (tmp_path / 'a' / 'tiny.log.txt').read_text()

    assert 'Random seed' in log
    assert 'generated' in log, (
        'the log must say the seed was drawn rather than asked for, or a '
        'reader cannot tell a reproduced run from an original one'
    )
    assert 0 <= logged_seed(stem) < 2 ** SEED_BITS


def test_an_unseeded_run_can_be_repeated_from_its_own_log(tiny_image, tmp_path):
    """The whole point of the default: reproducible *after the fact*.

    A run nobody thought to seed is repeated by reading the seed it wrote down
    and handing it back.
    """
    original = run_tiny(tiny_image, tmp_path / 'a')
    repeated = run_tiny(tiny_image, tmp_path / 'b',
                        seed = logged_seed(original))

    assert np.array_equal(
        fits.getdata('%s.restored.fits' % original),
        fits.getdata('%s.restored.fits' % repeated),
    )


def test_a_depth_rule_fires_even_when_the_image_supplies_the_depth(tiny_image,
                                                                    tmp_path):
    """The one rule that cannot be checked at the first construction.

    ``n_levels=None`` is the default, so the depth is not known until the
    image is open and ``Config`` cannot compare a scale index against it. The
    pipeline rebuilds the configuration as soon as it has read one, and the
    rule fires there — after the read, but before a single wavelet transform,
    which is what the criterion is actually about.
    """
    with pytest.raises(ConfigError):
        run_tiny(tiny_image, tmp_path / 'a', starting_level = 99)

    assert not (tmp_path / 'a' / 'tiny.restored.fits').exists(), (
        'the run solved something before the contradiction was caught')


def test_a_contradictory_configuration_never_becomes_a_run(tmp_path):
    """The check fires where the configuration is built, not where it is used.

    Before #54 the entry point took the knobs and built the ``Config``, so the
    earliest a contradiction could surface was inside the run. Now the caller
    builds it, and a configuration that cannot work has no ``run`` call to
    reach at all — which is why the assertion is that construction raises and
    ``tmp_path`` stays empty.
    """
    with pytest.raises(ConfigError):
        Config(starting_level = 9, n_levels = 9)

    assert list(tmp_path.iterdir()) == []
