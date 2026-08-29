"""Tier 0 — the pure half of the simulation gallery (#68, spec #67).

The gallery generator lives in the top-level ``simulations/`` directory and
runs in its own conda environment, ``dawis-sim``, because it needs GalSim and
DAWIS must not. That split is the whole reason the generator is built in three
layers: a **recipe** layer that imports numpy only, a **render** layer that
hands the recipe to GalSim, and a **write** layer that puts files on disk.

Only the recipe layer is a test seam, and this module is the suite that sits
on it. Nothing here imports GalSim, so these tests run in the ordinary test
environment in seconds, exactly like the à trous and Anscombe property suites.

Two decisions from #67 are pinned here, both named in #68's acceptance
criteria.

**Seed derivation.** The seed is a function of class, regime and realisation
and of nothing else — not the tier, and above all not the restoration variant
under test. That is what makes a variant comparison paired by construction:
every variant opens the same file, so no discipline is required of the person
running the sweep. A seed that moved under a refactor would silently break the
pairing of every number already recorded, so the values are pinned to
literals, not merely to properties.

**The storage contract.** #67 states the entry directory name and the six file
names so that the metric half of #25 can be written against them before any
image exists. A contract a later reader has to infer from a generator run is
not a contract.

The wing-radius refusal and the manifest keys are pinned here too: both are
recipe decisions, both are #68 acceptance criteria, and both are cheaper to
defend here than in a GalSim run.
"""

from __future__ import annotations

import inspect
import json
import sys
from dataclasses import fields, replace
from pathlib import Path

import pytest

# ``simulations`` is deliberately not part of the distributed package (see
# ``test_package_surface.py``), so it is not importable off the install. It is
# a sibling of ``dawis`` in the source tree, and that is where it is found.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from simulations import recipe as recipe_module  # noqa: E402

#: The one path #68 builds. Every other combination of the four axes is
#: refused until its own ticket, so a half-decided number never reaches disk.
TRACER = dict(tier='full', source_class='sersic_n4', regime='balanced', realisation=0)


@pytest.fixture(scope='module')
def tracer_recipe():
    """The single entry #68 builds, described once for the whole module."""
    return recipe_module.build_recipe(**TRACER)


# --------------------------------------------------------------------------
# Seed derivation
# --------------------------------------------------------------------------


def test_the_seed_is_a_function_of_class_regime_and_realisation_only():
    """The three inputs #67 names, and no fourth one.

    ``derive_seed`` takes exactly those three arguments, so there is no fourth
    input a caller could pass by accident. Asserting the signature is how the
    rule is kept true of code rather than of one call.
    """
    parameters = list(inspect.signature(recipe_module.derive_seed).parameters)

    assert parameters == ['source_class', 'regime', 'realisation']


def test_the_seed_is_stable_across_calls():
    """Same inputs, same seed — the property the baked images rest on."""
    first = recipe_module.derive_seed('sersic_n4', 'balanced', 0)
    second = recipe_module.derive_seed('sersic_n4', 'balanced', 0)

    assert first == second


@pytest.mark.parametrize(
    ('source_class', 'regime', 'realisation', 'expected'),
    [
        ('sersic_n4', 'balanced', 0, 2079390869432466733),
        ('sersic_n4', 'balanced', 1, 5040620112393017472),
        ('sersic_n4', 'balanced', 2, 4622553546721531895),
        ('sersic_n4', 'read_noise', 0, 3605341216902279743),
        ('sersic_n4', 'photon', 0, 6478791949852208122),
        ('point', 'balanced', 0, 4830625683245079366),
        ('sersic_n1_halo', 'balanced', 0, 6287667974108719768),
        ('blend', 'balanced', 0, 8455701417552405588),
    ],
)
def test_the_seed_values_are_pinned(source_class, regime, realisation, expected):
    """The derivation is a published constant, not an implementation detail.

    Every number the battery ever records is paired to the image bytes these
    seeds produce. A refactor that moved a seed would leave the old numbers
    silently incomparable with the new ones and nothing would report it, so
    the values are written down here rather than merely constrained.
    """
    assert recipe_module.derive_seed(source_class, regime, realisation) == expected


@pytest.mark.parametrize(
    ('left', 'right'),
    [
        (('sersic_n4', 'balanced', 0), ('point', 'balanced', 0)),
        (('sersic_n4', 'balanced', 0), ('sersic_n4', 'photon', 0)),
        (('sersic_n4', 'balanced', 0), ('sersic_n4', 'balanced', 1)),
    ],
)
def test_changing_any_of_the_three_inputs_changes_the_seed(left, right):
    """Two entries that differ at all differ in their noise draw."""
    assert recipe_module.derive_seed(*left) != recipe_module.derive_seed(*right)


def test_the_seed_fits_the_range_a_generator_can_use():
    """A non-negative 63-bit integer, which numpy and GalSim both accept."""
    seed = recipe_module.derive_seed('sersic_n4', 'balanced', 0)

    assert isinstance(seed, int)
    assert 0 <= seed < 2**63


def test_the_tier_does_not_reach_the_seed(tracer_recipe):
    """Both tiers of one entry draw the same noise.

    #67 lists tier as an axis of the gallery and *not* as an input to the
    seed. The two tiers are different image sizes of the same experiment, so
    they share a realisation.
    """
    small = recipe_module.build_recipe(
        tier='small', source_class='sersic_n4', regime='balanced', realisation=0
    )

    assert small.seed == tracer_recipe.seed


def test_the_recipe_carries_the_derived_seed(tracer_recipe):
    """The seed on a recipe is the one ``derive_seed`` gives, not another."""
    assert tracer_recipe.seed == recipe_module.derive_seed(
        TRACER['source_class'], TRACER['regime'], TRACER['realisation']
    )


# --------------------------------------------------------------------------
# The storage contract
# --------------------------------------------------------------------------


def test_the_entry_directory_name_matches_the_contract():
    """``<class>_<regime>_r<N>``, exactly as #67 writes it."""
    assert recipe_module.entry_name('sersic_n4', 'balanced', 0) == 'sersic_n4_balanced_r0'
    assert recipe_module.entry_name('blend', 'photon', 2) == 'blend_photon_r2'


def test_the_entry_directory_sits_under_the_tier(tmp_path, tracer_recipe):
    """``<root>/<tier>/<class>_<regime>_r<N>`` — one directory, one run."""
    directory = tracer_recipe.entry_directory(tmp_path)

    assert directory == tmp_path / 'full' / 'sersic_n4_balanced_r0'


def test_the_six_file_names_are_the_contract():
    """The metric half of #25 is written against these names.

    They are asserted as a set of literals rather than derived from anything,
    because a name that changed with the code would not be a contract.
    """
    assert recipe_module.ENTRY_FILE_NAMES == (
        'image.fits',
        'truth_preconv.fits',
        'truth_conv.fits',
        'psf.fits',
        'sources.csv',
        'manifest.json',
    )


def test_every_output_path_sits_inside_the_entry_directory(tmp_path, tracer_recipe):
    """The recipe hands the write layer complete paths, not fragments."""
    directory = tracer_recipe.entry_directory(tmp_path)
    paths = tracer_recipe.output_paths(tmp_path)

    assert set(paths) == set(recipe_module.ENTRY_FILE_NAMES)
    for name, path in paths.items():
        assert path == directory / name


def test_the_manifest_path_is_the_one_the_skip_reads(tmp_path, tracer_recipe):
    """Idempotence keys on ``manifest.json``, so it must be a named path."""
    paths = tracer_recipe.output_paths(tmp_path)

    assert paths['manifest.json'].name == 'manifest.json'


# --------------------------------------------------------------------------
# The manifest
# --------------------------------------------------------------------------


def test_the_manifest_records_every_key_sixty_seven_names(tracer_recipe):
    """Provenance a regeneration months later can be rebuilt from.

    The GalSim version and the source commit are filled in by the generator,
    which knows them; the recipe leaves the keys present and empty so that a
    missing one is a failure here rather than a surprise on disk.
    """
    manifest = tracer_recipe.manifest()

    required = {
        'tier',
        'class',
        'regime',
        'realisation',
        'seed',
        'noise',
        'image_shape',
        'grid_shape',
        'grid_spacing',
        'psf_type',
        'psf_fwhm',
        'galsim_version',
        'numpy_version',
        'source_commit',
    }

    assert required <= set(manifest)


def test_the_manifest_records_the_true_noise_parameters(tracer_recipe):
    """Sky level, read noise and gain, in camera terms.

    These are recorded for a human reading a result. They are never passed
    into a DAWIS run: the algorithm estimates the noise itself, as it would on
    real data (#67).
    """
    noise = tracer_recipe.manifest()['noise']

    assert set(noise) == {'sky_level', 'read_noise', 'gain'}
    assert all(isinstance(value, float) for value in noise.values())


def test_the_manifest_describes_the_axes_it_was_built_from(tracer_recipe):
    manifest = tracer_recipe.manifest()

    assert manifest['tier'] == 'full'
    assert manifest['class'] == 'sersic_n4'
    assert manifest['regime'] == 'balanced'
    assert manifest['realisation'] == 0


def test_the_manifest_describes_the_geometry(tracer_recipe):
    """Image dimensions, grid shape and spacing, as #68 requires."""
    manifest = tracer_recipe.manifest()

    assert manifest['image_shape'] == [1024, 1024]
    assert manifest['grid_shape'] == [5, 5]
    assert manifest['grid_spacing'] == pytest.approx(tracer_recipe.grid_spacing)
    assert 0.0 < manifest['grid_spacing'] < 1024


def test_the_manifest_is_json_serialisable(tracer_recipe):
    """It is written with ``json.dump``, so numpy scalars would break it."""
    text = json.dumps(tracer_recipe.manifest())

    assert json.loads(text)['class'] == 'sersic_n4'


# --------------------------------------------------------------------------
# The grid and the sources
# --------------------------------------------------------------------------


def test_the_full_tier_lays_out_one_source_per_grid_cell(tracer_recipe):
    assert tracer_recipe.grid_shape == (5, 5)
    assert len(tracer_recipe.sources) == 25


def test_every_source_carries_the_columns_the_table_promises(tracer_recipe):
    """``sources.csv`` is one row per source with these fields (#67)."""
    names = {field.name for field in fields(tracer_recipe.sources[0])}

    assert {
        'source_id',
        'x',
        'y',
        'magnitude',
        'flux',
        'sersic_index',
        'half_light_radius',
        'e1',
        'e2',
    } <= names


def test_the_magnitude_ladder_has_one_step_per_grid_column(tracer_recipe):
    """Brightness varies along one axis; the other axis is position jitter.

    Five brightness steps with five samples each is the shape "bin residuals
    by SNR" wants, and it is why the grid is spent this way rather than on
    twenty-five distinct brightnesses.
    """
    columns, _ = tracer_recipe.grid_shape
    by_column = {}
    for source in tracer_recipe.sources:
        by_column.setdefault(source.column, set()).add(source.magnitude)

    assert len(by_column) == columns
    assert all(len(values) == 1 for values in by_column.values())


def test_the_magnitude_ladder_is_monotonic(tracer_recipe):
    """Ordered brightest to faintest, so a column index reads as an SNR bin."""
    ladder = tracer_recipe.magnitude_ladder

    assert list(ladder) == sorted(ladder)


def test_the_magnitude_ladder_spans_the_intended_snr_range(tracer_recipe):
    """Roughly 3 (marginally detectable) to 1000 (bright), per #67."""
    ladder = tracer_recipe.peak_snr_ladder

    assert ladder[0] == pytest.approx(1000.0, rel=0.05)
    assert ladder[-1] == pytest.approx(3.0, rel=0.05)
    assert list(ladder) == sorted(ladder, reverse=True)


def test_the_sources_sit_on_the_tier_grid(tracer_recipe):
    """Evenly spaced grid positions, plus a bounded jitter around each.

    The grid is inset rather than a tiling of the image: the outermost
    positions sit an edge margin in from the border, so that a bright source's
    truncated profile is never cut by it.
    """
    spacing = tracer_recipe.grid_spacing
    columns, rows = tracer_recipe.grid_shape
    origin = min(
        source.x - source.column * spacing for source in tracer_recipe.sources
    )

    for source in tracer_recipe.sources:
        assert abs(source.x - (origin + source.column * spacing)) <= (
            2.0 * tracer_recipe.jitter_amplitude
        )
        assert abs(source.y - (origin + source.row * spacing)) <= (
            2.0 * tracer_recipe.jitter_amplitude
        )

    # The last position plus its own margin still fits the image.
    assert origin + (columns - 1) * spacing < tracer_recipe.image_shape[1]
    assert origin + (rows - 1) * spacing < tracer_recipe.image_shape[0]


def test_every_source_of_one_brightness_gets_its_own_jitter(tracer_recipe):
    """The second grid axis buys independent samples, not copies.

    Five sources share a magnitude. If they shared an offset too they would be
    the same measurement five times over, and the samples-per-SNR-bin count
    #67 relies on would be a fiction.
    """
    spacing = tracer_recipe.grid_spacing
    by_column: dict[int, set[tuple[float, float]]] = {}
    for source in tracer_recipe.sources:
        offset = (
            round(source.x - source.column * spacing, 9),
            round(source.y - source.row * spacing, 9),
        )
        by_column.setdefault(source.column, set()).add(offset)

    columns, rows = tracer_recipe.grid_shape
    assert all(len(offsets) == rows for offsets in by_column.values())
    assert len(by_column) == columns


def test_the_recipe_is_reproducible(tracer_recipe):
    """Two builds of one entry agree source for source.

    The jitter and the ellipticities are drawn from the derived seed, so a
    rebuild after a crash produces the same entry rather than a new one.
    """
    rebuilt = recipe_module.build_recipe(**TRACER)

    assert rebuilt.sources == tracer_recipe.sources


# --------------------------------------------------------------------------
# The wing-radius refusal
# --------------------------------------------------------------------------


def test_the_tracer_entry_passes_the_wing_radius_check(tracer_recipe):
    """The entry #68 builds is a real isolated-galaxy field, not a blend."""
    assert tracer_recipe.wing_radius < tracer_recipe.grid_spacing


def test_a_grid_tighter_than_the_brightest_wings_is_refused():
    """Without this the isolated class silently becomes a second blend class.

    #67 makes the check the generator's own, not a test's, because it must
    fire at generation time when a human is present. The refusal is asserted
    here so the message is pinned as well as the behaviour.
    """
    with pytest.raises(recipe_module.GridTooTightError) as raised:
        recipe_module.check_grid_spacing(wing_radius=250.0, grid_spacing=204.8)

    assert 'wing radius' in str(raised.value)


def test_a_grid_wider_than_the_brightest_wings_is_accepted():
    recipe_module.check_grid_spacing(wing_radius=105.0, grid_spacing=204.8)


def test_the_wing_radius_is_not_bounded_by_the_truncation(tracer_recipe):
    """It is measured on a far wider probe than the one that gets rendered.

    Measuring it on the rendered profile would make the radius its own
    prisoner — the cut bounds the profile, so no brightness could ever push
    the measurement past it and :func:`check_grid_spacing` could never fire.
    That is the one scenario #67 names for the refusal, so the measurement is
    taken past the cut.
    """
    truncation_pixels = (
        recipe_module.SERSIC_N4_HALF_LIGHT_RADIUS
        * recipe_module.SERSIC_TRUNCATION_FACTOR
        / recipe_module.PIXEL_SCALE
    )

    assert tracer_recipe.wing_radius < truncation_pixels


def test_a_ladder_bright_enough_to_outgrow_the_truncation_is_refused(monkeypatch):
    """Brightening the ladder far enough really does trip a refusal.

    A hundredfold brighter and the source is still above the background well
    past where it gets cut, which would show as a hard circular edge in the
    truth image.
    """
    monkeypatch.setattr(recipe_module, 'PEAK_SNR_BRIGHTEST', 1e5)

    with pytest.raises(recipe_module.TruncationTooTightError):
        recipe_module.build_recipe(**TRACER)


def test_a_truncation_past_the_wings_is_accepted():
    recipe_module.check_truncation(wing_radius=52.0, truncation_radius=80.0)


def test_the_tracer_entry_keeps_every_source_clear_of_the_image_edge(tracer_recipe):
    """Nothing is cut off, so the analytic flux sum is recoverable.

    The generator's flux check compares the rendered truth image against that
    sum. Light falling off the edge would make the check fail without saying
    whether the render or the layout was at fault.
    """
    recipe_module.check_image_margin(
        tracer_recipe.sources,
        tracer_recipe.image_shape[0],
        margin=6.0 * tracer_recipe.psf_fwhm / tracer_recipe.pixel_scale,
        pixel_scale=tracer_recipe.pixel_scale,
    )


def test_a_source_cut_by_the_image_edge_is_refused(tracer_recipe):
    clipped = replace(tracer_recipe.sources[0], x=5.0, y=5.0)

    with pytest.raises(recipe_module.SourceClippedError) as raised:
        recipe_module.check_image_margin(
            (clipped,),
            tracer_recipe.image_shape[0],
            margin=0.0,
            pixel_scale=tracer_recipe.pixel_scale,
        )

    assert 'outside' in str(raised.value)


# --------------------------------------------------------------------------
# The axes that are not built yet
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    'axes',
    [
        dict(TRACER, source_class='point'),
        dict(TRACER, source_class='sersic_n1_halo'),
        dict(TRACER, source_class='blend'),
        dict(TRACER, regime='read_noise'),
        dict(TRACER, regime='photon'),
        dict(TRACER, realisation=1),
    ],
)
def test_an_axis_value_this_ticket_did_not_decide_is_refused(axes):
    """#68 builds one narrow path; the rest belong to their own tickets.

    A recipe that guessed at the undecided numbers would put an entry on disk
    that looks as authoritative as a decided one. Refusing is the honest
    behaviour until the ticket that decides those numbers lands.
    """
    with pytest.raises(recipe_module.AxisNotBuiltError):
        recipe_module.build_recipe(**axes)


def test_an_unknown_axis_value_is_refused():
    with pytest.raises(recipe_module.AxisNotBuiltError):
        recipe_module.build_recipe(
            tier='enormous', source_class='sersic_n4', regime='balanced', realisation=0
        )


def test_the_small_tier_is_available_because_it_only_changes_geometry():
    """Both tiers are decided by #67 outright — size and grid shape, nothing else."""
    small = recipe_module.build_recipe(
        tier='small', source_class='sersic_n4', regime='balanced', realisation=0
    )

    assert small.image_shape == (512, 512)
    assert small.grid_shape == (3, 3)
    assert len(small.sources) == 9


# --------------------------------------------------------------------------
# The noise regime
# --------------------------------------------------------------------------


def test_the_balanced_regime_matches_the_gallery_image_statistics(tracer_recipe):
    """Anchored to the packaged HST ACS F814W image, per #67.

    Median ~0.0108 and standard deviation ~0.0118 are the numbers measured on
    ``dawis/gallery/tidal_group_f814w_sci.fits``, so the camera parameters here
    correspond to a real observation rather than to a round number.
    """
    noise = tracer_recipe.noise

    assert noise.sky_level == pytest.approx(0.0108, rel=0.02)
    assert noise.sigma == pytest.approx(0.0118, rel=0.02)


def test_the_balanced_regime_splits_its_variance_between_the_two_sources(
    tracer_recipe,
):
    """Balanced means photon and read noise contribute about equally."""
    noise = tracer_recipe.noise

    assert noise.photon_variance == pytest.approx(noise.read_variance, rel=0.05)


def test_the_noise_is_poisson_gaussian_not_pure_gaussian(tracer_recipe):
    """A gain finite enough for the photon term to matter (#67)."""
    noise = tracer_recipe.noise

    assert noise.gain > 0.0
    assert noise.photon_variance > 0.0
    assert noise.read_noise > 0.0
