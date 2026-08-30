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

import numpy as np
import pytest

# ``simulations`` is deliberately not part of the distributed package (see
# ``test_package_surface.py``), so it is not importable off the install. It is
# a sibling of ``dawis`` in the source tree, and that is where it is found.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from simulations import recipe as recipe_module  # noqa: E402

#: The tolerance the generator's flux check runs at. Named here rather than
#: imported, because ``simulations.render`` imports GalSim and this suite is
#: the half of the gallery that runs without it.
FLUX_TOLERANCE = 1e-2

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
        * recipe_module.SERSIC_N4_TRUNCATION_FACTOR
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
        dict(TRACER, regime='read_noise'),
        dict(TRACER, regime='photon'),
        dict(TRACER, realisation=1),
    ],
)
def test_an_axis_value_no_ticket_has_decided_is_refused(axes):
    """Each axis value arrives with the ticket that decides its numbers.

    A recipe that guessed at the undecided ones would put an entry on disk
    that looks as authoritative as a decided one. Refusing is the honest
    behaviour until that ticket lands. #70 lifted the refusal on ``point`` and
    ``sersic_n1_halo`` and #71 the last one, on ``blend``, so only the regime
    and the realisation are still narrowed.
    """
    with pytest.raises(recipe_module.AxisNotBuiltError):
        recipe_module.build_recipe(**axes)


def test_every_class_sixty_seven_specifies_now_builds():
    """The class axis is complete: three profiles and one layout (#71)."""
    for source_class in recipe_module.SOURCE_CLASSES:
        recipe = recipe_module.build_recipe(**dict(TRACER, source_class=source_class))

        assert recipe.source_class == source_class


def test_a_class_with_neither_a_profile_nor_a_layout_is_refused(monkeypatch):
    """A name on the axis without a layout is refused, not crashed on.

    The refusal predates #71 — it is what held ``blend`` back — and #71 keeps
    it rather than dropping it now that every class of #67 builds. #25 already
    names two more classes, cirrus and diffraction spikes, that #67 defers; the
    refusal is what stands between the first of them and a ``KeyError`` from
    the middle of the recipe.
    """
    monkeypatch.setattr(
        recipe_module, 'SOURCE_CLASSES', recipe_module.SOURCE_CLASSES + ('cirrus',)
    )

    with pytest.raises(recipe_module.AxisNotBuiltError):
        recipe_module.build_recipe(**dict(TRACER, source_class='cirrus'))


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


# --------------------------------------------------------------------------
# The other two isolated classes (#70)
# --------------------------------------------------------------------------

#: The three classes of #67 that lay sources out on a brightness ladder. They
#: share the grid outright — magnitude across, jitter down — and differ only
#: in the profile each draws, which is why they are asserted together here
#: rather than one suite per class. ``blend`` is not among them: it spends its
#: grid on separation and flux ratio instead, so it is a different layout and
#: not a different profile.
ISOLATED_CLASSES = ('sersic_n4', 'point', 'sersic_n1_halo')


def build_isolated(source_class: str, tier: str = 'full'):
    """One entry of an isolated class, on :data:`TRACER`'s other three axes.

    Every assertion below is about the class or the tier, so the regime and
    the realisation are held at the tracer's values rather than restated.
    """
    return recipe_module.build_recipe(
        **dict(TRACER, source_class=source_class, tier=tier)
    )


@pytest.fixture(
    scope='module',
    params=[
        (source_class, tier)
        for source_class in ISOLATED_CLASSES
        for tier in ('small', 'full')
    ],
    ids=lambda parameter: '%s-%s' % parameter,
)
def isolated_recipe(request):
    """One entry per isolated class per tier, built once for the module.

    Both tiers, because the grid semantics #70 asks about are a property of
    the class and not of one image size — and because ``small`` is the tier
    with the least room, so a class that lays out correctly there is the
    interesting case rather than the incidental one.
    """
    source_class, tier = request.param
    return build_isolated(source_class, tier=tier)


@pytest.fixture(scope='module')
def point_recipe():
    """The ``point`` entry, for the assertions only that class can carry."""
    return build_isolated('point')


@pytest.fixture(scope='module')
def halo_recipe():
    """The ``sersic_n1_halo`` entry, likewise."""
    return build_isolated('sersic_n1_halo')


# --- the shared grid ------------------------------------------------------


def test_every_isolated_class_fills_the_tier_grid(isolated_recipe):
    """One source per cell, on the tier's own grid — no class has its own."""
    grid_side = recipe_module.TIERS[isolated_recipe.tier].grid_side

    assert isolated_recipe.grid_shape == (grid_side, grid_side)
    assert len(isolated_recipe.sources) == grid_side**2


def test_every_isolated_class_varies_magnitude_across_the_grid(isolated_recipe):
    """One brightness step per grid column, per #67 and #70.

    The other axis is position jitter. Five brightness steps with five samples
    each is the shape "bin residuals by SNR" wants, and it is the semantics
    all three isolated classes share.
    """
    columns, _ = isolated_recipe.grid_shape
    by_column: dict[int, set[float]] = {}
    for source in isolated_recipe.sources:
        by_column.setdefault(source.column, set()).add(source.magnitude)

    assert len(by_column) == columns
    assert all(len(magnitudes) == 1 for magnitudes in by_column.values())


def test_every_isolated_class_jitters_down_the_grid(isolated_recipe):
    """Sources sharing a magnitude get their own offsets, not copies of one."""
    spacing = isolated_recipe.grid_spacing
    by_column: dict[int, set[tuple[float, float]]] = {}
    for source in isolated_recipe.sources:
        by_column.setdefault(source.column, set()).add(
            (
                round(source.x - source.column * spacing, 9),
                round(source.y - source.row * spacing, 9),
            )
        )

    _, rows = isolated_recipe.grid_shape
    assert all(len(offsets) == rows for offsets in by_column.values())


def test_every_isolated_class_spans_the_intended_peak_snr_range(isolated_recipe):
    """Roughly 1000 (bright) down to 3 (marginally detectable), per #67.

    The range is the same for all three classes on purpose. What makes the
    halo the ICL-like case is its *surface* brightness, not a shorter ladder,
    and holding the peak SNR common is what lets a result on one class be read
    against another at matched detectability.
    """
    ladder = isolated_recipe.peak_snr_ladder

    assert ladder[0] == pytest.approx(1000.0, rel=0.05)
    assert ladder[-1] == pytest.approx(3.0, rel=0.05)


def test_every_isolated_class_orders_its_ladder_brightest_first(isolated_recipe):
    """Monotonic, so a column index reads straight off as an SNR bin."""
    assert list(isolated_recipe.peak_snr_ladder) == sorted(
        isolated_recipe.peak_snr_ladder, reverse=True
    )
    assert list(isolated_recipe.magnitude_ladder) == sorted(
        isolated_recipe.magnitude_ladder
    )


def test_every_isolated_class_has_one_ladder_step_per_grid_column(isolated_recipe):
    columns, _ = isolated_recipe.grid_shape

    assert len(isolated_recipe.magnitude_ladder) == columns
    assert len(isolated_recipe.peak_snr_ladder) == columns


@pytest.mark.parametrize('tier', ['small', 'full'])
def test_the_three_isolated_classes_share_their_grid_semantics(tier):
    """The classes differ in profile and in nothing else (#70).

    #70 adds source construction, not new grid logic. If a class ever grew a
    grid of its own, the shape "bin residuals by SNR" would stop meaning the
    same thing across the gallery and this is where that would show.
    """
    recipes = [build_isolated(name, tier=tier) for name in ISOLATED_CLASSES]

    assert len({recipe.grid_shape for recipe in recipes}) == 1
    assert len({recipe.jitter_amplitude for recipe in recipes}) == 1
    assert len({recipe.peak_snr_ladder for recipe in recipes}) == 1


# --- the refusals stay live for the new classes ---------------------------


def test_every_isolated_class_is_an_isolated_field(isolated_recipe):
    """The brightest source's wings stay inside its own cell.

    The ``small`` tier is the binding one: three grid steps on a 512-pixel
    image leave the least room, so a class that fits there fits everywhere.
    This is the check that says the two classes #70 adds are really isolated
    fields and not accidental blend fields.
    """
    assert isolated_recipe.wing_radius < isolated_recipe.grid_spacing


def test_every_isolated_class_stays_isolated_once_the_jitter_has_moved_it(
    isolated_recipe,
):
    """Two neighbours jittered towards each other are still not blended.

    :func:`check_grid_spacing` compares the wings to the *nominal* spacing,
    which is the comparison #67 states. The jitter can close two cells by
    twice its amplitude, so the margin that actually matters is smaller than
    the one the refusal reads — and for the halo, the widest class, it is
    smaller by a third. Pinning it here is what stops a later radius change
    from passing the stated refusal while quietly blending the field.
    """
    closest = isolated_recipe.grid_spacing - 2.0 * isolated_recipe.jitter_amplitude

    assert isolated_recipe.wing_radius < closest


def test_every_isolated_class_keeps_its_sources_off_the_image_edge(isolated_recipe):
    """Nothing is cut, so the generator's analytic flux sum stays recoverable."""
    recipe_module.check_image_margin(
        isolated_recipe.sources,
        isolated_recipe.image_shape[0],
        margin=recipe_module.PSF_MARGIN_FWHM
        * isolated_recipe.psf_fwhm
        / isolated_recipe.pixel_scale,
        pixel_scale=isolated_recipe.pixel_scale,
    )


@pytest.mark.parametrize(
    ('source_class', 'grid_side'),
    [('sersic_n4', 8), ('point', 54), ('sersic_n1_halo', 5)],
)
def test_every_isolated_class_answers_to_the_spacing_refusal(
    monkeypatch, source_class, grid_side
):
    """Narrow the cells past a class's own wings and it refuses to build.

    #70 asks for both new classes to be *subject* to the refusal, and the
    check is called for all three — but only a class whose wings can actually
    reach a neighbour demonstrates it. None can at the two real tiers, which
    is the point of those tiers, so each is given a grid step count chosen to
    be the first that closes its cells past its own wings. The counts differ
    by two orders of magnitude precisely because the classes do: a point
    source's wings end eight pixels out and a halo's seventy-one.
    """
    monkeypatch.setitem(
        recipe_module.TIERS,
        'cramped',
        recipe_module.Tier(image_size=512, grid_side=grid_side),
    )

    with pytest.raises(recipe_module.GridTooTightError):
        build_isolated(source_class, tier='cramped')


def test_the_halo_truncation_lands_below_the_background(halo_recipe):
    """The cut is there to bound the halo, not to shape it.

    An exponential falls off fast, so five half-light radii is already past
    the radius at which the brightest halo on the ladder drops under the
    noise. A cut inside that radius would show as a hard circular edge in the
    truth image, which a metric would read as structure.
    """
    truncation_pixels = (
        recipe_module.SERSIC_N1_HALO_HALF_LIGHT_RADIUS
        * recipe_module.SERSIC_N1_HALO_TRUNCATION_FACTOR
        / recipe_module.PIXEL_SCALE
    )

    assert halo_recipe.wing_radius < truncation_pixels


def test_a_halo_ladder_bright_enough_to_outgrow_its_truncation_is_refused(
    monkeypatch,
):
    """The truncation refusal is live for the halo, as it is for the n=4 class."""
    monkeypatch.setattr(recipe_module, 'PEAK_SNR_BRIGHTEST', 1e5)

    with pytest.raises(recipe_module.TruncationTooTightError):
        build_isolated('sersic_n1_halo')


def test_the_point_class_is_not_subject_to_the_truncation_refusal(point_recipe):
    """A delta function has no cut, so there is no cut that could show.

    The class builds with a truncation radius of zero, which every other
    refusal would read as "smaller than the wings". Skipping the check for
    this class is what lets it build at all, and this pins that it is skipped
    rather than passed by luck.
    """
    assert point_recipe.sources[0].truncation_radius == 0.0
    assert point_recipe.wing_radius > 0.0


def test_the_point_class_wing_radius_is_not_bounded_by_its_own_probe(point_recipe):
    """Measured well inside the probe's reach, so the spacing refusal is live.

    A point source has no truncation to probe past, so the probe needs a
    stated reach of its own. If the measured radius ever sat at that reach it
    would be the probe's edge in disguise, and :func:`check_grid_spacing`
    could never fire — the same trap the Sérsic classes avoid by probing past
    their cut.
    """
    probe_pixels = (
        recipe_module.POINT_WING_PROBE_FWHM
        * recipe_module.PSF_FWHM
        / recipe_module.PIXEL_SCALE
    )

    assert point_recipe.wing_radius < probe_pixels / 2.0


# --- what each class actually is ------------------------------------------


def test_the_point_class_records_a_true_ellipticity_of_zero(point_recipe):
    """Zero is the point source's true *post-PSF* shape, not a missing value.

    A delta function has no shape of its own, so the shape #25's shape-bias
    fit can measure is the PSF's — and #67's PSF is a circular Moffat. The
    truth value that fit compares against is therefore exactly zero, and #70
    asks for it to be recorded rather than left out.
    """
    assert all(source.e1 == 0.0 for source in point_recipe.sources)
    assert all(source.e2 == 0.0 for source in point_recipe.sources)


def test_the_point_class_has_no_sersic_index_and_no_radius(point_recipe):
    """``sources.csv`` is a fixed-column table, so both are recorded as zero.

    Zero is literally true of the half-light radius — a delta encloses all of
    its light at every radius — and is not a Sérsic index any profile takes,
    so it reads as "not a Sérsic" rather than as a value.
    """
    assert all(source.sersic_index == 0.0 for source in point_recipe.sources)
    assert all(source.half_light_radius == 0.0 for source in point_recipe.sources)


def test_the_halo_class_is_an_exponential_with_a_real_ellipticity(halo_recipe):
    """n=1 at the stated radius, and a shape the bias fit can actually miss.

    A round halo would make #25's shape-bias metric a test of nothing on this
    class, which is why the ellipticity is drawn rather than left at zero —
    the point class is the only one for which zero is the truth.
    """
    assert all(source.sersic_index == 1.0 for source in halo_recipe.sources)
    assert all(
        source.half_light_radius
        == recipe_module.SERSIC_N1_HALO_HALF_LIGHT_RADIUS
        for source in halo_recipe.sources
    )
    assert any(source.e1 != 0.0 or source.e2 != 0.0 for source in halo_recipe.sources)


def test_every_class_keeps_its_ellipticity_under_the_ceiling(isolated_recipe):
    for source in isolated_recipe.sources:
        assert np.hypot(source.e1, source.e2) <= recipe_module.MAX_ELLIPTICITY


def test_the_halo_spreads_its_light_far_more_thinly_than_the_bright_galaxy(
    tracer_recipe, halo_recipe
):
    """The ICL-like case, stated as a number rather than as a class name.

    Both ladders reach the same peak SNR, so at the bright end the two classes
    put the same signal in their brightest pixel. The halo needs far more
    total flux to do it, and that ratio *is* the low-surface-brightness
    regime #67 wants the under-restoration case to sit in.
    """
    bright_galaxy = tracer_recipe.sources[0]
    bright_halo = halo_recipe.sources[0]

    assert bright_halo.peak_snr == pytest.approx(bright_galaxy.peak_snr)
    assert bright_halo.flux > 5.0 * bright_galaxy.flux


def test_the_point_class_concentrates_its_light_more_than_any_profile(
    tracer_recipe, point_recipe
):
    """The peaky case: all of the flux inside the PSF, and none outside it."""
    bright_galaxy = tracer_recipe.sources[0]
    bright_point = point_recipe.sources[0]

    assert bright_point.peak_snr == pytest.approx(bright_galaxy.peak_snr)
    assert bright_point.flux < bright_galaxy.flux
    assert point_recipe.wing_radius < tracer_recipe.wing_radius


def test_the_class_table_holds_exactly_the_isolated_classes():
    """``blend`` is absent because it is a different grid, not a profile."""
    assert tuple(recipe_module.SOURCE_PROFILES) == ISOLATED_CLASSES
    assert set(recipe_module.SOURCE_PROFILES) < set(recipe_module.SOURCE_CLASSES)


# --------------------------------------------------------------------------
# The brightness bins the generator weighs (#71)
# --------------------------------------------------------------------------


def test_every_source_falls_in_exactly_one_brightness_bin(isolated_recipe):
    """The bins are built from the sources, so they cover them exactly once."""
    binned = [
        source for group in isolated_recipe.flux_groups for source in group
    ]

    assert sorted(binned, key=lambda source: source.source_id) == sorted(
        isolated_recipe.sources, key=lambda source: source.source_id
    )
    assert len(binned) == len(isolated_recipe.sources)


def test_a_brightness_bin_holds_one_flux(isolated_recipe):
    """One bin per ladder step, brightest first, one flux each."""
    columns, rows = isolated_recipe.grid_shape

    assert len(isolated_recipe.flux_groups) == columns
    for group in isolated_recipe.flux_groups:
        assert len({source.flux for source in group}) == 1
        assert len(group) == rows
    assert [group[0].flux for group in isolated_recipe.flux_groups] == sorted(
        (group[0].flux for group in isolated_recipe.flux_groups), reverse=True
    )


def test_the_faint_bins_are_why_the_check_is_binned_at_all(tracer_recipe):
    """A whole-image flux check cannot see the faint end, so it is not the check.

    The gallery is built out of sources three orders of magnitude apart. The
    faintest bin here is a few parts in a thousand of the entry's total, so
    every source in it could fail to draw and the generator's one per cent
    tolerance on the *total* would still be met. Weighing each bin is what
    makes the check say something about the end the gallery exists to measure.
    """
    faintest = tracer_recipe.flux_groups[-1]
    share = sum(source.flux for source in faintest) / tracer_recipe.total_flux

    assert share < FLUX_TOLERANCE


# --------------------------------------------------------------------------
# The blend class (#71)
# --------------------------------------------------------------------------

#: The class that is a *layout* rather than a profile. It puts a pair in every
#: grid cell, so its two axes are separation and flux ratio instead of
#: magnitude and jitter — which is why none of the shared assertions above
#: applies to it and it gets a section of its own.
BLEND = dict(TRACER, source_class='blend')


def build_blend(tier: str = 'full'):
    """One blend entry, on :data:`TRACER`'s other three axes."""
    return recipe_module.build_recipe(**dict(BLEND, tier=tier))


@pytest.fixture(scope='module', params=['small', 'full'], ids=lambda tier: tier)
def blend_recipe(request):
    """The blend entry of each tier, built once for the module.

    Both tiers, because the layout is a property of the class and not of one
    image size — and because ``small`` is the tier with the least room, so it
    is the one where a pair that does not fit shows up.
    """
    return build_blend(request.param)


def blend_pairs(recipe):
    """The entry's sources as ``(bright, companion)`` tuples, cell by cell."""
    by_id = {source.source_id: source for source in recipe.sources}
    return [
        (source, by_id[source.companion_id])
        for source in recipe.sources
        if source.flux > by_id[source.companion_id].flux
    ]


# --- the two grid axes ----------------------------------------------------


def test_the_blend_class_puts_a_pair_in_every_grid_cell(blend_recipe):
    """Two sources per cell on the tier's own grid, not one (#67)."""
    grid_side = recipe_module.TIERS[blend_recipe.tier].grid_side

    assert blend_recipe.grid_shape == (grid_side, grid_side)
    assert len(blend_recipe.sources) == 2 * grid_side**2
    assert len(blend_pairs(blend_recipe)) == grid_side**2


def test_separation_varies_down_the_rows_and_nowhere_else(blend_recipe):
    """One separation per grid row, and every row a different one (#67).

    This is half of what makes the entry a map rather than a list: read a row
    index and you have read a separation.
    """
    _, rows = blend_recipe.grid_shape
    by_row = {}
    for source in blend_recipe.sources:
        by_row.setdefault(source.row, set()).add(source.separation)

    assert len(by_row) == rows
    assert all(len(separations) == 1 for separations in by_row.values())
    assert len({next(iter(values)) for values in by_row.values()}) == rows


def test_the_flux_ratio_varies_across_the_columns_and_nowhere_else(blend_recipe):
    """One flux ratio per grid column, and every column a different one (#67).

    The other half of the map. The two axes together are why the blend grid
    spends nothing on brightness: that binning is the isolated classes' job.
    """
    columns, _ = blend_recipe.grid_shape
    by_column = {}
    for source in blend_recipe.sources:
        by_column.setdefault(source.column, set()).add(source.flux_ratio)

    assert len(by_column) == columns
    assert all(len(ratios) == 1 for ratios in by_column.values())
    assert len({next(iter(values)) for values in by_column.values()}) == columns


def test_the_two_axes_are_the_two_ladders_the_recipe_carries(blend_recipe):
    """The ladders and the sources say the same thing, one per row and column."""
    columns, rows = blend_recipe.grid_shape

    assert len(blend_recipe.separation_ladder) == rows
    assert len(blend_recipe.flux_ratio_ladder) == columns
    for source in blend_recipe.sources:
        assert source.separation == blend_recipe.separation_ladder[source.row]
        assert source.flux_ratio == blend_recipe.flux_ratio_ladder[source.column]


def test_both_axes_are_monotonic_so_an_index_reads_as_a_value(blend_recipe):
    """Tightest pair first, brightest companion first — like the SNR ladder."""
    assert list(blend_recipe.separation_ladder) == sorted(
        blend_recipe.separation_ladder
    )
    assert list(blend_recipe.flux_ratio_ladder) == sorted(
        blend_recipe.flux_ratio_ladder, reverse=True
    )


def test_the_isolated_classes_grid_semantics_are_not_the_blend_class_s(
    blend_recipe,
):
    """The blend grid is spent on something else, which is the whole point.

    Magnitude across and jitter down is what the three isolated classes do.
    Were the blend class to do it too, #67's fourth class would be a fourth
    copy of the first three and the separation-versus-ratio map would not
    exist.
    """
    by_column = {}
    for _, companion in blend_pairs(blend_recipe):
        by_column.setdefault(companion.column, set()).add(companion.magnitude)

    assert recipe_module.SOURCE_PROFILES.get('blend') is None
    assert all(len(magnitudes) == 1 for magnitudes in by_column.values())
    assert blend_recipe.separation_ladder != ()
    assert build_isolated('sersic_n4').separation_ladder == ()


# --- the fixed bright component -------------------------------------------


def test_the_bright_component_is_one_brightness_over_the_whole_grid(blend_recipe):
    """#67 holds it fixed, so neither grid axis is brightness in disguise."""
    bright = [source for source, _ in blend_pairs(blend_recipe)]

    assert len({source.flux for source in bright}) == 1
    assert len({source.magnitude for source in bright}) == 1
    assert len({source.peak_snr for source in bright}) == 1


def test_the_bright_component_is_the_isolated_ladder_s_top_rung(blend_recipe):
    """The same source the isolated ``sersic_n4`` entry puts in its first column.

    Reusing that rung rather than choosing a brightness of its own is what lets
    a blend result at the widest separation be read straight against the
    isolated entry: the hole is the same hole, so what changed is the
    companion and nothing else.
    """
    isolated = build_isolated('sersic_n4', tier=blend_recipe.tier)
    bright, _ = blend_pairs(blend_recipe)[0]

    assert bright.peak_snr == pytest.approx(isolated.sources[0].peak_snr)
    assert bright.flux == pytest.approx(isolated.sources[0].flux)
    assert bright.sersic_index == 4.0


def test_the_recipe_records_the_bright_component_once_not_as_a_ladder(
    blend_recipe,
):
    """It does not step across the grid, so it is not a ladder (#67)."""
    bright, _ = blend_pairs(blend_recipe)[0]

    assert blend_recipe.bright_peak_snr == pytest.approx(bright.peak_snr)
    assert blend_recipe.bright_magnitude == pytest.approx(bright.magnitude)
    assert build_isolated('sersic_n4').bright_magnitude is None


def test_the_ladders_describe_the_companion_because_the_columns_do(blend_recipe):
    """``magnitude_ladder`` is the companion's here, per the class docstring."""
    for _, companion in blend_pairs(blend_recipe):
        assert companion.magnitude == blend_recipe.magnitude_ladder[companion.column]
        assert companion.peak_snr == pytest.approx(
            blend_recipe.peak_snr_ladder[companion.column]
        )


# --- what the per-source table records -------------------------------------


def test_every_source_names_the_other_member_of_its_pair(blend_recipe):
    """The companion column of #67, and it points both ways.

    A one-way link would let a metric group the table into pairs one way and
    not the other, and nothing would report the disagreement.
    """
    by_id = {source.source_id: source for source in blend_recipe.sources}

    for source in blend_recipe.sources:
        assert source.companion_id in by_id
        assert by_id[source.companion_id].companion_id == source.source_id
        assert source.companion_id != source.source_id


def test_the_recorded_separation_is_the_distance_between_the_pair(blend_recipe):
    """Recorded in pixels, the unit ``x`` and ``y`` are in.

    Measured off the two positions rather than trusted, because the separation
    axis means nothing if the number in the table and the number in the pixels
    are allowed to drift apart.
    """
    for bright, companion in blend_pairs(blend_recipe):
        measured = np.hypot(companion.x - bright.x, companion.y - bright.y)

        assert measured == pytest.approx(bright.separation)
        assert companion.separation == bright.separation


def test_the_recorded_flux_ratio_is_the_pair_s_flux_ratio(blend_recipe):
    """Faint over bright, and the same number on both rows of the pair."""
    for bright, companion in blend_pairs(blend_recipe):
        assert companion.flux / bright.flux == pytest.approx(bright.flux_ratio)
        assert companion.flux_ratio == bright.flux_ratio
        assert 0.0 < bright.flux_ratio < 1.0


def test_an_isolated_class_records_no_pair_at_all(isolated_recipe):
    """The three blend columns are an absence there, not a zero measurement.

    All three say it the same way. A ``None`` companion beside a separation of
    zero would encode one absence twice over, and the zero is the half a reader
    can mistake for a measurement — which is the same reason the write layer
    leaves the columns out of an isolated class's table entirely.
    """
    for source in isolated_recipe.sources:
        assert source.companion_id is None
        assert source.separation is None
        assert source.flux_ratio is None


# --- the ends of the two ladders ------------------------------------------


def test_the_separation_ladder_runs_from_the_psf_to_the_bright_wings(
    blend_recipe,
):
    """Both ends measured rather than chosen (#71).

    The tight end is one resolution element — the pair is a single blob and no
    deblender can win. The wide end is the radius at which the bright source
    finally drops under the background, which is the boundary the class exists
    to locate: the widest row lands on it. What lies outside it is the isolated
    ``sersic_n4`` entry, the same bright source with no companion at all.
    """
    psf_fwhm_pixels = recipe_module.PSF_FWHM / recipe_module.PIXEL_SCALE

    assert blend_recipe.separation_ladder[0] == pytest.approx(
        recipe_module.BLEND_SEPARATION_MIN_FWHM * psf_fwhm_pixels
    )
    assert blend_recipe.separation_ladder[-1] == pytest.approx(
        blend_recipe.wing_radius
    )


def test_the_companion_spans_detectable_down_to_marginal(blend_recipe):
    """The flux-ratio axis is decided in peak SNR, and this is its range.

    An order of magnitude below the bright component at one end — which is what
    makes the pair a bright-plus-faint one — down to the same marginally
    detectable rung the isolated ladders end on.
    """
    assert blend_recipe.peak_snr_ladder[0] == pytest.approx(
        recipe_module.BLEND_COMPANION_PEAK_SNR_BRIGHTEST, rel=0.05
    )
    assert blend_recipe.peak_snr_ladder[-1] == pytest.approx(
        recipe_module.PEAK_SNR_FAINTEST, rel=0.05
    )


def test_the_companion_is_a_point_source_and_the_bright_one_a_sersic(
    blend_recipe,
):
    """A measurement decision: the companion's whole flux is inside one PSF.

    That is what keeps "how much of the companion survived" a statement about
    the hole even at the tightest separation, where an extended companion's own
    wings would be inside the bright source and no measurement could separate
    light lost to over-subtraction from light that was never separable.
    """
    for bright, companion in blend_pairs(blend_recipe):
        assert companion.is_point
        assert companion.half_light_radius == 0.0
        assert companion.truncation_radius == 0.0
        assert not bright.is_point
        assert bright.truncation_radius > 0.0


# --- the refusals ---------------------------------------------------------


def test_neighbouring_pairs_do_not_reach_each_other(blend_recipe):
    """A whole pair fits inside its own cell, wings and separation together.

    This is the refusal the class needs of its own: the isolated one asks
    whether a *source* reaches its neighbour's cell, and at the widest
    separation a pair reaches half as far again as its bright component does.
    """
    reach = blend_recipe.wing_radius + blend_recipe.separation_ladder[-1]

    assert reach < blend_recipe.grid_spacing


def test_neighbouring_pairs_stay_clear_once_the_jitter_has_moved_them(
    blend_recipe,
):
    """Two cells jittered towards each other are still two separate pairs.

    This is the comparison :func:`check_pair_spacing` itself makes, restated
    here on a built entry so that the property is pinned against the numbers
    the class actually lays out and not only against the refusal's arithmetic.
    """
    closest = blend_recipe.grid_spacing - 2.0 * blend_recipe.jitter_amplitude
    reach = blend_recipe.wing_radius + blend_recipe.separation_ladder[-1]

    assert reach < closest


def test_a_pair_wider_than_its_cell_is_refused():
    recipe_module.check_pair_spacing(
        wing_radius=40.0, separation=50.0, grid_spacing=100.0, jitter_amplitude=0.0
    )

    with pytest.raises(recipe_module.PairsTooCloseError):
        recipe_module.check_pair_spacing(
            wing_radius=40.0,
            separation=70.0,
            grid_spacing=100.0,
            jitter_amplitude=0.0,
        )


def test_the_pair_refusal_counts_the_jitter_against_the_spacing():
    """A pair that fits the nominal spacing and not the jittered one is refused.

    This is where :func:`check_pair_spacing` parts company with
    :func:`check_grid_spacing`, which #67 states against the nominal spacing. A
    pair reaches half as far again as its bright component does, so the
    twenty-four pixels two cells can close by are a real part of the margin
    rather than slack around it.
    """
    fits = dict(wing_radius=40.0, separation=50.0, grid_spacing=100.0)

    recipe_module.check_pair_spacing(**fits, jitter_amplitude=0.0)

    with pytest.raises(recipe_module.PairsTooCloseError):
        recipe_module.check_pair_spacing(**fits, jitter_amplitude=12.0)


def test_the_pair_refusal_catches_what_the_isolated_one_lets_through(
    monkeypatch,
):
    """A grid roomy enough for the bright source alone, but not for the pair.

    Four grid steps on a 512-pixel image leave 96 pixels between cells. The
    bright component's wings end at about 52, so :func:`check_grid_spacing` is
    satisfied — and the widest pair on the ladder reaches about 103, so it is
    not an isolated field at all. Without a refusal of its own the entry would
    build, and every separation in the map would be measured across a boundary
    that another pair was already sitting on.
    """
    monkeypatch.setitem(
        recipe_module.TIERS,
        'cramped',
        recipe_module.Tier(image_size=512, grid_side=4),
    )

    with pytest.raises(recipe_module.PairsTooCloseError):
        build_blend(tier='cramped')


def test_the_blend_entry_keeps_every_source_clear_of_the_image_edge(blend_recipe):
    """Including the companion, which is displaced outward from the cell centre.

    The grid is laid out on the bright component's margins; nothing about that
    layout knows the companion is pushed up to a whole separation further out,
    so this refusal is what says it still clears the border.
    """
    recipe_module.check_image_margin(
        blend_recipe.sources,
        blend_recipe.image_shape[0],
        margin=recipe_module.PSF_MARGIN_FWHM
        * blend_recipe.psf_fwhm
        / blend_recipe.pixel_scale,
        pixel_scale=blend_recipe.pixel_scale,
    )


# --- independence and reproducibility -------------------------------------


def test_no_two_pairs_share_an_orientation(blend_recipe):
    """The companion's position angle is drawn, not fixed.

    Neither grid axis is free to buy independent samples the way the isolated
    classes' jitter row does, so the pair centre's jitter and this angle are
    what stop twenty-five cells from being twenty-five copies of one geometry.
    """
    angles = {
        round(float(np.arctan2(companion.y - bright.y, companion.x - bright.x)), 9)
        for bright, companion in blend_pairs(blend_recipe)
    }

    assert len(angles) == len(blend_pairs(blend_recipe))


def test_the_blend_entry_is_reproducible(blend_recipe):
    """Same axes, same layout — the jitter and the angle are seeded, not random."""
    again = build_blend(tier=blend_recipe.tier)

    assert again.sources == blend_recipe.sources
    assert again.separation_ladder == blend_recipe.separation_ladder
    assert again.flux_ratio_ladder == blend_recipe.flux_ratio_ladder


# --- the manifest ---------------------------------------------------------


def test_the_blend_manifest_records_its_own_two_axes(blend_recipe):
    manifest = blend_recipe.manifest()

    assert manifest['separation_ladder'] == list(blend_recipe.separation_ladder)
    assert manifest['flux_ratio_ladder'] == list(blend_recipe.flux_ratio_ladder)
    assert manifest['bright_magnitude'] == pytest.approx(
        blend_recipe.bright_magnitude
    )
    assert manifest['bright_peak_snr'] == pytest.approx(blend_recipe.bright_peak_snr)
    assert manifest['n_pairs'] == len(blend_recipe.sources) // 2
    assert json.loads(json.dumps(manifest))['class'] == 'blend'


def test_the_blend_companions_are_bins_of_their_own(blend_recipe):
    """Every companion column is weighed on its own, and the bright one too.

    This is the class the binned check exists for. All the companions together
    carry well under a hundredth of the entry, so a whole-image check at a one
    per cent tolerance could not tell a rendered companion from a missing one —
    and a companion the hole did not eat is the entry's whole point.
    """
    columns, _ = blend_recipe.grid_shape
    companions = [companion for _, companion in blend_pairs(blend_recipe)]
    share = sum(source.flux for source in companions) / blend_recipe.total_flux

    assert share < FLUX_TOLERANCE
    assert len(blend_recipe.flux_groups) == columns + 1
    assert len(blend_recipe.flux_groups[0]) == len(blend_pairs(blend_recipe))


def test_the_blend_manifest_says_whose_ladders_it_carries(blend_recipe):
    """``magnitude_ladder`` changes subject with the class, so it is labelled.

    For the isolated classes it is the source's own brightness; here the column
    axis is the flux ratio, so it is the companion's. A consumer reading the
    two uniformly would be comparing different quantities, and the key is what
    stops that being something it has to infer.
    """
    assert blend_recipe.manifest()['ladders_describe'] == 'companion'
    assert 'ladders_describe' not in build_isolated('sersic_n4').manifest()


def test_the_blend_keys_are_absent_from_every_other_manifest(tracer_recipe):
    """So no entry already on disk is made stale by a class it is not (#71).

    The comparison behind the generator's skip is exact. Adding these keys to
    every class would have made every entry already generated fail it, and a
    rebuild of six entries whose pixels this ticket does not move is the same
    waste ``PROVENANCE_KEYS`` exists to prevent for an upgrade.
    """
    manifest = tracer_recipe.manifest()

    assert 'separation_ladder' not in manifest
    assert 'flux_ratio_ladder' not in manifest
    assert 'bright_magnitude' not in manifest
    assert tracer_recipe.matches_manifest(manifest)
