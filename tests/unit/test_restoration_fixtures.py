"""The fixture layer of the restoration test battery (#75).

Two seams, agreed before a line was written.

The **pure layer** — matching, the settings record, the fixture store — is
where the decisions of #75 live, and it imports neither DAWIS's pipeline nor
matplotlib. It is tested the way ``simulations/recipe.py`` is: every decision
asserted directly, in seconds.

The **mock iteration** is tested end to end, on a small synthetic image, for
one reason worth stating. DAWIS indexes a tree by ``(row, column)`` — a
region's ``x_max`` is a numpy row — while ``sources.csv`` records ``x`` as a
column and ``y`` as a row. The two conventions are transposes of each other,
and a swapped axis matches the wrong source without raising anything. Only a
run that goes from pixels to a match can catch it.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

# ``restoration_tests`` is deliberately not part of the distributed package
# (see ``test_package_surface.py``), so it is not importable off the install.
# It is a sibling of ``dawis`` in the source tree, and that is where it is
# found -- the same prologue ``test_simulation_recipe.py`` carries for
# ``simulations``, and for the same reason.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from restoration_tests.fixtures.build import (  # noqa: E402
    already_built,
    build_fixtures,
    selects,
    sweep,
)
from restoration_tests.fixtures.entry import GalleryEntry  # noqa: E402
from restoration_tests.fixtures.mock import (  # noqa: E402
    FROZEN_SETTINGS,
    run_mock_iteration,
)
from restoration_tests.fixtures.store import (  # noqa: E402
    Fixture,
    TreeGeometry,
    read_fixtures,
    read_manifest,
    write_fixtures,
)
from restoration_tests.fixtures.matching import (  # noqa: E402
    AMBIGUOUS,
    MATCHED,
    UNMATCHED,
    GallerySource,
    match_radius_pixels,
    match_tree,
)


def _write_entry(tmp_path, tier = 'small', entry_class = 'sersic_n4',
                 sources = None, image_shape = (16, 16)):
    """A minimal simulation-gallery entry on disk, shaped like a real one.

    Only the files and manifest keys the fixture layer reads. The pixels are
    a ramp rather than a rendered source: nothing here asserts anything about
    them, and a test that drew a real profile would be a second renderer.
    """
    from astropy.io import fits

    directory = tmp_path / tier / ('%s_balanced_r0' % entry_class)
    directory.mkdir(parents = True)

    ramp = np.arange(image_shape[0] * image_shape[1], dtype = float)
    ramp = ramp.reshape(image_shape)
    fits.writeto(directory / 'image.fits', ramp)
    fits.writeto(directory / 'truth_conv.fits', ramp / 2.0)

    if sources is None:
        sources = [(0, 8.0, 8.0, 18.13, 4211.6, 4.0, 0.30, 0.0, 0.0)]

    columns = ('source_id,x,y,magnitude,flux,sersic_index,'
               'half_light_radius,e1,e2')
    rows = '\n'.join(','.join(repr(value) for value in source)
                     for source in sources)
    (directory / 'sources.csv').write_text('%s\n%s\n' % (columns, rows))

    (directory / 'manifest.json').write_text(json.dumps({
        'tier': tier,
        'class': entry_class,
        'regime': 'balanced',
        'realisation': 0,
        'pixel_scale': 0.03,
        'psf_fwhm': 0.09,
        'image_shape': list(image_shape),
        'source_commit': 'abcdef0',
    }))

    return directory


def _source(source_id: int = 0, x: float = 0.0, y: float = 0.0,
            half_light_radius: float = 0.30) -> GallerySource:
    """A gallery source with only the fields a match reads set to anything."""
    return GallerySource(source_id = source_id, x = x, y = y,
                         magnitude = 20.0, flux = 1.0, sersic_index = 4.0,
                         half_light_radius = half_light_radius,
                         e1 = 0.0, e2 = 0.0)


class TestMatchRadius:
    """What counts as "inside the source", in pixels."""

    def test_an_extended_source_is_matched_within_its_half_light_radius(self):
        # 0.30 arcsec at 0.03 arcsec per pixel is 10 pixels, and that is
        # wider than the PSF, so the profile sets the radius.
        radius = match_radius_pixels(_source(half_light_radius = 0.30),
                                     pixel_scale = 0.03, psf_fwhm = 0.09)

        assert radius == pytest.approx(10.0)

    def test_a_point_source_is_matched_within_the_psf(self):
        # A delta function's half-light radius is zero -- literally true, and
        # a radius no interscale maximum could ever fall inside. The light
        # that reaches the image is the PSF's, so the PSF sets the floor.
        radius = match_radius_pixels(_source(half_light_radius = 0.0),
                                     pixel_scale = 0.03, psf_fwhm = 0.09)

        assert radius == pytest.approx(3.0)


class TestMatchTree:
    """Which source a tree's interscale maximum belongs to."""

    def test_the_source_the_maximum_falls_on_is_the_match(self):
        # sources.csv measures x along the columns and y down the rows; the
        # interscale maximum is a numpy (row, column). The source below sits
        # at column 100, row 250, so a maximum at row 250, column 100 is on
        # it -- and a match that read the axes the other way round would look
        # at row 100, column 250 and find nothing.
        source = _source(source_id = 7, x = 100.4, y = 250.6)

        match = match_tree(peak_row = 250, peak_col = 100, sources = [source],
                           pixel_scale = 0.03, psf_fwhm = 0.09)

        assert match.outcome == MATCHED
        assert match.source is source

    def test_a_maximum_outside_every_radius_matches_nothing(self):
        source = _source(x = 100.0, y = 250.0)

        match = match_tree(peak_row = 250, peak_col = 130, sources = [source],
                           pixel_scale = 0.03, psf_fwhm = 0.09)

        assert match.outcome == UNMATCHED
        assert match.source is None

    def test_a_maximum_inside_two_sources_matches_neither(self):
        # Two sources close enough to share a maximum is a blend by another
        # name, and #75 scores single sources. Refusing it keeps a fixture
        # from carrying a truth image that is really two objects.
        near = _source(source_id = 1, x = 100.0, y = 250.0)
        also_near = _source(source_id = 2, x = 104.0, y = 250.0)

        match = match_tree(peak_row = 250, peak_col = 102,
                           sources = [near, also_near],
                           pixel_scale = 0.03, psf_fwhm = 0.09)

        assert match.outcome == AMBIGUOUS
        assert match.source is None


class TestGalleryEntry:
    """Reading one simulation-gallery entry off disk."""

    def test_it_reads_the_sources_and_the_two_numbers_matching_needs(self, tmp_path):
        entry_dir = _write_entry(tmp_path, sources = [
            (0, 100.4, 250.6, 18.13, 4211.6, 4.0, 0.30, 0.01, -0.02),
            (1, 244.0, 250.0, 21.29, 231.0, 4.0, 0.30, 0.00, 0.00),
        ])

        entry = GalleryEntry.read(entry_dir)

        assert entry.pixel_scale == pytest.approx(0.03)
        assert entry.psf_fwhm == pytest.approx(0.09)
        assert [source.source_id for source in entry.sources] == [0, 1]
        assert entry.sources[0].x == pytest.approx(100.4)
        assert entry.sources[0].y == pytest.approx(250.6)
        assert entry.sources[0].magnitude == pytest.approx(18.13)

    def test_it_names_itself_by_tier_and_entry_directory(self, tmp_path):
        entry = GalleryEntry.read(_write_entry(tmp_path))

        assert entry.name == 'small/sersic_n4_balanced_r0'

    def test_the_image_and_the_truth_come_back_as_arrays(self, tmp_path):
        entry = GalleryEntry.read(_write_entry(tmp_path))

        assert entry.image.shape == (16, 16)
        assert entry.truth_conv.shape == (16, 16)

    def test_a_blend_entry_is_refused(self, tmp_path):
        # #75 scores single sources; #74 leaves blends to a later pass. The
        # refusal is here rather than in the sweep so that pointing the
        # builder straight at a blend directory fails the same way.
        entry_dir = _write_entry(tmp_path, entry_class = 'blend')

        with pytest.raises(ValueError, match = 'blend'):
            GalleryEntry.read(entry_dir)


def _two_blobs(shape = (128, 128), seed = 7):
    """Two Gaussian blobs at positions no transpose could confuse.

    Both sit off the diagonal and off each other's row and column, so an
    image read with its axes swapped puts every blob somewhere no blob is.
    """
    rows, columns = np.indices(shape).astype(float)
    image = np.zeros(shape)

    for row, column, amplitude, width in ((30.0, 90.0, 400.0, 3.0),
                                          (95.0, 40.0, 250.0, 4.0)):
        image += amplitude * np.exp(
            -((rows - row) ** 2 + (columns - column) ** 2) / (2.0 * width ** 2))

    rng = np.random.RandomState(seed)
    return image + rng.normal(loc = 10.0, scale = 1.0, size = shape)


class TestMockSettings:
    """The settings every fixture is dumped under, frozen (#75)."""

    def test_the_frozen_settings_are_the_ones_the_ticket_names(self):
        assert FROZEN_SETTINGS.n_levels == 8
        assert FROZEN_SETTINGS.min_span == 2
        assert FROZEN_SETTINGS.max_span == 8
        assert FROZEN_SETTINGS.ceps == pytest.approx(1e-3)

    def test_tau_is_the_value_the_measured_tree_counts_chose(self):
        # #75 wrote tau=0.1. At that value one mock iteration on a small-tier
        # entry returns three trees and all three sit on the brightest
        # magnitude column, so the fixture set would carry no brightness
        # spread at all and #25's brightness-binned residuals would have one
        # bin. 0.01 adds the middle column. Amael's call, this session.
        assert FROZEN_SETTINGS.tau == pytest.approx(0.01)

    def test_the_settings_record_themselves_for_a_manifest(self):
        recorded = FROZEN_SETTINGS.as_dict()

        assert recorded['tau'] == pytest.approx(0.01)
        assert recorded['n_levels'] == 8


class TestMockIteration:
    """One pass: image, à trous, detection, interscale trees. No loop."""

    def test_it_finds_a_tree_on_each_blob_at_the_right_row_and_column(self):
        # The end-to-end seam. A tree's interscale maximum is (x_max, y_max)
        # and those are a numpy row and column; reading them as x and y puts
        # the first blob's peak at row 90, column 30, where the image is
        # noise. Asserting the pair catches the swap.
        settings = replace(FROZEN_SETTINGS, n_levels = 5)

        iteration = run_mock_iteration(_two_blobs(), settings)

        peaks = sorted((int(tree.interscale_maximum.x_max),
                        int(tree.interscale_maximum.y_max))
                       for tree in iteration.trees)

        assert len(peaks) == 2
        assert peaks[0] == pytest.approx((30, 90), abs = 3)
        assert peaks[1] == pytest.approx((95, 40), abs = 3)

    def test_it_hands_back_the_cubes_the_trees_were_built_on(self):
        # A tree cuts its tubes out of the cubes rather than carrying them,
        # so a caller that has trees and no cubes has nothing to dump.
        settings = replace(FROZEN_SETTINGS, n_levels = 5)

        iteration = run_mock_iteration(_two_blobs(), settings)

        tubes = iteration.trees[0].tubes(iteration.wavelet_cube,
                                         iteration.label_cube)
        assert all(tube.ndim == 3 for tube in tubes)

    def test_the_same_image_and_settings_give_the_same_trees(self):
        # The Anscombe fit and the inpainting both draw. #72 froze the
        # gallery's seeds so every arm sees the same bytes; the same argument
        # applies one layer up, or a re-dump would silently move every tube.
        settings = replace(FROZEN_SETTINGS, n_levels = 5)
        image = _two_blobs()

        first = run_mock_iteration(image, settings)
        second = run_mock_iteration(image, settings)

        assert np.array_equal(first.wavelet_cube.array, second.wavelet_cube.array)
        assert np.array_equal(first.label_cube.array, second.label_cube.array)


def _fixture(fixture_id = 'small/sersic_n4_balanced_r0/tree000',
             tube_shape = (6, 5, 4)):
    """One fixture with distinguishable arrays in every slot."""
    wavelet_tube = np.arange(np.prod(tube_shape), dtype = float)
    wavelet_tube = wavelet_tube.reshape(tube_shape)

    return Fixture(
        fixture_id = fixture_id,
        entry = 'small/sersic_n4_balanced_r0',
        source = _source(source_id = 3, x = 100.4, y = 250.6),
        geometry = TreeGeometry(peak_row = 250, peak_col = 100, level = 3,
                                bbox = (247, 97, 253, 102), extent = 0.42,
                                eccentricity = 0.61),
        wavelet_tube = wavelet_tube,
        support_tube = (wavelet_tube % 3 == 0).astype(float),
        label_tube = (wavelet_tube % 7).astype(float),
        truth_cutout = wavelet_tube[:, :, 0] * 2.0,
        reference_reconstruction = wavelet_tube[:, :, 0] * 3.0,
        reference_arm = {'filter': 'BSPL', 'synthesis_operator': 'ADJOINT',
                         'deconv': False, 'norm_wr': 0.25,
                         'flag_convergence': True},
    )


class TestFixtureStore:
    """Writing a set of fixtures and reading them back."""

    def test_a_fixture_survives_a_round_trip_unchanged(self, tmp_path):
        original = _fixture()

        write_fixtures(tmp_path, [original],
                       entry_name = 'small/sersic_n4_balanced_r0',
                       settings = FROZEN_SETTINGS,
                       entry_manifest = {'class': 'sersic_n4'},
                       match_counts = {MATCHED: 1})
        restored, = read_fixtures(tmp_path)

        assert restored.fixture_id == original.fixture_id
        assert restored.entry == original.entry
        assert restored.source == original.source
        assert restored.geometry == original.geometry
        assert restored.reference_arm == original.reference_arm
        for name in ('wavelet_tube', 'support_tube', 'label_tube',
                     'truth_cutout', 'reference_reconstruction'):
            assert np.array_equal(getattr(restored, name),
                                  getattr(original, name)), name

    def test_fixtures_come_back_in_the_order_they_went_in(self, tmp_path):
        # A set is indexed by position elsewhere -- a quicklook filename, a
        # row of a results table -- so h5py's alphabetical group order must
        # not be what decides it. Ten is enough for 'tree10' to sort before
        # 'tree2'.
        written = [_fixture(fixture_id = 'e/tree%d' % index)
                   for index in range(12)]

        write_fixtures(tmp_path, written, entry_name = 'e',
                       settings = FROZEN_SETTINGS,
                       entry_manifest = {}, match_counts = {})

        assert [fixture.fixture_id for fixture in read_fixtures(tmp_path)] \
            == [fixture.fixture_id for fixture in written]

    def test_the_manifest_records_what_a_reader_cannot_reconstruct(self, tmp_path):
        write_fixtures(tmp_path, [_fixture()],
                       entry_name = 'small/sersic_n4_balanced_r0',
                       settings = FROZEN_SETTINGS,
                       entry_manifest = {'class': 'sersic_n4',
                                         'source_commit': 'abcdef0'},
                       match_counts = {MATCHED: 1, UNMATCHED: 2})

        manifest = read_manifest(tmp_path)

        assert manifest['entry'] == 'small/sersic_n4_balanced_r0'
        assert manifest['settings']['tau'] == pytest.approx(0.01)
        assert manifest['gallery_entry']['source_commit'] == 'abcdef0'
        assert manifest['match_counts'] == {MATCHED: 1, UNMATCHED: 2}
        assert manifest['n_fixtures'] == 1
        assert manifest['dtypes']['wavelet_tube'] == 'float64'
        assert manifest['dawis_commit']

    def test_an_empty_set_writes_a_manifest_and_no_fixtures(self, tmp_path):
        # An entry whose mock iteration matched nothing is a result, not a
        # failure -- and the sweep has to be able to record it, or a rerun
        # would redo it every time.
        write_fixtures(tmp_path, [], entry_name = 'small/point_photon_r2',
                       settings = FROZEN_SETTINGS,
                       entry_manifest = {}, match_counts = {UNMATCHED: 3})

        assert read_fixtures(tmp_path) == []
        assert read_manifest(tmp_path)['n_fixtures'] == 0
        # The entry still names itself. A set that matched nothing is the one
        # a reader most needs to be able to identify, and taking the name off
        # the first fixture leaves it null in exactly that case.
        assert read_manifest(tmp_path)['entry'] == 'small/point_photon_r2'


def _blob_entry(sources = None, shape = (128, 128)):
    """A gallery entry whose image is :func:`_two_blobs`.

    Built in memory rather than read off disk: what the build step is being
    asked is whether it matches, cuts and numbers correctly, and a FITS round
    trip is not part of that question. The sources sit on the blobs, with
    ``x`` on the column and ``y`` on the row — the same transposed pair the
    matching test uses, so a swap anywhere in the chain shows up here too.
    """
    if sources is None:
        sources = [_source(source_id = 0, x = 90.0, y = 30.0),
                   _source(source_id = 1, x = 40.0, y = 95.0)]

    image = _two_blobs(shape)
    return GalleryEntry(
        name = 'small/sersic_n4_balanced_r0',
        path = Path('/nowhere'),
        image = image,
        truth_conv = np.ones(shape) * 4.0,
        sources = sources,
        pixel_scale = 0.03,
        psf_fwhm = 0.09,
        manifest = {'class': 'sersic_n4', 'source_commit': 'abcdef0'},
    )


class TestBuildFixtures:
    """Turning one entry and its mock iteration into fixtures."""

    def test_each_matched_tree_becomes_one_fixture_on_its_own_source(self):
        entry = _blob_entry()
        settings = replace(FROZEN_SETTINGS, n_levels = 5)

        fixtures, counts = build_fixtures(
            entry, run_mock_iteration(entry.image, settings), settings)

        assert counts[MATCHED] == 2
        by_source = {fixture.source.source_id: fixture for fixture in fixtures}
        assert set(by_source) == {0, 1}
        assert by_source[0].geometry.peak_row == pytest.approx(30, abs = 3)
        assert by_source[0].geometry.peak_col == pytest.approx(90, abs = 3)

    def test_the_truth_cutout_and_the_tubes_cover_the_same_box(self):
        entry = _blob_entry()
        settings = replace(FROZEN_SETTINGS, n_levels = 5)

        fixtures, _ = build_fixtures(
            entry, run_mock_iteration(entry.image, settings), settings)
        fixture = fixtures[0]

        rows, columns, _ = fixture.wavelet_tube.shape
        assert fixture.truth_cutout.shape == (rows, columns)
        assert fixture.reference_reconstruction.shape == (rows, columns)

        row_min, col_min, row_max, col_max = fixture.geometry.bbox
        assert np.array_equal(
            fixture.truth_cutout,
            entry.truth_conv[row_min:row_max, col_min:col_max])

    def test_an_unmatched_tree_is_dropped_and_counted(self):
        # Sources moved off the blobs. Detection still finds the blobs, so
        # the trees exist; nothing they can be scored against does.
        entry = _blob_entry(sources = [_source(source_id = 0, x = 5.0, y = 5.0)])
        settings = replace(FROZEN_SETTINGS, n_levels = 5)

        fixtures, counts = build_fixtures(
            entry, run_mock_iteration(entry.image, settings), settings)

        assert fixtures == []
        assert counts[UNMATCHED] == 2
        assert counts[MATCHED] == 0

    def test_fixture_ids_number_the_entry_and_survive_the_store(self, tmp_path):
        entry = _blob_entry()
        settings = replace(FROZEN_SETTINGS, n_levels = 5)

        fixtures, counts = build_fixtures(
            entry, run_mock_iteration(entry.image, settings), settings)
        write_fixtures(tmp_path, fixtures, entry_name = entry.name,
                       settings = settings,
                       entry_manifest = entry.manifest, match_counts = counts)

        assert [fixture.fixture_id for fixture in read_fixtures(tmp_path)] == [
            'small/sersic_n4_balanced_r0/tree000',
            'small/sersic_n4_balanced_r0/tree001',
        ]

    def test_the_reference_arm_records_what_the_published_default_chose(self):
        entry = _blob_entry()
        settings = replace(FROZEN_SETTINGS, n_levels = 5)

        fixtures, _ = build_fixtures(
            entry, run_mock_iteration(entry.image, settings), settings)
        arm = fixtures[0].reference_arm

        assert arm['filter'] in ('BSPL', 'HAAR')
        assert arm['synthesis_operator'] in ('ADJOINT', 'SUM')
        assert arm['deconv'] is False
        assert arm['flag_convergence'] is True
        assert arm['norm_wr'] >= 0.0


class TestQuicklook:
    """The panel a human eyeballs before a metric exists."""

    def test_it_draws_one_panel_per_fixture_named_by_its_tree(self, tmp_path):
        pytest.importorskip('matplotlib')
        from restoration_tests.fixtures.quicklook import draw_quicklooks

        fixtures = [_fixture(fixture_id = 'small/e/tree000'),
                    _fixture(fixture_id = 'small/e/tree001')]

        written = draw_quicklooks(fixtures, tmp_path)

        assert [path.name for path in written] == ['tree000.png', 'tree001.png']
        assert all(path.stat().st_size > 0 for path in written)

    def test_truth_and_reconstruction_share_a_stretch(self, tmp_path):
        # The panel exists to be compared across, so the two images it
        # compares cannot be scaled independently -- a reconstruction at half
        # the truth's flux would look identical to it. The residual is scaled
        # on its own, symmetrically about zero, because what it shows is sign.
        pytest.importorskip('matplotlib')
        from restoration_tests.fixtures.quicklook import panel_limits

        fixture = _fixture()

        shared, residual = panel_limits(fixture)

        assert shared == (pytest.approx(min(fixture.truth_cutout.min(),
                                            fixture.reference_reconstruction.min())),
                          pytest.approx(max(fixture.truth_cutout.max(),
                                            fixture.reference_reconstruction.max())))
        assert residual[0] == pytest.approx(-residual[1])


class TestEntrySelection:
    """Which gallery entries a sweep picks up."""

    def _manifest(self, tier = 'small', entry_class = 'sersic_n4',
                  regime = 'balanced', realisation = 0):
        return {'tier': tier, 'class': entry_class, 'regime': regime,
                'realisation': realisation}

    def test_no_axis_given_selects_everything_but_blends(self):
        assert selects(self._manifest())
        assert not selects(self._manifest(entry_class = 'blend'))

    def test_an_axis_given_is_held_fixed(self):
        assert selects(self._manifest(regime = 'photon'), regime = 'photon')
        assert not selects(self._manifest(regime = 'photon'), regime = 'balanced')
        assert selects(self._manifest(realisation = 2), realisation = 2)
        assert not selects(self._manifest(realisation = 2), realisation = 0)

    def test_a_regime_whose_name_holds_an_underscore_is_read_correctly(self):
        # The defect the first real sweep hit. The entry directory is
        # <class>_<regime>_r<N>, and splitting it on underscores makes
        # 'blend_read_noise_r0' a 'blend_read' of regime 'noise' -- so the
        # blend skip missed it and the run stopped on a blend entry. The
        # manifest names all three axes outright, so nothing parses the
        # directory name any more.
        assert not selects(self._manifest(entry_class = 'blend',
                                          regime = 'read_noise'))
        assert selects(self._manifest(entry_class = 'point',
                                      regime = 'read_noise'),
                       regime = 'read_noise')

    def test_a_sweep_of_a_gallery_of_blends_writes_nothing(self, tmp_path):
        _write_entry(tmp_path, entry_class = 'blend')

        assert sweep(tmp_path, tmp_path / 'out') == []


class TestMatchTreePixelCentres:
    """The half-pixel convention between a source and a peak pixel."""

    # sources.csv measures from the lower-left *corner* of the first pixel,
    # so pixel index c holds coordinates [c, c + 1) and its centre is at
    # c + 0.5. Verified against the gallery: for every point source,
    # floor(x) is the column of the one non-zero pixel of truth_preconv.
    # Comparing an index against a coordinate directly is therefore a fixed
    # half-pixel bias on each axis -- 0.71 px radially, against a point
    # source's 3 px radius.

    def test_a_maximum_just_outside_the_radius_by_the_half_pixel_is_refused(self):
        # Source centre 100.0, peak pixel 103, whose centre is 103.5. The
        # true distance is 3.5, outside a point source's 3 px radius;
        # comparing 103 against 100.0 makes it exactly 3 and lets it in.
        source = _source(x = 100.0, y = 250.0, half_light_radius = 0.0)

        match = match_tree(peak_row = 250, peak_col = 103, sources = [source],
                           pixel_scale = 0.03, psf_fwhm = 0.09)

        assert match.outcome == UNMATCHED

    def test_a_maximum_just_inside_the_radius_by_the_half_pixel_is_kept(self):
        # The same slip in the other direction. Source centre 100.4, peak
        # pixel 97 whose centre is 97.5: the true distance is 2.9 and the
        # tree belongs to the source, but 97 against 100.4 reads 3.4 and
        # drops it.
        source = _source(x = 100.4, y = 250.5, half_light_radius = 0.0)

        match = match_tree(peak_row = 250, peak_col = 97, sources = [source],
                           pixel_scale = 0.03, psf_fwhm = 0.09)

        assert match.outcome == MATCHED


class TestGalleryEntryShapes:

    def test_a_truth_image_of_a_different_shape_is_refused(self, tmp_path):
        # The tree's bbox is clamped to the cube built from image.fits and
        # then used to slice truth_conv. A truth image of another shape gives
        # a cutout smaller than the tubes, and the mismatch surfaces much
        # later inside a metric -- or does not surface, if it broadcasts.
        from astropy.io import fits

        entry_dir = _write_entry(tmp_path)
        (entry_dir / 'truth_conv.fits').unlink()
        fits.writeto(entry_dir / 'truth_conv.fits', np.zeros((8, 8)))

        with pytest.raises(ValueError, match = 'shape'):
            GalleryEntry.read(entry_dir)


class TestDumpsUnderDifferentSettings:

    def test_an_entry_built_under_other_settings_is_not_skipped(self, tmp_path):
        # The resume test was 'a manifest exists', and the output path says
        # nothing about the settings. So a second sweep at another tau
        # skipped every entry and reported success, leaving the first set on
        # disk under a manifest that named the first tau.
        out = tmp_path / 'out' / 'small' / 'sersic_n4_balanced_r0'
        write_fixtures(out, [], entry_name = 'small/sersic_n4_balanced_r0',
                       settings = FROZEN_SETTINGS, entry_manifest = {},
                       match_counts = {})

        assert already_built(out, FROZEN_SETTINGS)
        assert not already_built(out, replace(FROZEN_SETTINGS, tau = 0.1))

    def test_an_entry_never_built_is_not_skipped(self, tmp_path):
        assert not already_built(tmp_path / 'nothing-here', FROZEN_SETTINGS)


class TestStoreRecordsRatherThanAsserts:

    def test_the_dtype_table_is_read_off_the_arrays(self, tmp_path):
        # The manifest sells dtypes as what a reader cannot get out of the
        # HDF5 file. Hardcoding them makes it a claim rather than a record,
        # and it would lie the first time an arm stored a float32 tube.
        fixture = _fixture()
        fixture = replace(fixture,
                          label_tube = fixture.label_tube.astype(np.float32))

        write_fixtures(tmp_path, [fixture], entry_name = 'e',
                       settings = FROZEN_SETTINGS, entry_manifest = {},
                       match_counts = {})

        assert read_manifest(tmp_path)['dtypes'] == {
            'wavelet_tube': 'float64', 'support_tube': 'float64',
            'label_tube': 'float32', 'truth_cutout': 'float64',
            'reference_reconstruction': 'float64'}


class TestQuicklookIsDrawnBeforeTheEntryIsRecorded:

    def test_a_perfect_reconstruction_still_gets_a_usable_residual_scale(self):
        pytest.importorskip('matplotlib')
        from restoration_tests.fixtures.quicklook import panel_limits

        fixture = _fixture()
        fixture = replace(fixture,
                          reference_reconstruction = fixture.truth_cutout)

        _, residual = panel_limits(fixture)

        assert residual[0] < residual[1]
