"""Tier 0 — the generator's command-line surface (#69, spec #67).

#68 built one entry through all three layers. #69 turns that into a gallery
you can build in one command, interrupt, and run again: ``--all`` sweeps every
entry the recipe knows how to build, an entry already on disk is skipped, and
``--force`` rebuilds one deliberately.

Everything asserted here runs **without GalSim**, which is what makes it a
Tier 0 suite rather than a second pytest tier inside ``dawis-sim``. Two
choices in the generator buy that:

* the write layer no longer imports the render layer at run time, so the skip
  and the manifest comparison are reachable from the ordinary test
  environment;
* :func:`simulations.generate.build_entry` takes the renderer as an argument,
  so the resume and rebuild paths can be driven with a stub that produces
  arrays of the right shape and nothing else.

That stub is deliberately not a fake GalSim. Nothing here asserts anything
about pixels — #67 puts the check on the drawn flux in the generator itself,
where a human is present. What is asserted here is *which entries a run
touches*, which is exactly what a resumable generation gets wrong.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

# ``simulations`` is deliberately not part of the distributed package (see
# ``test_package_surface.py``), so it is not importable off the install. It is
# a sibling of ``dawis`` in the source tree, and that is where it is found.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from simulations import generate as generate_module  # noqa: E402
from simulations import recipe as recipe_module  # noqa: E402
from simulations.write import (  # noqa: E402
    BLEND_SOURCE_COLUMNS,
    SOURCE_COLUMNS,
    EntryState,
    entry_state,
    write_entry,
)

#: The cheap tier, so a test that writes real FITS files writes 512x512 ones.
TRACER = recipe_module.EntryAxes(
    tier='small', source_class='sersic_n4', regime='balanced', realisation=0
)


@dataclass
class StubRendered:
    """What the write layer reads off a rendered entry, and nothing more."""

    image: np.ndarray
    truth_preconv: np.ndarray
    truth_conv: np.ndarray
    psf: np.ndarray
    galsim_version: str = 'stub'


class StubRenderer:
    """A renderer that counts its calls, so a skip can be shown to be a skip."""

    def __init__(self):
        self.calls = 0

    def __call__(self, recipe):
        self.calls += 1
        blank = np.zeros(recipe.image_shape, dtype=np.float32)
        return StubRendered(
            image=blank,
            truth_preconv=blank,
            truth_conv=blank,
            psf=np.zeros((8, 8), dtype=np.float32),
        )


def describe(axes: recipe_module.EntryAxes):
    """One entry's recipe, from its axes — the spelling the generator uses."""
    return recipe_module.build_recipe(**asdict(axes))


@pytest.fixture(scope='module')
def tracer_recipe():
    """The entry the resume tests write and re-read, described once."""
    return describe(TRACER)


# --------------------------------------------------------------------------
# What --all sweeps
# --------------------------------------------------------------------------


def test_the_four_axes_of_the_spec_make_seventy_two_entries():
    """#67 states the tally outright, so the axes are pinned against it."""
    assert len(recipe_module.all_entries()) == 72
    assert len(set(recipe_module.all_entries())) == 72


def test_every_entry_all_builds_is_one_the_recipe_can_actually_build():
    """The single property that makes ``--all`` safe to run unattended.

    An entry that appears in the sweep but raises on the way to disk would
    turn one undecided axis into a failed generation, which is the opposite of
    resumable.
    """
    for axes in recipe_module.built_entries():
        describe(axes)


def test_the_sweep_is_the_refusals_complement_not_a_second_list():
    """Everything ``--all`` leaves out is left out because it is refused.

    ``built_entries`` asks the refusal rather than keeping its own list, so a
    ticket that decides an axis grows the sweep with no list to update. This
    is what asserts the two cannot drift apart.
    """
    built = set(recipe_module.built_entries())
    for axes in recipe_module.all_entries():
        refused = False
        try:
            describe(axes)
        except recipe_module.AxisNotBuiltError:
            refused = True
        assert (axes not in built) == refused


def test_the_sweep_holds_the_entry_the_tracer_ticket_built():
    assert TRACER in recipe_module.built_entries()


def test_the_sweep_is_now_the_whole_gallery():
    """Every axis value #67 names is swept (#72).

    #68 decided ``sersic_n4`` / ``balanced`` / ``r0`` and narrowed the rest;
    #70 added the other two isolated classes, #71 the ``blend`` layout, and #72
    the two remaining regimes and the two remaining realisations. There is
    nothing left for ``--all`` to leave out, and this is where that is stated
    once rather than inferred from a count.
    """
    swept = recipe_module.built_entries()

    assert set(swept) == set(recipe_module.all_entries())
    assert {axes.tier for axes in swept} == set(recipe_module.TIERS)
    assert {axes.source_class for axes in swept} == set(recipe_module.SOURCE_CLASSES)
    assert {axes.regime for axes in swept} == set(recipe_module.REGIMES)
    assert {axes.realisation for axes in swept} == set(recipe_module.REALISATIONS)


@pytest.mark.parametrize(
    'axis, value',
    [
        ('tier', 'small'),
        ('source_class', 'sersic_n4'),
        ('regime', 'balanced'),
        ('realisation', 0),
    ],
)
def test_an_axis_given_alongside_all_holds_that_axis_fixed(axis, value):
    selected = recipe_module.built_entries(**{axis: value})
    assert selected
    assert {getattr(axes, axis) for axes in selected} == {value}
    assert set(selected) <= set(recipe_module.built_entries())


def test_a_filter_matching_no_built_entry_selects_nothing():
    """Asking for a value off an axis with ``--all`` sweeps nothing, quietly.

    Since #72 every value #67 names is built, so the filter that matches
    nothing is one that names a value the gallery does not have. The generator
    turns that into an error message of its own; the recipe's job is only to
    report that there is nothing to build.
    """
    assert recipe_module.built_entries(realisation=3) == ()


# --------------------------------------------------------------------------
# What counts as already built
# --------------------------------------------------------------------------


def test_a_recipes_own_manifest_matches_it(tracer_recipe):
    assert tracer_recipe.matches_manifest(tracer_recipe.manifest())


def test_provenance_alone_does_not_make_a_manifest_stale(tracer_recipe):
    """A numpy or GalSim upgrade moves no ground truth by itself.

    Were it counted as a difference, the first run after any upgrade would
    silently rebuild all 72 entries, which is the cost #67 stores the gallery
    to avoid.
    """
    manifest = tracer_recipe.manifest(
        galsim_version='9.9.9', source_commit='deadbeef'
    )
    manifest['numpy_version'] = '0.0.0'
    assert tracer_recipe.matches_manifest(manifest)


@pytest.mark.parametrize('key', ['seed', 'grid_spacing', 'total_flux', 'regime'])
def test_a_manifest_describing_different_numbers_does_not_match(tracer_recipe, key):
    manifest = tracer_recipe.manifest()
    manifest[key] = 'changed' if isinstance(manifest[key], str) else 12345.0
    assert not tracer_recipe.matches_manifest(manifest)


def test_a_manifest_missing_a_key_does_not_match(tracer_recipe):
    manifest = tracer_recipe.manifest()
    del manifest['seed']
    assert not tracer_recipe.matches_manifest(manifest)


def test_a_changed_recipe_does_not_match_the_manifest_on_disk(tracer_recipe):
    """The case the comparison exists for: the recipe moved, the files did not."""
    moved = replace(tracer_recipe, seed=tracer_recipe.seed + 1)
    assert not moved.matches_manifest(tracer_recipe.manifest())


def test_an_entry_that_was_never_built_is_absent(tmp_path, tracer_recipe):
    assert entry_state(tracer_recipe, tmp_path) is EntryState.ABSENT


def test_an_entry_interrupted_before_its_manifest_is_absent(tmp_path, tracer_recipe):
    """#67 writes the manifest last so a half-built entry is visibly unfinished."""
    directory = tracer_recipe.entry_directory(tmp_path)
    directory.mkdir(parents=True)
    (directory / 'image.fits').write_bytes(b'partial')
    assert entry_state(tracer_recipe, tmp_path) is EntryState.ABSENT


def test_an_entry_written_by_the_write_layer_is_built(tmp_path, tracer_recipe):
    write_entry(tracer_recipe, StubRenderer()(tracer_recipe), tmp_path, commit='x')
    assert entry_state(tracer_recipe, tmp_path) is EntryState.BUILT


def test_an_entry_whose_manifest_describes_other_numbers_is_stale(
    tmp_path, tracer_recipe
):
    write_entry(tracer_recipe, StubRenderer()(tracer_recipe), tmp_path, commit='x')
    path = tracer_recipe.output_paths(tmp_path)[recipe_module.MANIFEST_FILE_NAME]
    manifest = json.loads(path.read_text())
    manifest['seed'] += 1
    path.write_text(json.dumps(manifest))
    assert entry_state(tracer_recipe, tmp_path) is EntryState.STALE


def test_an_unreadable_manifest_is_stale_not_built(tmp_path, tracer_recipe):
    directory = tracer_recipe.entry_directory(tmp_path)
    directory.mkdir(parents=True)
    (directory / recipe_module.MANIFEST_FILE_NAME).write_text('{ truncated')
    assert entry_state(tracer_recipe, tmp_path) is EntryState.STALE


# --------------------------------------------------------------------------
# Skipping and rebuilding
# --------------------------------------------------------------------------


def test_the_first_build_of_an_entry_renders_and_writes(tmp_path):
    renderer = StubRenderer()
    directory, built = generate_module.build_entry(
        TRACER, root=tmp_path, render=renderer
    )
    assert built
    assert renderer.calls == 1
    for name in recipe_module.ENTRY_FILE_NAMES:
        assert (directory / name).exists()


def test_building_an_entry_twice_skips_the_second_time(tmp_path, tracer_recipe):
    """The acceptance criterion an interrupted ``--all`` rests on."""
    renderer = StubRenderer()
    generate_module.build_entry(TRACER, root=tmp_path, render=renderer)
    directory, built = generate_module.build_entry(
        TRACER, root=tmp_path, render=renderer
    )
    assert not built
    assert renderer.calls == 1, 'the skip must happen before anything is drawn'
    assert directory == tracer_recipe.entry_directory(tmp_path)


def test_force_rebuilds_an_entry_that_is_already_there(tmp_path):
    renderer = StubRenderer()
    generate_module.build_entry(TRACER, root=tmp_path, render=renderer)
    _, built = generate_module.build_entry(
        TRACER, root=tmp_path, force=True, render=renderer
    )
    assert built
    assert renderer.calls == 2


def test_force_does_not_read_the_manifest_it_is_about_to_replace(
    tmp_path, tracer_recipe
):
    """A rebuild is a decision already made, so staleness cannot block it."""
    renderer = StubRenderer()
    generate_module.build_entry(TRACER, root=tmp_path, render=renderer)
    directory = tracer_recipe.entry_directory(tmp_path)
    (directory / recipe_module.MANIFEST_FILE_NAME).write_text('{ truncated')

    _, built = generate_module.build_entry(
        TRACER, root=tmp_path, force=True, render=renderer
    )
    assert built
    assert entry_state(tracer_recipe, tmp_path) is EntryState.BUILT


def test_a_stale_entry_is_refused_rather_than_skipped_or_overwritten(tmp_path):
    """Neither silently kept nor silently discarded — #67 wants both refused."""
    renderer = StubRenderer()
    generate_module.build_entry(TRACER, root=tmp_path, render=renderer)
    directory = generate_module.build_entry(TRACER, root=tmp_path, render=renderer)[0]
    path = directory / recipe_module.MANIFEST_FILE_NAME
    manifest = json.loads(path.read_text())
    manifest['seed'] += 1
    path.write_text(json.dumps(manifest))

    with pytest.raises(generate_module.StaleEntryError):
        generate_module.build_entry(TRACER, root=tmp_path, render=renderer)

    assert renderer.calls == 1, 'a refused entry must not be redrawn'
    assert json.loads(path.read_text()) == manifest, 'nor overwritten'


# --------------------------------------------------------------------------
# What one invocation selects
# --------------------------------------------------------------------------


def test_without_all_the_four_axes_name_one_entry():
    selected = generate_module.select_entries(
        build_all=False,
        tier='full',
        source_class='sersic_n4',
        regime='balanced',
        realisation=0,
    )
    assert selected == (
        recipe_module.EntryAxes('full', 'sersic_n4', 'balanced', 0),
    )


def test_without_all_an_entry_off_the_axes_is_still_selected():
    """So the refusal, not the sweep, is what says what is wrong with it.

    Naming four axes asks for one entry, and one named entry is passed through
    to the recipe even when the recipe will refuse it. Dropping it here instead
    would answer "build r3" with "nothing matched", which reads as a typo
    rather than as a realisation the gallery does not have.
    """
    selected = generate_module.select_entries(
        build_all=False,
        tier='full',
        source_class='blend',
        regime='photon',
        realisation=3,
    )
    assert len(selected) == 1
    with pytest.raises(recipe_module.AxisNotBuiltError):
        recipe_module.build_recipe(
            tier='full', source_class='blend', regime='photon', realisation=3
        )


def test_all_selects_every_entry_the_recipe_knows_about():
    selected = generate_module.select_entries(
        build_all=True,
        tier=None,
        source_class=None,
        regime=None,
        realisation=None,
    )
    assert set(selected) == set(recipe_module.built_entries())
    assert len(selected) > 1


def test_an_axis_given_alongside_all_narrows_the_sweep():
    selected = generate_module.select_entries(
        build_all=True,
        tier='small',
        source_class=None,
        regime=None,
        realisation=None,
    )
    assert selected == recipe_module.built_entries(tier='small')
    assert {axes.tier for axes in selected} == {'small'}


# --------------------------------------------------------------------------
# The command line itself
# --------------------------------------------------------------------------


def test_naming_fewer_than_four_axes_without_all_is_an_error(capsys):
    with pytest.raises(SystemExit) as raised:
        generate_module.main(['--tier', 'full'])
    assert raised.value.code == 2
    assert '--class' in capsys.readouterr().err


def test_all_with_a_filter_matching_nothing_is_an_error_not_a_silent_success(capsys):
    """``--all`` narrowed to nothing exits 2 rather than reporting 0 built."""
    with pytest.raises(SystemExit) as raised:
        generate_module.main(['--all', '--realisation', '3'])
    assert raised.value.code == 2
    assert 'nothing to build' in capsys.readouterr().err


def test_an_undecided_realisation_is_refused_by_the_recipe_not_by_argparse(capsys):
    """The refusal names the ticket that owns ``r3``; "invalid choice" does not.

    #67 wants realisations to be addable without disturbing the ones already
    generated, so the command line deliberately holds no list of them to
    update — the recipe's own refusal is the single place that knows.
    """
    with pytest.raises(SystemExit) as raised:
        generate_module.main(
            [
                '--tier',
                'full',
                '--class',
                'sersic_n4',
                '--regime',
                'balanced',
                '--realisation',
                '3',
            ]
        )
    assert raised.value.code == 2
    message = capsys.readouterr().err
    assert 'invalid choice' not in message
    assert 'r3' in message


def test_a_full_sweep_of_an_already_built_gallery_draws_nothing(tmp_path, capsys):
    """The resume, end to end: a second ``--all`` needs no GalSim at all.

    If the skip were keyed on anything the renderer produces, this would fail
    to import GalSim rather than print a skip line.

    Narrowed to the cheap tier's ``point`` entries — nine of them, one per
    regime per realisation. Since #72 the unnarrowed sweep is the whole
    72-entry gallery, and writing it to ``tmp_path`` would put about 700 MB of
    blank FITS on disk to assert a property that nine entries assert exactly as
    well. Which entries a sweep *holds* is asserted against the full grid
    above, where it costs nothing.
    """
    sweep = ['--all', '--tier', 'small', '--class', 'point']
    entries = recipe_module.built_entries(tier='small', source_class='point')
    assert len(entries) == 9

    renderer = StubRenderer()
    for axes in entries:
        generate_module.build_entry(axes, root=tmp_path, render=renderer)
    drawn = renderer.calls

    assert generate_module.main(sweep + ['--out', str(tmp_path)]) == 0
    assert renderer.calls == drawn

    output = capsys.readouterr().out
    assert output.count('already built, skipping') == len(entries)
    assert '0 built, %d already present' % len(entries) in output


def test_the_output_root_defaults_to_the_data_directory():
    """#67 puts synthetic data beside the real Euclid and HST data."""
    arguments = generate_module.build_parser().parse_args(['--all'])
    assert arguments.out == Path.home() / 'dawis-data' / 'simulations'


def test_the_output_root_stays_overridable():
    arguments = generate_module.build_parser().parse_args(['--all', '--out', '/tmp/g'])
    assert arguments.out == Path('/tmp/g')


# --------------------------------------------------------------------------
# What ``sources.csv`` carries (#71)
# --------------------------------------------------------------------------

#: The blend entry of the cheap tier, so the table under test is eighteen rows
#: rather than fifty.
BLEND = recipe_module.EntryAxes(
    tier='small', source_class='blend', regime='balanced', realisation=0
)


def written_table(tmp_path, axes):
    """One entry's ``sources.csv``, written and read back as rows of strings."""
    recipe = describe(axes)
    write_entry(recipe, StubRenderer()(recipe), tmp_path, commit='x')
    path = recipe.output_paths(tmp_path)['sources.csv']
    with path.open(newline='') as handle:
        return recipe, list(csv.DictReader(handle))


def test_the_blend_table_records_the_companion_the_separation_and_the_ratio(
    tmp_path,
):
    """The three columns #67 adds for this class, and every row carries them."""
    recipe, rows = written_table(tmp_path, BLEND)

    assert list(rows[0]) == list(SOURCE_COLUMNS + BLEND_SOURCE_COLUMNS)
    assert len(rows) == len(recipe.sources)
    for row, source in zip(rows, recipe.sources):
        assert int(row['companion_id']) == source.companion_id
        assert float(row['separation']) == pytest.approx(source.separation)
        assert float(row['flux_ratio']) == pytest.approx(source.flux_ratio)


def test_an_isolated_table_carries_no_blend_columns(tmp_path):
    """A class with no pairs has no pair to describe (#71).

    Three columns of zeros in every other entry would read as a measured
    separation of nothing rather than as the absence of a companion.
    """
    _, rows = written_table(tmp_path, TRACER)

    assert list(rows[0]) == list(SOURCE_COLUMNS)
    for column in BLEND_SOURCE_COLUMNS:
        assert column not in rows[0]


# --------------------------------------------------------------------------
# Where the true noise parameters are allowed to appear (#72)
# --------------------------------------------------------------------------

#: The three numbers that parameterise a regime. #67 records them for a human
#: reading a result and hands them to no run: DAWIS estimates the noise itself,
#: as it would on real data, and a run that could read them off an input would
#: be testing a weaker claim than the one the algorithm makes.
NOISE_FIELDS = ('sky_level', 'read_noise', 'gain')


def written_entry(tmp_path, axes):
    """One entry's six files on disk, keyed by their contract names."""
    recipe = describe(axes)
    write_entry(recipe, StubRenderer()(recipe), tmp_path, commit='x')
    return recipe, recipe.output_paths(tmp_path)


@pytest.mark.parametrize('regime', recipe_module.REGIME_NAMES)
def test_the_manifest_is_where_the_true_noise_parameters_are(tmp_path, regime):
    """All three, written down, for every regime the gallery now has (#72)."""
    recipe, paths = written_entry(tmp_path, replace(TRACER, regime=regime))
    manifest = json.loads(paths['manifest.json'].read_text())

    assert set(manifest['noise']) == set(NOISE_FIELDS)
    for field in NOISE_FIELDS:
        assert manifest['noise'][field] == pytest.approx(
            getattr(recipe.noise, field)
        )


@pytest.mark.parametrize('regime', recipe_module.REGIME_NAMES)
def test_no_file_a_run_opens_carries_a_noise_parameter(tmp_path, regime):
    """The image DAWIS reads says nothing about the noise in it (#67).

    A run opens ``image.fits``. The other three FITS files and ``sources.csv``
    are read by the metric half of #25, which is not the algorithm under test
    but is written by the same hands — so the rule is asserted across all five
    rather than on the one file a run happens to open today.

    The check is on the *values*, not on the key names, because the leak that
    would matter is a number reaching a run under a header keyword nobody
    thought to look at.
    """
    recipe, paths = written_entry(tmp_path, replace(TRACER, regime=regime))
    secrets = [getattr(recipe.noise, field) for field in NOISE_FIELDS]

    for name in ('image.fits', 'truth_preconv.fits', 'truth_conv.fits', 'psf.fits'):
        with fits.open(paths[name]) as hdus:
            written = [
                value
                for hdu in hdus
                for value in hdu.header.values()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            ]
        for secret in secrets:
            assert not any(
                value == pytest.approx(secret, rel=1e-6) for value in written
            ), '%s leaks a noise parameter' % name

    table = paths['sources.csv'].read_text()
    for secret in secrets:
        assert repr(secret) not in table


#: Every keyword the entry's FITS headers are allowed to carry: the seven the
#: write layer sets, plus the structural ones astropy writes for any image.
#: Asserted as a closed set rather than as an absence of particular names,
#: because the leak that would matter is a noise parameter arriving under a
#: keyword nobody thought to forbid (#72).
ALLOWED_HEADER_KEYWORDS = {
    'SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'EXTEND',
    'CONTENT', 'SIMTIER', 'SIMCLASS', 'SIMREGIM', 'SIMREAL', 'PIXSCALE',
    'ORIGIN',
}


def test_the_regime_reaches_the_files_by_name_only(tmp_path):
    """Its *name* is on every file; its numbers are on none of them.

    The distinction is the point. A human comparing DAWIS's noise estimate
    against the truth needs to know which entry they are looking at, and the
    entry directory and the FITS headers say so. What they must not be able to
    do is read the answer off the input.
    """
    recipe, paths = written_entry(tmp_path, replace(TRACER, regime='photon'))

    assert recipe.entry_directory(tmp_path).name.startswith('sersic_n4_photon_r')
    for name in ('image.fits', 'truth_preconv.fits', 'truth_conv.fits', 'psf.fits'):
        with fits.open(paths[name]) as hdus:
            header = hdus[0].header

        assert set(header) <= ALLOWED_HEADER_KEYWORDS
        assert header['SIMREGIM'] == 'photon' 
