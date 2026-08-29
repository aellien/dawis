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

import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pytest

# ``simulations`` is deliberately not part of the distributed package (see
# ``test_package_surface.py``), so it is not importable off the install. It is
# a sibling of ``dawis`` in the source tree, and that is where it is found.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from simulations import generate as generate_module  # noqa: E402
from simulations import recipe as recipe_module  # noqa: E402
from simulations.write import EntryState, entry_state, write_entry  # noqa: E402

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


def test_an_axis_no_ticket_has_decided_is_not_swept():
    """The sweep holds exactly the axis values a ticket has written down.

    #68 decided ``sersic_n4`` / ``balanced`` / ``r0``, and #70 added the other
    two isolated classes. ``blend`` needs a grid of its own, and the other two
    regimes and realisations need numbers, so all three axes stay narrowed
    until the ticket that owns them lands.
    """
    swept = recipe_module.built_entries()
    assert {axes.source_class for axes in swept} == {
        'sersic_n4',
        'point',
        'sersic_n1_halo',
    }
    assert {axes.regime for axes in swept} == {'balanced'}
    assert {axes.realisation for axes in swept} == {0}


def test_both_tiers_are_swept_because_neither_is_refused():
    """#67 decides both tiers completely, so both build (#68's README)."""
    assert {axes.tier for axes in recipe_module.built_entries()} == set(
        recipe_module.TIERS
    )


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
    """Asking for an undecided axis with ``--all`` sweeps nothing, quietly.

    The generator turns that into an error message of its own; the recipe's
    job is only to report that there is nothing to build.
    """
    assert recipe_module.built_entries(source_class='blend') == ()


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


def test_without_all_an_undecided_entry_is_still_selected():
    """So the refusal, not the sweep, is what reports which ticket owns it.

    Dropping it here instead would answer "build the blend class" with
    "nothing matched", which reads as a typo rather than as work not yet done.
    """
    selected = generate_module.select_entries(
        build_all=False,
        tier='full',
        source_class='blend',
        regime='photon',
        realisation=2,
    )
    assert len(selected) == 1
    with pytest.raises(recipe_module.AxisNotBuiltError):
        recipe_module.build_recipe(
            tier='full', source_class='blend', regime='photon', realisation=2
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


def test_all_with_a_filter_matching_nothing_says_it_is_undecided_work(capsys):
    """Not "no such thing" — the class exists, its numbers are not decided."""
    with pytest.raises(SystemExit) as raised:
        generate_module.main(['--all', '--class', 'blend'])
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
    """
    renderer = StubRenderer()
    for axes in recipe_module.built_entries():
        generate_module.build_entry(axes, root=tmp_path, render=renderer)
    drawn = renderer.calls

    assert generate_module.main(['--all', '--out', str(tmp_path)]) == 0
    assert renderer.calls == drawn

    output = capsys.readouterr().out
    assert output.count('already built, skipping') == len(
        recipe_module.built_entries()
    )
    assert '0 built, %d already present' % len(recipe_module.built_entries()) in output


def test_the_output_root_defaults_to_the_data_directory():
    """#67 puts synthetic data beside the real Euclid and HST data."""
    arguments = generate_module.build_parser().parse_args(['--all'])
    assert arguments.out == Path.home() / 'dawis-data' / 'simulations'


def test_the_output_root_stays_overridable():
    arguments = generate_module.build_parser().parse_args(['--all', '--out', '/tmp/g'])
    assert arguments.out == Path('/tmp/g')
