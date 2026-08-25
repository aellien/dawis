"""Tier 0 behaviour of the cubes that stack per-scale planes.

A cube is a container: it holds a stack of 2D planes, it says how big it is,
it refuses anything that is not a stack, and it survives a trip to disk and
back unchanged. Each kind of cube also carries the one label that says what
its numbers mean.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import dawis
import dawis.cubes as cubes

#: Shape of the cubes under test: 8 by 6 pixels over 4 scales.
CUBE_SHAPE = (8, 6, 4)

#: Every cube class, in the one vocabulary the package exports.
CUBE_CLASSES = (
    'datacube',
    'wavelet_datacube',
    'coarse_datacube',
    'support_datacube',
    'label_datacube',
)


@pytest.fixture
def planes() -> np.ndarray:
    """A cube whose every plane is distinguishable from every other."""
    rng = np.random.default_rng(20260823)
    return rng.normal(0.0, 1.0, CUBE_SHAPE)


def test_a_cube_reports_the_shape_it_was_built_from(planes):
    """The advertised sizes are the sizes of the array."""
    cube = dawis.datacube(planes)

    assert cube.shape == CUBE_SHAPE, (
        f"the cube reports shape {cube.shape} but was built from an array of "
        f"shape {CUBE_SHAPE}"
    )
    assert (cube.x_size, cube.y_size, cube.z_size) == CUBE_SHAPE, (
        "the per-axis sizes disagree with the shape: got "
        f"{(cube.x_size, cube.y_size, cube.z_size)}, expected {CUBE_SHAPE}"
    )


def test_a_cube_hands_back_the_plane_that_was_put_in(planes):
    """Slicing by scale returns that scale, unchanged and in order."""
    cube = dawis.datacube(planes)

    for scale in range(CUBE_SHAPE[2]):
        np.testing.assert_array_equal(
            cube.array[:, :, scale],
            planes[:, :, scale],
            err_msg=f"plane {scale} came back changed or out of order",
        )


@pytest.mark.parametrize("shape", [(8, 6), (8,), (8, 6, 4, 2)])
def test_anything_that_is_not_a_stack_of_planes_is_refused(shape):
    """A cube is three-dimensional; nothing else will do."""
    with pytest.raises(dawis.DawisDimensionError):
        dawis.datacube(np.zeros(shape))


def test_a_wavelet_cube_remembers_its_wavelet(planes):
    """A wavelet cube carries the transform its numbers came from."""
    cube = dawis.wavelet_datacube(planes, "BSPL")

    assert cube.wavelet_type == "BSPL"
    assert cube.fheader["DCTYPE"] == "WAVELET"
    assert cube.fheader["WAVTYPE"] == "BSPL"


def test_a_support_cube_remembers_its_threshold_and_its_noise(planes):
    """A support cube carries how it was cut and the noise it was cut on."""
    noise = planes[:, :, 0].flatten()

    cube = dawis.support_datacube(planes, "HARD", noise)

    assert cube.threshold_type == "HARD"
    assert cube.fheader["DCTYPE"] == "SUPPORT"
    np.testing.assert_array_equal(cube.noise_pixels, noise)


def test_a_label_cube_remembers_how_many_regions_it_holds(planes):
    """A label cube carries the region count of every scale."""
    counts = np.array([3.0, 5.0, 2.0, 0.0])

    cube = dawis.label_datacube(planes, counts)

    np.testing.assert_array_equal(cube.lab_counts, counts)
    assert cube.fheader["LABC"] == counts.sum()


def test_a_cube_survives_a_trip_to_disk_and_back(planes, tmp_path: Path):
    """Writing a cube to FITS and reading it back changes nothing."""
    written = dawis.wavelet_datacube(planes.astype(np.float32), "BSPL")
    filename = tmp_path / "cube.fits"
    written.to_fits(filename)

    read_back = dawis.wavelet_datacube.from_fits(filename)

    np.testing.assert_array_equal(
        read_back.array,
        written.array,
        err_msg="the cube came back from FITS with different numbers",
    )
    assert read_back.shape == written.shape, (
        f"the cube came back from FITS with shape {read_back.shape} instead "
        f"of {written.shape}"
    )
    assert read_back.wavelet_type == written.wavelet_type


def test_region_labels_are_unique_across_the_whole_cube():
    """No label is reused on another scale, so a label names one region."""
    support_array = np.zeros(CUBE_SHAPE)
    for scale in range(CUBE_SHAPE[2]):
        support_array[1, 1, scale] = 1.0
        support_array[5, 4, scale] = 1.0
    support = dawis.support_datacube(support_array, "HARD", np.zeros(10))

    labels = dawis.label_regions(support)

    flagged = labels.array[labels.array != 0]
    assert flagged.size == np.unique(flagged).size, (
        "label uniqueness broken: the same label is used for regions on more "
        "than one scale"
    )
    np.testing.assert_array_equal(
        labels.lab_counts,
        np.full(CUBE_SHAPE[2], 2.0),
        err_msg="the per-scale region counts do not match the two regions "
                "planted on every scale",
    )


def test_the_cube_classes_live_in_the_top_level_cubes_module():
    """The cube classes are shared vocabulary, so they sit above the blocks.

    Named off :mod:`dawis.cubes` and off the package, which re-exports them:
    the two have to be the same objects, or a caller reaching for one
    spelling gets a different class from a caller reaching for the other.
    """

    for name in CUBE_CLASSES:
        assert getattr(cubes, name) is getattr(dawis, name)
