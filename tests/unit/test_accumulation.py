"""Tier 0 — the running images of a run (#49).

``Reconstruction`` holds the four arrays a run advances together: the
residual, the restored image, the restored image split by wavelet scale, and
the accumulated detection error. Every atom updates all four inside its own
bounding box, and the defect this class exists to prevent is one of the four
drifting out of step with the others — which is what four separate locals in
the pipeline made easy.
"""

from __future__ import annotations

import numpy as np
import pytest

from dawis.restoration import Reconstruction


class FakeAtom:
    """The four attributes ``Reconstruction`` reads off a restored object."""

    def __init__(self, bbox, image, level, det_err_image=None):
        self.bbox = bbox
        self.image = np.asarray(image, dtype=float)
        self.level = level
        self.det_err_image = (
            np.zeros_like(self.image) if det_err_image is None
            else np.asarray(det_err_image, dtype=float)
        )


def atom_of(x_min, y_min, size, value, level, det_err=0.0):
    """One square atom of constant value, placed at ``(x_min, y_min)``."""
    return FakeAtom(
        bbox=(x_min, y_min, x_min + size, y_min + size),
        image=np.full((size, size), value),
        level=level,
        det_err_image=np.full((size, size), det_err),
    )


def test_the_input_image_is_copied_not_held():
    """The caller's array is the run's input and must survive the run."""
    image = np.ones((8, 8))
    run = Reconstruction(image, n_levels=3)

    run.add([atom_of(0, 0, 2, value=1.0, level=1)])

    np.testing.assert_array_equal(image, np.ones((8, 8)))


def test_the_residual_keeps_the_dtype_of_the_image():
    """A promoted residual would move every number the loop computes after it."""
    run = Reconstruction(np.ones((8, 8), dtype=np.float32), n_levels=3)

    assert run.residuals.dtype == np.float32


def test_an_atom_lands_in_its_bounding_box_and_nowhere_else():
    run = Reconstruction(np.zeros((8, 8)), n_levels=3)

    atom = run.add([atom_of(2, 3, 2, value=5.0, level=1)])

    assert atom[2:4, 3:5].tolist() == [[5.0, 5.0], [5.0, 5.0]]
    assert atom.sum() == pytest.approx(4 * 5.0)


def test_every_image_moves_by_the_same_atoms():
    """Restored up, residual down, per-level and detection error along with them."""
    run = Reconstruction(np.full((8, 8), 10.0), n_levels=3)

    run.add([atom_of(0, 0, 2, value=1.0, level=2, det_err=0.25)])

    np.testing.assert_allclose(run.restored + run.residuals, np.full((8, 8), 10.0))
    assert run.restored.sum() == pytest.approx(4.0)
    assert run.restored_per_level[..., 2].sum() == pytest.approx(4.0)
    assert run.restored_per_level[..., 0].sum() == 0.0
    assert run.det_err.sum() == pytest.approx(1.0)


def test_the_per_level_image_splits_atoms_by_the_scale_they_came_from():
    run = Reconstruction(np.zeros((8, 8)), n_levels=4)

    run.add([
        atom_of(0, 0, 2, value=1.0, level=1),
        atom_of(4, 4, 2, value=3.0, level=3),
    ])

    assert run.restored_per_level[..., 1].sum() == pytest.approx(4.0)
    assert run.restored_per_level[..., 3].sum() == pytest.approx(12.0)
    assert run.restored_per_level.shape == (8, 8, 4)


def test_overlapping_atoms_accumulate():
    """Two atoms over the same pixel both count; neither replaces the other."""
    run = Reconstruction(np.zeros((8, 8)), n_levels=3)

    atom = run.add([
        atom_of(1, 1, 3, value=2.0, level=1),
        atom_of(1, 1, 3, value=5.0, level=1),
    ])

    assert atom[1, 1] == pytest.approx(7.0)


def test_iterations_accumulate_across_calls():
    run = Reconstruction(np.zeros((8, 8)), n_levels=3)

    run.add([atom_of(0, 0, 2, value=1.0, level=1)])
    run.add([atom_of(0, 0, 2, value=1.0, level=1)])

    assert run.restored[0, 0] == pytest.approx(2.0)
    assert run.residuals[0, 0] == pytest.approx(-2.0)


def test_an_iteration_that_restored_nothing_changes_nothing():
    run = Reconstruction(np.full((8, 8), 3.0), n_levels=3)

    atom = run.add([])

    assert not atom.any()
    np.testing.assert_array_equal(run.residuals, np.full((8, 8), 3.0))


def test_inpainting_refills_the_holes_subtraction_dug():
    """Deep negative pixels are replaced; the rest of the residual is left alone."""
    run = Reconstruction(np.zeros((8, 8)), n_levels=3)
    run.residuals[0, 0] = -100.0

    np.random.seed(20260823)
    run.inpaint(mean=0.0, sigma=1.0, iptd_sigma=3)

    assert abs(run.residuals[0, 0]) < 100.0
