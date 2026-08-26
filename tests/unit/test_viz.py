"""Tier 0 — the viz block, and the drawing the carve moved into it (#46).

Two things are checked here. First, that ``import dawis`` works with the
``[viz]`` extra absent. Second, the two functions that replaced code which used
to run inside the pipeline — the per-type cube dispatch, and the animation
rebuilt from a finished run's own output.

The import rule itself moved to ``test_import_graph.py`` with #58, which
walks the source tree once for all three of #4's dependency rules instead of
pinning each beside the code that moved. What stays here is the half that is
about the drawing: that a headless install still runs.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")

import dawis  # noqa: E402
import dawis.viz as viz  # noqa: E402
# Named off `dawis.cubes` rather than off `dawis`, where the classes are also
# re-exported: this is the address the carve gave them (#41), and it is the one
# that stayed put when the flat `dawis.datacube` module went with #50.
from dawis.cubes import (  # noqa: E402
    CoarseCube,
    Cube,
    LabelCube,
    SupportCube,
    WaveletCube,
)


def test_dawis_imports_without_matplotlib():
    """A headless install of dawis, with no [viz] extra, still runs."""
    blocked = (
        "import builtins, sys\n"
        "real = builtins.__import__\n"
        "def blocked(name, *a, **k):\n"
        "    if name.split('.')[0] == 'matplotlib':\n"
        "        raise ImportError('matplotlib is not installed')\n"
        "    return real(name, *a, **k)\n"
        "builtins.__import__ = blocked\n"
        "import dawis\n"
        "assert not any(m.startswith('matplotlib') for m in sys.modules)\n"
        "print(dawis.run.__name__)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", blocked], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr


def test_the_pipeline_takes_no_animation_knob():
    """The animation is not a flag on the solver any more, but a tool."""
    import inspect

    from dawis.config import Config

    assert "gif" not in inspect.signature(dawis.run).parameters
    assert "gif" not in Config.__dataclass_fields__


# --- the per-type cube dispatch -------------------------------------------


@pytest.fixture
def cubes():
    """One cube of each type, small enough to draw quickly."""
    array = np.random.default_rng(46).normal(0, 1, (8, 8, 3))
    return {
        "base": Cube(np.copy(array), dctype="NONE"),
        "wavelet": WaveletCube(np.copy(array), "BSPL"),
        "coarse": CoarseCube(np.copy(array), "BSPL"),
        "support": SupportCube(
            np.zeros((8, 8, 3)), "HARD", noise_pixels=array[:, :, 0].flatten()
        ),
        "label": LabelCube(np.zeros((8, 8, 3)), lab_counts=[1, 2, 3]),
    }


def test_plot_cube_dispatches_on_cube_type(cubes):
    """A wavelet cube gets its own branch; every other cube gets the plain one."""
    base = viz.plot_cube.dispatch(Cube)
    wavelet = viz.plot_cube.dispatch(WaveletCube)

    assert wavelet is not base
    assert viz.plot_cube.dispatch(CoarseCube) is base
    assert viz.plot_cube.dispatch(LabelCube) is base
    assert viz.plot_cube.dispatch(SupportCube) is base


@pytest.mark.parametrize("kind", ["base", "wavelet", "coarse", "support", "label"])
def test_plot_cube_draws_every_cube_type(cubes, kind, tmp_path):
    """One call draws any cube, which is what replaced the method overrides."""
    save_path = tmp_path / f"{kind}.png"
    viz.plot_cube(cubes[kind], ncol=3, show=False, save_path=str(save_path))
    assert save_path.is_file()


def test_plot_label_counts_draws(cubes, tmp_path):
    save_path = tmp_path / "counts.pdf"
    viz.plot_label_counts(cubes["label"], show=False, save_path=str(save_path))
    assert save_path.is_file()


def test_plot_noise_histogram_draws(cubes, tmp_path):
    save_path = tmp_path / "hist.png"
    viz.plot_noise_histogram(cubes["support"], show=False, save_path=str(save_path))
    assert save_path.is_file()


# --- the two structures: a region and a tree ------------------------------


@pytest.fixture
def detection_cubes():
    """A wavelet cube and the label cube of two nested regions on it.

    The coefficients straddle zero because the drawing takes the log of both
    the maximum and the negated minimum to place its colourbar ticks.
    """
    array = np.random.default_rng(46).normal(0, 50, (24, 24, 3))
    labels = np.zeros((24, 24, 3))
    labels[8:14, 8:14, 0] = 1
    labels[6:16, 6:16, 1] = 2
    return (
        WaveletCube(array, "BSPL"),
        LabelCube(labels, lab_counts=[1, 1, 0]),
    )


def a_region(label, level, x0, x1, y0, y1, with_bbox=True):
    """One measured region, filled in as the detection block fills them."""
    from dawis.detection import region

    coords = np.array([(x, y) for x in range(x0, x1) for y in range(y0, y1)])
    properties = dict(
        label=label,
        level=level,
        coords=coords,
        x_max=(x0 + x1) // 2,
        y_max=(y0 + y1) // 2,
        eccentricity=0.3,
        area=coords.shape[0],
    )
    if with_bbox:
        properties["bbox"] = (x0, y0, x1, y1)
    return region(**properties)


def test_plot_region_draws(detection_cubes, tmp_path):
    """``region.plot`` before #46, now a function.

    The region is built without a ``bbox`` because the moved body can only
    draw that kind — see the defect recorded in ``plot_region``'s own Notes.
    """
    save_path = tmp_path / "region.pdf"
    viz.plot_region(
        a_region(1, 0, 8, 14, 8, 14, with_bbox=False),
        *detection_cubes,
        show=False,
        save_path=str(save_path),
    )
    assert save_path.is_file()


@pytest.mark.parametrize(
    'shape, expected',
    [((2, 2), True), ((5, 7), True), ((1, 1), False), ((1, 5), False)],
)
def test_contourable_is_the_grid_matplotlib_needs(shape, expected):
    """The outline guard of #52, which replaced a handler that caught nothing.

    ``plt.contour`` raises ``TypeError`` below a 2x2 grid. A one-pixel region
    is a normal detection at the finest scale, so the drawers test for it
    instead of catching — catching would have swallowed every other
    ``TypeError`` the drawing can raise.
    """
    assert viz.regions.contourable(np.zeros(shape)) is expected


def test_plot_interscale_tree_draws(detection_cubes, tmp_path):
    """``InterscaleTree.plot`` before #46, now a function."""
    from dawis.trees import InterscaleTree

    wavelet, labels = detection_cubes
    fine = a_region(1, 0, 8, 14, 8, 14)
    root = a_region(2, 1, 6, 16, 6, 16)
    tree = InterscaleTree(
        interscale_maximum=root,
        region_list=[fine, root],
        wavelet_cube=wavelet,
        label_cube=labels,
        det_err_array=np.zeros((24, 24, 3)),
        clip=4,
    )

    save_path = tmp_path / "tree.png"
    viz.plot_interscale_tree(tree, wavelet, labels, show=False,
                             save_path=str(save_path))
    assert save_path.is_file()


def test_plot_wavelet_stds_is_broken_by_matplotlib(cubes):
    """The one moved body that cannot run, pinned rather than quietly carried.

    ``WaveletCube.std_plot`` set its log scale with ``basey``, which
    matplotlib removed in 3.3. The defect travelled with the move; this test
    fails the day someone fixes it, which is when the docstring note in
    ``plot_wavelet_stds`` should go too.
    """
    with pytest.raises(TypeError, match="basey"):
        viz.plot_wavelet_stds(cubes["wavelet"], show=False)


# --- the animation, rebuilt from a finished run ---------------------------


def fake_run(work_dir: Path, n_iterations: int = 3, size: int = 16) -> str:
    """Write what a finished run leaves behind, in miniature.

    One atom per iteration, plus the restored and residual images the run
    writes at the end — which is everything
    :func:`dawis.viz.rebuild_animation` reads.
    """
    from astropy.io import fits

    rng = np.random.default_rng(46)
    image = rng.normal(10, 1, (size, size))
    outpath = str(work_dir / "run")

    restored = np.zeros((size, size))
    for it in range(1, n_iterations + 1):
        atom = dawis.Atom(
            image=np.full((4, 4), float(it)),
            det_err_image=np.zeros((4, 4)),
            bbox=(it, it, it + 4, it + 4),
            level=it,
            eccentricity=0.5,
            filter_kw="BSPL",
            flag_convergence=True,
            sum_wr=1.0,
            norm_wr=1.0,
        )
        restored[it : it + 4, it : it + 4] += atom.image
        dawis.write_ol_to_pickle([atom], f"{outpath}.ol.it{it:03d}.pkl")

    fits.PrimaryHDU(restored).writeto(f"{outpath}.restored.fits")
    fits.PrimaryHDU(image - restored).writeto(f"{outpath}.residuals.fits")
    return outpath


def test_rebuild_animation_draws_one_frame_per_iteration(tmp_path):
    """The animation comes out of the files a run already wrote."""
    outpath = fake_run(tmp_path)

    assert viz.rebuild_animation(outpath, show_progress=False) == 3

    frames = sorted(tmp_path.glob("run.frame.it*.png"))
    assert [frame.name for frame in frames] == [
        "run.frame.it001.png",
        "run.frame.it002.png",
        "run.frame.it003.png",
    ]


def test_iteration_dumps_are_ordered_and_deduplicated(tmp_path):
    """A resumed run can hold both formats; the same iteration is one frame."""
    outpath = fake_run(tmp_path, n_iterations=2)
    (tmp_path / "run.ol.it002.hdf5").touch()

    dumps = viz.animation.iteration_dumps(outpath)

    assert [it for it, _ in dumps] == [1, 2]


def test_rebuild_animation_needs_a_finished_run(tmp_path):
    """Without the atom dumps there is nothing to replay, and it says so."""
    with pytest.raises(FileNotFoundError, match="no atom dumps"):
        viz.rebuild_animation(str(tmp_path / "nothing"))


def test_rebuild_convergence_plot_reads_the_trace(tmp_path):
    """The convergence figure the pipeline used to draw, drawn afterwards."""
    outpath = str(tmp_path / "run")
    with open(f"{outpath}.cpar.txt", "w") as trace:
        trace.write("# level, it, window_normeps, normeps, eps, n_obj, time_it\n")
        trace.write("2 1 1.0 1.0 0.5 0.1 0.9 3 0:00:01.5\n")
        trace.write("2 2 0.5 0.5 0.2 0.3 0.7 5 0:00:01.2\n")

    viz.rebuild_convergence_plot(outpath)

    assert (tmp_path / "run.conv.pdf").is_file()
