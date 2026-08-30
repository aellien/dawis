"""The panel a human looks at before a single metric is written.

#75 says this is what makes the fixture ticket worth having on its own: the
moment fixtures exist, a researcher can put a reconstruction beside its truth
and learn something. Three images per fixture — the injected truth over the
tree's box, what the published default arm made of it, and the residual
between them.

The stretch
-----------
Truth and reconstruction share one linear scale, because the panel exists to
be compared across and two independently scaled images cannot be. A
reconstruction at half the truth's flux, scaled to its own maximum, looks
exactly like the truth; scaled to the truth's, it looks like what it is.

The residual gets its own scale, symmetric about zero on a diverging map,
because what it carries is sign — light left behind against light invented —
and a scale that put zero anywhere but the middle would hide which.

Linear, not logarithmic. A tree's box is a patch of one source, not a field,
so the dynamic range inside it is the source's own profile rather than the
gallery's three orders of magnitude, and a log stretch would spend the whole
colour bar on the wings.

This is the only module of the fixture layer that imports matplotlib, the way
:mod:`dawis.viz` is for DAWIS. Nothing else in the battery needs it, and a
dump with ``--no-quicklook`` never reaches here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from restoration_tests.fixtures.store import Fixture

__all__ = ['draw_quicklook', 'draw_quicklooks', 'panel_limits']


def panel_limits(fixture: Fixture) -> tuple[tuple[float, float], tuple[float, float]]:
    """The two colour ranges the panel draws on.

    Parameters
    ----------
    fixture : Fixture

    Returns
    -------
    shared : tuple of float
        ``(low, high)`` spanning truth and reconstruction together.
    residual : tuple of float
        ``(-r, r)`` about zero, with ``r`` the largest absolute residual.
    """

    # A tree's box is floored at 30 pixels a side, so a fixture's arrays are
    # never empty and there is no size to guard. A reconstruction that
    # matches its truth exactly is reachable, though -- and a colour range of
    # (0, 0) is not one matplotlib can draw -- so that one is.
    residual = fixture.reference_reconstruction - fixture.truth_cutout
    reach = float(np.max(np.abs(residual))) or 1.0

    low = float(min(fixture.truth_cutout.min(),
                    fixture.reference_reconstruction.min()))
    high = float(max(fixture.truth_cutout.max(),
                     fixture.reference_reconstruction.max()))

    return (low, high), (-reach, reach)


def draw_quicklook(fixture: Fixture, path: str | Path) -> Path:
    """Draw one fixture's panel to a PNG.

    Parameters
    ----------
    fixture : Fixture
        The fixture to draw.
    path : str or pathlib.Path
        Where to write. The parent directory is created if it does not exist.

    Returns
    -------
    pathlib.Path
        The file written.
    """

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)

    residual = fixture.reference_reconstruction - fixture.truth_cutout
    shared, symmetric = panel_limits(fixture)

    figure, axes = plt.subplots(1, 3, figsize = (11, 4))
    panels = (
        ('truth (PSF-convolved)', fixture.truth_cutout, shared, 'viridis'),
        ('%s / %s' % (fixture.reference_arm['filter'],
                      fixture.reference_arm['synthesis_operator']),
         fixture.reference_reconstruction, shared, 'viridis'),
        ('reconstruction - truth', residual, symmetric, 'coolwarm'),
    )

    for axis, (title, image, (low, high), colours) in zip(axes, panels):
        drawn = axis.imshow(image, origin = 'lower', vmin = low, vmax = high,
                            cmap = colours, interpolation = 'nearest')
        axis.set_title(title, fontsize = 9)
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(drawn, ax = axis, fraction = 0.046)

    figure.suptitle(
        '%s  |  mag %.2f, n=%.1f, r_half=%.2f"  |  level %d, extent %.2f, e %.2f'
        % (fixture.fixture_id, fixture.source.magnitude,
           fixture.source.sersic_index, fixture.source.half_light_radius,
           fixture.geometry.level, fixture.geometry.extent,
           fixture.geometry.eccentricity),
        fontsize = 9)
    figure.tight_layout()
    figure.savefig(path, dpi = 110)
    plt.close(figure)

    return path


def draw_quicklooks(fixtures: list[Fixture], directory: str | Path) -> list[Path]:
    """Draw a panel for every fixture of one entry.

    Parameters
    ----------
    fixtures : list of Fixture
    directory : str or pathlib.Path
        Where the panels go. Each is named after the tree in its fixture id,
        so a panel and a fixture are matched by name rather than by position.

    Returns
    -------
    list of pathlib.Path
        The files written, in the order the fixtures came in.
    """

    directory = Path(directory)

    return [draw_quicklook(fixture,
                           directory / ('%s.png' % fixture.fixture_id.rsplit('/', 1)[-1]))
            for fixture in fixtures]
