"""Shared fixtures for the DAWIS test suite.

The suite is organised in the three tiers decided on #3:

* ``tests/unit`` — Tier 0. Mathematical properties and validation, seconds,
  no goldens.
* ``tests/regression`` — Tier 1 (256x256 cutout, every commit) and Tier 2
  (full 900x900 gallery run, ``slow``).
* ``tests/goldens`` — the recorded outputs plus ``provenance.toml``.

Fixtures here are the frame the later tiers hang on: the gallery image, the
agreed cutout of it, a seeded configuration and a temporary run directory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

#: Reference image shipped inside the package, used by Tier 1 and Tier 2.
GALLERY_IMAGE = "tidal_group_f814w_sci.fits"

#: Centre of the Tier 1 cutout, as ``(x, y)``, supplied by Amael on #3.
#: The FITS array is indexed ``(ny, nx)``, so ``x`` is the *second* axis.
CUTOUT_CENTRE = (222, 242)

#: Side of the square Tier 1 cutout, in pixels.
CUTOUT_SIZE = 256

#: The one seed every seeded test uses. Explicit seeds are for tests only;
#: the science default stays ``seed=None`` (#3).
TEST_SEED = 20260823


@pytest.fixture(scope="session")
def gallery_image_path() -> Path:
    """Path to the gallery FITS image inside the installed package."""
    import dawis

    path = Path(dawis.__file__).parent / "gallery" / GALLERY_IMAGE
    if not path.is_file():
        pytest.skip(f"gallery image not found: {path}")
    return path


@pytest.fixture(scope="session")
def gallery_image(gallery_image_path: Path) -> np.ndarray:
    """The full 900x900 gallery image, read once per session.

    The array is returned read-only so one test cannot corrupt another's
    input. Copy it if you need to modify it.
    """
    from astropy.io import fits

    data = fits.getdata(gallery_image_path)
    data.setflags(write=False)
    return data


@pytest.fixture(scope="session")
def cutout(gallery_image: np.ndarray) -> np.ndarray:
    """The agreed 256x256 Tier 1 cutout of the gallery image.

    Centred on :data:`CUTOUT_CENTRE`, so the slice is
    ``image[yc - 128:yc + 128, xc - 128:xc + 128]``. Read-only, like
    :func:`gallery_image`.
    """
    xc, yc = CUTOUT_CENTRE
    half = CUTOUT_SIZE // 2
    patch = np.array(gallery_image[yc - half:yc + half, xc - half:xc + half])
    patch.setflags(write=False)
    return patch


@pytest.fixture
def seeded_config() -> dict:
    """Pipeline knobs for a deterministic test run.

    Returned as a plain mapping of keyword arguments because ``Config`` does
    not exist yet; it lands in #24 and this fixture becomes a ``Config``
    instance then. The current pipeline also takes no ``seed`` argument, so
    callers pop ``seed`` and pin the generator themselves until #24 threads
    it through.

    ``max_iter`` is pinned low on purpose. It is a declared speed knob that
    keeps the default suite inside its five-minute budget, not a science
    claim; Tier 2 runs the shipping value of 200. The authoritative Tier 1
    values are owned by #36 and may be overridden there.
    """
    return {
        "seed": TEST_SEED,
        "n_cpus": 1,
        "n_levels": 9,
        "starting_level": 2,
        "tau": 0.1,
        "gamma": 0.5,
        "ceps": 1e-4,
        "min_span": 2,
        "max_span": 3,
        "lvl_sep_big": 5,
        "max_iter": 3,
        "data_dump": False,
        "gif": False,
        "resume": False,
    }


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """An empty directory for one run's outputs, removed after the test."""
    path = tmp_path / "run"
    path.mkdir()
    return path
