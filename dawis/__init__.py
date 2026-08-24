# Compatibility, deleted with the shims by the contract ticket (#50): these
# flat modules were bound onto the ``dawis`` namespace before their carve —
# the detection ones by #42, ``deconvolution`` and ``inpainting`` by #44,
# ``store_objects`` by #45 — so
# they are kept reachable rather than dropped as a side effect of a move.
# Every shim is imported here, and first: the import machinery binds a
# submodule onto its parent package, so a shim first imported by a later
# caller would replace the function of the same name that the star imports
# below bind. Importing them all now means the star imports have the last
# word, as they did before the carve.
from dawis import (  # noqa: F401
    deconvolution,
    detect_and_deblend,
    hard_threshold,
    inpainting,
    label_regions,
    make_regions,
    store_objects,
    table_noise_stds,
)
from dawis.make_regions import AnchoredText  # noqa: F401
from dawis.hard_threshold import normaltest, sigmaclip  # noqa: F401
# The store leaked ``h5py`` into ``dawis``; the star imports below no longer
# reach it, because ``synthesis_by_analysis`` names what it imports (#45).
from dawis.store_objects import h5py  # noqa: F401

from dawis.cubes import *
from dawis.detection import *
from dawis.io import *
from dawis.wavelet_transforms import *
from dawis.anscombe_transforms import *
from dawis.make_interscale_trees import *
from dawis.restore_objects import *
from dawis.gif import *
from dawis.exceptions import *
from dawis.synthesis_by_analysis import *
