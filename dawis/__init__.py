# Compatibility, deleted with the shims by the contract ticket (#50): the flat
# detection modules leaked these into the ``dawis`` namespace before the carve
# (#42), so they are kept reachable rather than dropped as a side effect of a
# move. Every shim is imported here, and first: the import machinery binds a
# submodule onto its parent package, so a shim first imported by a later
# caller would replace the function of the same name that the star imports
# below bind. Importing them all now means the star imports have the last
# word, as they did before the carve.
from dawis import (  # noqa: F401
    detect_and_deblend,
    hard_threshold,
    label_regions,
    make_regions,
    table_noise_stds,
)
from dawis.make_regions import AnchoredText  # noqa: F401
from dawis.hard_threshold import normaltest, sigmaclip  # noqa: F401

from dawis.cubes import *
from dawis.detection import *
from dawis.wavelet_transforms import *
from dawis.anscombe_transforms import *
from dawis.make_interscale_trees import *
from dawis.restore_objects import *
from dawis.gif import *
from dawis.exceptions import *
from dawis.synthesis_by_analysis import *
