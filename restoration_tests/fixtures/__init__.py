"""The fixture layer — what every arm of #74 is measured on.

A **fixture** is one interscale tree, cut out of one simulation-gallery entry
by one mock iteration, together with the injected source it is trying to
reconstruct. It holds enough to run a reconstruction and score it with
nothing to look up elsewhere: the three tubes, the tree's geometry, the
source's parameters, the truth image cut to the tree's footprint, and a
reconstruction from the published default arm to look at.

Five modules and a command line, split the way ``simulations/`` is split —
the decisions on one side of the seam, the machinery that needs a heavy
import on the other.

:mod:`~restoration_tests.fixtures.matching` decides which injected source a
tree belongs to. It imports nothing outside the standard library, and it is
where the two coordinate conventions of #75 meet.

:mod:`~restoration_tests.fixtures.store` is what a fixture *is*, and how a
set of them is written and read. It imports h5py and nothing of DAWIS.

:mod:`~restoration_tests.fixtures.mock` runs the mock iteration: one pass
from a gallery image to a list of trees. This is the module that imports
DAWIS.

:mod:`~restoration_tests.fixtures.quicklook` draws the panel, and is the only
module that imports matplotlib.

:mod:`~restoration_tests.fixtures.build` is the command line that puts the
four together over the gallery.
"""
