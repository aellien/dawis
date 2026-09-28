"""Selection: turning atoms into source models.

The seventh block of the target layout (#4), carved empty and now being
filled. Nothing was moved here, because nothing in the flat package does this
job: atom selection is a *new* concept, and reserving its namespace ahead of
the work is what made the concept visible in the layout instead of living only
in the RFC.

The distinction it rests on is the one #4 drew. An **atom** is the output unit
of synthesis: one interscale tree reconstructed into an image-domain footprint
with a flux. It is a raw building block, not an astrophysical source — a
galaxy is many atoms, and one atom can straddle two galaxies. A **source
model** is an astrophysical structure expressed as a combination of selected
atoms, and it is the level at which catalog-style measurements — positions,
fluxes, intracluster light fractions — mean anything.

Selection is the step between the two: it filters and groups the atoms a run
produced, by scale, position and morphology, into source models. It is
post-processing, so it reads a finished run's object store rather than running
inside the greedy loop, and a single run can be selected many different ways
without being repeated. Today that work lives in per-paper scripts outside the
package; this is where it comes to live.

What belongs here, once specced:

* the selection API — object store in, source models out;
* the criteria, automatic and scripted, and how existing external selection
  scripts map onto them;
* ``SourceModel`` itself and the ``to_table()`` catalog of a set of them.

The API and the criteria are specced on **#17**. The interactive interface
that drives them — where a user builds or corrects a source model by hand —
is specced separately on **#18**, and its framework question includes whether
it ships in this package at all.

#17 has landed, and what is built is the vocabulary a selection is written
in. A **selection criterion** is a vectorised predicate over a run's atom
index, composed by ``&``, ``|``, ``~`` and :func:`split`, and it reads index
columns rather than pixels (#125). Six of them ship — :class:`Level`,
:class:`Size`, :class:`Distance`, :class:`InMask`, :class:`Kurtosis`,
:class:`Eccentricity` — and a user's own ``def`` is one too, which is most of
what "users build their own pipelines" requires.

Criteria compose two ways, and both partition by construction. :func:`split`
separates one group from the rest, order-free. :class:`Cascade` is an ordered
rule list where the first match claims the atom and the leftovers are
``unclassified``; :func:`mask_sort` is a wrapper over it. Thresholds are in
pixels and levels, or in kpc through a :class:`Units` frame the user supplies,
because the store keeps no pixel scale (#126).

:func:`source_models` turns those groups into :class:`SourceModel` objects —
a name, index rows and the run's path, synthesising an image and a
``det_err`` image on demand — and warns if hand-built groups share an atom.
The unclassified group is no source model: it synthesises to an image through
:func:`dawis.read_synthesised_image` and nothing more (#127).
The catalog is :func:`to_table`, one row per source model, and a flux's
error comes two ways: :func:`bootstrap`, the default, and
:func:`resample_borderline`, the CFIS scheme, which is not a bootstrap. A flux
floor is an argument to each, never state on the model (#128).
"""

from dawis.selection.cascade import Cascade, mask_sort
from dawis.selection.catalog import bootstrap, resample_borderline, to_table
from dawis.selection.criteria import (
    Criterion,
    Distance,
    Eccentricity,
    InMask,
    Kurtosis,
    Level,
    MissingColumn,
    Size,
    atom_position,
    split,
)
from dawis.selection.source_model import SourceModel, source_models
from dawis.selection.units import Units

__all__ = [
    'Cascade',
    'Criterion',
    'Distance',
    'Eccentricity',
    'InMask',
    'Kurtosis',
    'Level',
    'MissingColumn',
    'Size',
    'SourceModel',
    'Units',
    'atom_position',
    'bootstrap',
    'mask_sort',
    'resample_borderline',
    'source_models',
    'split',
    'to_table',
]
