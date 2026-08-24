"""Selection: turning atoms into source models. Reserved, not yet built.

The seventh block of the target layout (#4), and the only one carved empty.
Nothing is moved here, because nothing in the flat package does this job: atom
selection is a *new* concept, and reserving its namespace now is what makes
the concept visible in the layout instead of living only in the RFC.

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
* ``SourceModel`` itself and its ``to_table()`` catalog export.

The API and the criteria are specced on **#17**. The interactive interface
that drives them — where a user builds or corrects a source model by hand —
is specced separately on **#18**, and its framework question includes whether
it ships in this package at all.

Until #17 lands, this package holds no implementation, and no other module
imports it.
"""
