"""Every knob of a run, in one validated object.

A DAWIS run has thirty-odd parameters, and the entry point used to check
exactly one of them: that the input filename ended in ``.fits``. Everything
else was found out the expensive way or not at all. A ``conditions`` typo
surfaced deep inside the wavelet transform, an iteration into the run. A
``starting_level`` at or above ``n_levels`` surfaced as no error whatsoever —
the level loop was simply empty, and the run spent its time writing out an
untouched copy of its own input. :class:`Config` holds the whole set and
checks it in ``__post_init__``, so a run that cannot work fails before it
solves anything.

*Before it solves anything*, and not always before it reads: one rule cannot
fire that early. ``n_levels`` defaults to ``None``, meaning the image decides
the depth, so the rules that compare a scale index against that depth have no
number to compare against until the image is open. The pipeline rebuilds the
configuration the moment it has one — a second construction, so the same
``__post_init__`` runs — which is still before the first wavelet transform.

The checks come in two kinds, and the second is the reason this is one object
rather than thirty arguments. Most knobs have a range they must sit in on
their own — a fraction that must be a fraction, a scale that cannot be
negative. A handful are only wrong *in company*: ``min_span`` above
``max_span`` asks for a tree that spans more scales than it is allowed to,
``starting_level`` at or above ``n_levels`` asks the loop to run over no
scales at all. No single value is out of range in either case, so no
per-argument check can catch them.

The instance is frozen. Validation at construction only guarantees anything
if what was validated is what runs, and a run whose configuration changes
underneath it is exactly the bug the goldens cannot see.

Reproducibility
---------------

:attr:`Config.seed` defaults to ``None``, and ``None`` means *fresh entropy
every run*. That is the science default on purpose: a pinned seed left in a
configuration file quietly makes every run of a campaign draw the same noise,
and the bias that introduces is invisible in the output. Explicit seeds are
for tests.

Reproducibility is not lost by that choice, it is moved: :func:`resolve_seed`
draws the seed once at run start and the pipeline logs it into the run
directory, so any run — including one that was never meant to be repeated —
can be repeated afterwards by reading its own log.

The generator is a :class:`numpy.random.RandomState` rather than the
newer :class:`numpy.random.Generator`, and that is a deliberate and
load-bearing choice. ``RandomState`` reproduces the exact draws of
``numpy.random.seed`` plus the module-level ``numpy.random`` functions DAWIS
called before #53 — the same Mersenne Twister, the same polar transform for
normals. ``Generator`` draws normals by a different method and would produce
a different, equally valid, stream. That stream is what the published
Ellien+2021 results and every recorded golden were produced from, so
switching generator families would move the science for no gain. When there
is a reason to move it, it is a science change with its own ticket.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from dawis.exceptions import ConfigError

__all__ = [
    'AXIS_CARRIERS',
    'BOUNDARY_CONDITIONS',
    'CG_LINE_DIRECTIONS',
    'CG_STEP_SIZES',
    'UNWIRED_KNOBS',
    'DEFAULT_CG_MAX_ITER',
    'MINIMIZERS',
    'PARALLEL_BACKENDS',
    'SEED_BITS',
    'SYNTHESIS_FILTERS',
    'SYNTHESIS_OPERATORS',
    'Config',
    'make_rng',
    'resolve_seed',
]

#: Iteration cap of the conjugate-gradient solver, as published.
#:
#: Ellien+2021 ran 200; `7c5ba8e` cut it to 3 in 2024 without a decision
#: behind it, and #32 reverted that drift. At 200 the solver's own
#: ``sigma_flux`` test does the stopping, so the descent converges rather
#: than being cut off; at 3 that test is vestigial and atom fluxes sit 25-29%
#: lower (#29). It is a knob and not a constant because **#26 owns the
#: question of what it should be** — its sweep is the experiment, and this is
#: the value the sweep starts from.
DEFAULT_CG_MAX_ITER = 200

#: Formulae the conjugate-gradient solver has a branch for when it builds the
#: conjugate direction coefficient — Fletcher-Reeves, Polak-Ribière-Polyak,
#: Hestenes-Stiefel and conjugate descent, all four transcribed from the
#: published solve. #74 swept them and #82 chose ``'HS'``, which is why
#: :attr:`Config.step_size` defaults to it and the published method does not.
#: Stated here beside the ``if`` chain in
#: :meth:`~dawis.trees.construction.InterscaleTree.CG_minimization` rather
#: than derived from it — this module is shared vocabulary that imports no
#: block of the package. That every value in it has a branch behind it is
#: asserted on a real tree, in ``tests/unit/test_cg_axes.py``.
CG_STEP_SIZES = ('FR', 'PRP', 'HS', 'CD')

#: Readings of the conjugate direction recursion the solver has a branch for.
#: ``'as_written'`` is ``v = fr + beta * old_fr``, what the published
#: pseudocode prints and what DAWIS has always run; ``'textbook'`` is
#: ``v = fr + beta * v``, what both published CG algorithms on this problem
#: do (#76). They agree on the first update, because ``v`` starts as a copy
#: of the residual gradient, and part company on the second. #82 ruled the
#: as-written form a documented variant rather than a defect — the textbook
#: form is better on neither metric on any class — so it stays the default.
CG_LINE_DIRECTIONS = ('as_written', 'textbook')

#: Operators the solver has a branch for when it rebuilds an image from a
#: tube — the cascaded smoothing of the published path, the plain sum of the
#: planes, and the genuine transpose of the transform (#84, #98).
#:
#: **This tuple is an allow-list, not a default.** The default is
#: :attr:`Config.synthesis_operator`, ``'TRANSPOSE'`` since #104 wired the
#: axis and #82 chose the value. The method's own default stays the published
#: ``'ADJOINT'`` and :func:`~dawis.restoration.synthesis.restore_object`'s
#: stays ``'SUM'``: a run reaches neither, because the pipeline names the
#: operator on every call, and a direct caller of either is asking for that
#: function rather than for DAWIS 1.0's arm.
#:
#: ``None`` is a value of the *knob* and not of this tuple. It means the
#: chosen minimizer has no operator axis — Condat-Vu — and which minimizers
#: those are is :data:`AXIS_CARRIERS`.
#:
#: Stated here beside the ``if`` chains in
#: :meth:`~dawis.trees.construction.InterscaleTree.CG_minimization` for the
#: same reason as the two tuples above, and guarded there for a sharper one:
#: those chains have no ``else``, so a value naming none of them left
#: ``reconstructed`` unassigned and surfaced as an ``UnboundLocalError``
#: several frames from the cause (#108).
SYNTHESIS_OPERATORS = ('ADJOINT', 'SUM', 'TRANSPOSE')

#: Algorithms that solve a tree — the minimizer axis (#74, #82, #102).
#: ``'bijaoui_rue'`` is the published conjugate gradient of Ellien+2021,
#: :meth:`~dawis.trees.construction.InterscaleTree.CG_minimization`; the other
#: three are the proximal arms in :mod:`dawis.trees.minimizers`. #82 kept this
#: axis user-settable with **no shortlist**, so every value #74 measured ships
#: — except ``'wolfe'``, which #20 deleted from the 1.0 tree and preserved on
#: the ``v0.0.7-h`` tag, a standing deletion #96 does not re-open.
#:
#: **This tuple is an allow-list, not a default**, like
#: :data:`SYNTHESIS_OPERATORS`. The default is :attr:`Config.minimizer`,
#: ``'bijaoui_rue'`` — #82's choice, and the published algorithm.
#:
#: Which axes each minimizer carries is stated twice, in two shapes: here as
#: :data:`AXIS_CARRIERS`, the public rule a knob is refused against, and in
#: :data:`dawis.trees.minimizers.MINIMIZER_AXES`, the solver's own table,
#: beside the branches it describes. This module is shared vocabulary that
#: imports no block of the package, so the two are written separately and
#: asserted equal instead, in ``tests/unit/test_minimizers.py``.
MINIMIZERS = ('bijaoui_rue', 'fista', 'condat_vu', 'projected_gradient')

#: Low-band à trous filters a solve can run with, by the name an atom records
#: under :attr:`~dawis.restoration.synthesis.Atom.filter`. Every minimizer
#: carries this axis; #82 chose ``'BSPL'``, the published B3-spline.
#:
#: Named here and built in :data:`dawis.restoration.synthesis.FILTERS`, which
#: is the one place a name becomes coefficients. Stated rather than imported
#: for the reason :data:`BOUNDARY_CONDITIONS` is, and the two agreeing is
#: asserted in ``tests/unit/test_config.py``.
SYNTHESIS_FILTERS = ('BSPL', 'HAAR')

#: Which minimizers carry each of the three optional arm axes — the public
#: face of :data:`dawis.trees.minimizers.MINIMIZER_AXES`, inverted, because
#: what a caller who named a knob needs back is *which arms take this knob*.
#:
#: The three axes in it are the ones a run can be configured **without**.
#: Minimizer and filter are not: every arm carries both, so there is no
#: configuration in which either is absent, and they are checked against
#: their own tuples alone.
#:
#: An axis the chosen minimizer does not carry is ``None`` on
#: :class:`Config`, never a value — a Condat-Vu that named ``'SUM'`` would
#: read as a Condat-Vu that chose an operator, when what it did was dissolve
#: the fork. The same rule the sweep's own table states
#: (``restoration_tests/arms/grid.py``), and :meth:`Config._check_arm` is
#: where a run is held to it.
AXIS_CARRIERS = {
    'synthesis_operator': ('bijaoui_rue', 'fista', 'projected_gradient'),
    'step_size': ('bijaoui_rue',),
    'line_direction': ('bijaoui_rue',),
}

#: Knobs this class accepts and validates that **no run reads**.
#:
#: One kind, since #104 emptied the other.
#:
#: *Not read yet.* Nothing, now. #96 landed the six arm axes one at a time
#: and #104 wired all six at once, because #3's goldens are re-recorded once,
#: after that wiring (#105). Between #100 and #104
#: :attr:`Config.step_size` and :attr:`Config.line_direction` sat here —
#: accepted, validated, and carried to no solver — and a run solved the
#: published ``'FR'`` and ``'as_written'`` whatever they said. Both are
#: honoured now, and every remaining name below is the second kind.
#:
#: *Read by nothing, by decision.* :attr:`Config.extent_sep`,
#: :attr:`Config.ecc_sep`, :attr:`Config.lvl_sep_lin` and
#: :attr:`Config.lvl_sep_op` were the per-tree routing rule until #103
#: switched it off. They are **kept** rather than deleted because whether a
#: routing rule should come back is a question #96 rules outside its
#: destination, and the evidence that would answer it does not exist — level
#: is perfectly confounded with source class on all 222 of #74's fixtures.
#: Nothing is going to wire these; a later map either gives them a meaning
#: again or deletes them.
#:
#: The gap is recorded here rather than left implicit, because provenance
#: reads this class: ``tests/goldens/recording.py`` and
#: ``benchmarks/run_benchmark.py`` write every field of a ``Config`` into a
#: table documented as the values a run *used*, and a knob nothing honours
#: would make that table false. They drop these names instead. Without the
#: four thresholds here, #105's re-recorded ``provenance.toml`` would list
#: ``lvl_sep_op = 3`` in that table and a reader reproducing from it would
#: conclude routing was applied.
UNWIRED_KNOBS = ('extent_sep', 'ecc_sep', 'lvl_sep_lin', 'lvl_sep_op')

#: Boundary conditions the à trous transform accepts. Stated here so an
#: unusable value is refused at construction rather than an iteration into the
#: run, and stated *again* rather than imported from
#: :func:`~dawis.wavelets.atrous.boundary_conditions`: this module is shared
#: vocabulary that every block may import, so it imports no block itself. The
#: two lists agreeing is asserted instead, in ``tests/unit/test_config.py``.
BOUNDARY_CONDITIONS = ('loop', 'prolongation', 'mirror')

#: Parallel backends a run may name, beside ``None`` for *let the worker count
#: decide*. Stated here rather than imported from :data:`dawis.parallel.BACKENDS`
#: for the same reason as :data:`BOUNDARY_CONDITIONS`, and the two lists
#: agreeing is asserted in ``tests/unit/test_config.py``.
PARALLEL_BACKENDS = ('serial', 'ray')

#: Width of a generated seed. ``RandomState`` accepts a 32-bit seed, so this
#: is the whole space it can be drawn from and not a truncation of a wider one.
SEED_BITS = 32


def resolve_seed(seed: int | None) -> int:
    """Settle what seed a run will actually use.

    Parameters
    ----------
    seed : int or None
        The configured seed. ``None`` — the science default — draws a fresh
        one from the operating system's entropy source.

    Returns
    -------
    int
        The seed, always a number. The caller is expected to record it: a run
        started from ``None`` is only reproducible if the seed it drew is
        written down.

    See Also
    --------
    make_rng : Turns this into the generator the run draws from.
    """

    if seed is None:
        return secrets.randbits(SEED_BITS)
    return seed


def make_rng(seed: int) -> np.random.RandomState:
    """Build the one generator a run threads to every random consumer.

    Parameters
    ----------
    seed : int
        A settled seed, as :func:`resolve_seed` returns.

    Returns
    -------
    numpy.random.RandomState
        The generator. One per run, shared by every consumer, which is what
        makes the draw order — and therefore the run — reproducible. See the
        module docstring for why this is ``RandomState`` and not
        :class:`numpy.random.Generator`.
    """

    return np.random.RandomState(seed)


@dataclass(frozen = True)
class Config:
    """Every knob of one DAWIS run, checked at construction.

    Holds what a run *does*, not where it reads or writes: the input image and
    the output directory are arguments of the entry point, because the same
    configuration is meant to be run over many images. Everything here is
    keyword-only in practice — there are twenty-nine of them and no useful
    order to them — and every one has the published default.

    Parameters
    ----------
    n_workers : int, optional
        Workers to spread tree building and restoration over. 1 runs in this
        process. Scheduling only: since #56 the worker count cannot change
        what a run computes, only how fast it computes it. Default 1.
    backend : str or None, optional
        Which of :data:`PARALLEL_BACKENDS` spreads that work. ``None``, the
        default, lets ``n_workers`` decide — serial on one worker, the default
        parallel backend above one, which is what DAWIS did before the knob
        existed. Naming one selects it instead of that guess, for any
        ``n_workers`` above 1. It does **not** override ``n_workers = 1``: a block
        asking for one worker runs in this process whatever is active, which
        is :func:`dawis.parallel.pmap`'s rule and not this knob's to break.
        ``backend = 'ray'`` with ``n_workers = 1`` therefore stands a cluster up
        and sends it nothing. Scheduling only, like ``n_workers``. Default None.
    n_threads : int, optional
        Threads each process — the driver and every worker alike — may give a
        single numpy or scipy array operation. **Not the same knob as**
        ``n_workers``: a worker is a whole process this run hands a patch of
        trees to, a thread is what one array operation is split across inside
        one process, and OpenBLAS, MKL and OpenMP own the second.

        Unlike ``n_workers`` this is *not* scheduling only. A sum split over
        four threads is added in a different association order from one added
        on a single thread, so raising it moves an atom's flux in its last
        digits — within the golden tolerances, never in a decision like an
        atom count or a convergence flag (#61). 1, the default, is what makes
        a run depend on its input and nothing else.

        Applied per process and **not** divided among the workers, so a run
        may use up to ``n_workers × n_threads`` cores at once: keep that
        product at or below the core count of the machine. Oversubscribing it
        costs time, never correctness. Default 1.
    starting_level : int, optional
        Coarsest wavelet scale the greedy loop starts at. Default 2.
    n_levels : int or None, optional
        Wavelet planes to decompose onto. ``None`` takes the deepest the image
        supports, from :func:`~dawis.wavelets.max_levels`, which is why the
        rule against ``starting_level`` only fires once it is a number.
        Default None.
    tau : float, optional
        Fraction of a region's peak intensity below which a neighbour on the
        same scale can become its own interscale maximum. Default 0.8.
    n_sigmas : float, optional
        Detection threshold, in sigmas of the image noise. Default 5.
    deblend_contrast : float, optional
        Contrast below which two blended peaks are not split. Default 0.1.
    gamma : float, optional
        Attenuation applied to every synthesised atom. Default 0.2.
    min_span, max_span : int, optional
        Fewest and most scales an interscale tree may span. Defaults 2 and 3.
    lvl_sep_big : int, optional
        Scale at or above which an object counts as big, for ``gamma``.
        Default 6.
    rm_gamma_for_big : bool, optional
        Skip the attenuation for big objects. Default False.
    lvl_deblend : int, optional
        Scale at or above which deblending runs; below it the noise makes
        splitting meaningless. Default 3.
    extent_sep : float, optional
        Routes nothing since #103; see :data:`UNWIRED_KNOBS`. Was: extent
        below which a tree took the Haar synthesis branch, as a fraction of
        its bounding box. Default 0.1.
    ecc_sep : float, optional
        Routes nothing since #103; see :data:`UNWIRED_KNOBS`. Was:
        eccentricity at or above which a tree took the Haar branch. Default
        0.95.
    lvl_sep_lin : int, optional
        Routes nothing since #103; see :data:`UNWIRED_KNOBS`. Was: scale at
        or above which the Haar branch was never taken. Default 2.
    lvl_sep_op : int, optional
        Routes nothing since #103; see :data:`UNWIRED_KNOBS`. Was: scale
        above which the adjoint synthesis operator replaced the plain sum of
        planes. Default 3.
    ceps : float, optional
        Convergence threshold. A scale stops when its window-averaged
        normalised epsilon falls below this. Default 1e-3.
    scale_lvl_eps : float, optional
        Factor on that epsilon, for moving the threshold without touching
        ``ceps``. Default 1.
    conditions : {'loop', 'prolongation', 'mirror'}, optional
        Boundary condition of the wavelet transform. Default ``'loop'``.
    max_iter : int, optional
        Hard cap on greedy iterations, counted across every scale. Unrelated
        to ``cg_max_iter`` despite the name they nearly share. Default 500.
    size_patch : int, optional
        Trees per unit of parallel work. Default 100.
    inpaint_res : bool, optional
        Refill the holes subtraction digs in the residual before each
        iteration. Default True.
    iptd_sigma : float, optional
        Depth, in sigmas of the noise, below which a residual pixel is
        inpainted. Default 3.
    resume : bool, optional
        Read an iteration back from disk when its atoms are already there.
        Default True.
    seed : int or None, optional
        Seed of the run's one generator. ``None``, the default, draws fresh
        entropy per run and the pipeline logs what it drew — see the module
        docstring. Explicit seeds are for tests. Default None.
    cg_max_iter : int, optional
        Iteration cap of the conjugate-gradient solver inside
        :meth:`~dawis.trees.construction.InterscaleTree.CG_minimization`.
        Default :data:`DEFAULT_CG_MAX_ITER`, the published 200; **#26 owns
        the question of what it should be.**
    minimizer : {'bijaoui_rue', 'fista', 'condat_vu', 'projected_gradient'}, optional
        Which algorithm solves every interscale tree of the run —
        :data:`MINIMIZERS`. Default ``'bijaoui_rue'``, the published
        conjugate gradient of Ellien+2021.
    filter_kw : {'BSPL', 'HAAR'}, optional
        Low-band à trous filter the solve runs with —
        :data:`SYNTHESIS_FILTERS`. Default ``'BSPL'``, the published
        B3-spline. The *detection* decomposition is B3-spline whatever this
        says; this is the arm's filter, not the pipeline's.
    synthesis_operator : {'TRANSPOSE', 'SUM', 'ADJOINT'} or None, optional
        Which operator rebuilds an image from a tube —
        :data:`SYNTHESIS_OPERATORS`. Default ``'TRANSPOSE'``, the true
        transpose of the transform, which is what #82 chose over the
        published ``'ADJOINT'``. ``None`` on a minimizer that carries no
        operator axis, and **required** to be None there; see
        :data:`AXIS_CARRIERS`.
    step_size : {'HS', 'FR', 'PRP', 'CD'} or None, optional
        Formula for the conjugate direction coefficient of the conjugate
        gradient. Default ``'HS'`` — Hestenes-Stiefel, what #82 chose, and
        the one default in this class that **moves** off the published arm,
        which ran ``'FR'``. ``None``, and required to be None, on every
        minimizer but ``'bijaoui_rue'``: no proximal arm has a conjugate
        direction.
    line_direction : {'as_written', 'textbook'} or None, optional
        Which reading of the conjugate direction recursion the solver takes —
        see :data:`CG_LINE_DIRECTIONS`. Default ``'as_written'``, the
        published one. ``None`` like ``step_size``, and for the same reason.
    coarse : bool, optional
        Whether a tree's tube carries its coarse term :math:`c_{L+1}` as one
        more channel (#88, #89). Default True, which is what #82 chose and
        the largest single effect #74 measured. A property of the tube, so
        every minimizer carries it.

    Attributes
    ----------
    Every parameter above, under the same name and holding what was passed.
    The instance is frozen, so what was validated is what the run reads.

    Raises
    ------
    ConfigError
        If any knob is outside its own range, or if two of them contradict
        each other. Every failure names the knob, the value and what would
        have been accepted. Also if a keyword is not a knob at all, which
        is what a caller passing a knob's older spelling gets.

    Examples
    --------
    >>> config = Config()
    >>> config.cg_max_iter
    200
    >>> config.seed is None
    True
    >>> Config(min_span = 4, max_span = 2)
    Traceback (most recent call last):
    dawis.exceptions.ConfigError: min_span 4 is above max_span 2 --> a tree cannot span more scales than it is allowed to
    """

    n_workers: int = 1
    backend: str | None = None
    n_threads: int = 1
    starting_level: int = 2
    n_levels: int | None = None
    tau: float = 0.8
    n_sigmas: float = 5
    deblend_contrast: float = 0.1
    gamma: float = 0.2
    min_span: int = 2
    max_span: int = 3
    lvl_sep_big: int = 6
    rm_gamma_for_big: bool = False
    lvl_deblend: int = 3
    extent_sep: float = 0.1
    ecc_sep: float = 0.95
    lvl_sep_lin: int = 2
    lvl_sep_op: int = 3
    ceps: float = 1E-3
    scale_lvl_eps: float = 1
    conditions: str = 'loop'
    max_iter: int = 500
    size_patch: int = 100
    inpaint_res: bool = True
    iptd_sigma: float = 3
    resume: bool = True
    seed: int | None = None
    cg_max_iter: int = DEFAULT_CG_MAX_ITER
    minimizer: str = 'bijaoui_rue'
    filter_kw: str = 'BSPL'
    synthesis_operator: str | None = 'TRANSPOSE'
    step_size: str | None = 'HS'
    line_direction: str | None = 'as_written'
    coarse: bool = True

    def __new__(cls, *args: Any, **knobs: Any) -> Config:
        """Refuse a name this class has no knob for, before anything is set.

        The generated ``__init__`` refuses one too, with a :exc:`TypeError`
        that names the argument and stops there. That is the wrong family —
        this is a run configured in a way that cannot work, like every other
        rejection in this class — and it is half the message. A name that is
        not a knob is as often one that was *renamed* as one that was
        mistyped, so what the caller needs back is the set of names that do
        work. #60 renamed the worker count with no alias and no deprecation
        warning, and this line is the whole of what a caller still passing
        the pre-1.0 spelling of it gets: an error, at construction, with
        ``n_workers`` in it.

        Raises
        ------
        ConfigError
            If any keyword is not a field of this class.
        """

        knob_names = [field.name for field in fields(cls)]
        unknown = [name for name in knobs if name not in knob_names]
        if unknown:
            raise ConfigError(
                'no such knob: %s --> the knobs are %s'
                % (', '.join(unknown), ', '.join(knob_names))
            )
        return super().__new__(cls)

    def __post_init__(self) -> None:
        self._check_ranges()
        self._check_pairs()

    # -- validation ------------------------------------------------------
    #
    # Split in two because the two kinds fail for different reasons and the
    # message has to say which. A range failure names one knob and what it
    # should have been; a pair failure names two and what they mean together,
    # and there is no single value to point at.

    def _check_ranges(self) -> None:
        """Check every knob against the range it must sit in on its own."""

        for name in ('n_workers', 'n_threads', 'max_iter', 'size_patch',
                     'cg_max_iter'):
            self._at_least(name, 1)

        for name in ('starting_level', 'lvl_sep_big', 'lvl_deblend',
                     'lvl_sep_lin', 'lvl_sep_op'):
            self._at_least(name, 0)

        for name in ('n_sigmas', 'ceps', 'scale_lvl_eps'):
            self._above(name, 0)

        self._at_least('iptd_sigma', 0)

        # Fractions. `tau`, `gamma` and `deblend_contrast` multiply a measured
        # quantity, `extent_sep` compares against a covered area fraction, and
        # `ecc_sep` against an eccentricity: none of the five means anything
        # outside the unit interval.
        for name in ('tau', 'gamma'):
            self._in_unit_interval(name, include_zero = False)
        for name in ('deblend_contrast', 'extent_sep', 'ecc_sep'):
            self._in_unit_interval(name, include_zero = True)

        self._one_of('conditions', BOUNDARY_CONDITIONS)
        self._one_of('minimizer', MINIMIZERS)
        self._one_of('filter_kw', SYNTHESIS_FILTERS)

        # None is a value on the three optional axes and not a missing one --
        # it means *the chosen minimizer has no such axis* -- so it is excused
        # the closed-set check here. Whether it is the right answer is a rule
        # about two knobs at once, and `_check_arm` owns it.
        if self.synthesis_operator is not None:
            self._one_of('synthesis_operator', SYNTHESIS_OPERATORS)
        if self.step_size is not None:
            self._one_of('step_size', CG_STEP_SIZES)
        if self.line_direction is not None:
            self._one_of('line_direction', CG_LINE_DIRECTIONS)

        # None is a value here and not a missing one -- it means *decide from
        # the worker count* -- so it is excused the closed-set check rather
        # than added to the set, which would let it be named as a backend.
        if self.backend is not None:
            self._one_of('backend', PARALLEL_BACKENDS)

        if self.n_levels is not None:
            self._at_least('n_levels', 1)

        if self.seed is not None and not 0 <= self.seed < 2 ** SEED_BITS:
            raise ConfigError(
                'seed %r is out of range --> must be None, or an integer in '
                '[0, 2**%d)' % (self.seed, SEED_BITS)
            )

    def _check_pairs(self) -> None:
        """Check the rules no single knob is wrong on its own for."""

        self._check_arm()

        if self.min_span > self.max_span:
            raise ConfigError(
                'min_span %s is above max_span %s --> a tree cannot span more '
                'scales than it is allowed to'
                % (self.min_span, self.max_span)
            )

        # Everything below compares a scale index against the number of
        # scales, so none of it is checkable until `n_levels` is a number.
        # When it is None the image decides: the pipeline rebuilds the
        # configuration as soon as it has read one, which runs this again with
        # a real depth and still before the first wavelet transform.
        if self.n_levels is None:
            return

        if self.starting_level >= self.n_levels:
            raise ConfigError(
                'starting_level %s is not below n_levels %s --> the greedy '
                'loop would run over no scales at all and the run would '
                'write out its own input'
                % (self.starting_level, self.n_levels)
            )

        # The four ``lvl_*`` knobs are deliberately *not* checked against
        # ``n_levels``. Each names the scale at which a branch switches, so
        # one sitting above the depth of the decomposition means that branch
        # is never taken — degenerate, but a real configuration, and the one a
        # small image gets from the defaults: a 64x64 cutout supports six
        # scales and ``lvl_sep_big`` defaults to 6. Rejecting it would refuse
        # runs that work.

    def _check_arm(self) -> None:
        """Refuse an arm: an axis set on a minimizer that has none, or missing.

        The rule #104 settled, and the one place a run is held to it. An axis
        the chosen minimizer does not carry has to be ``None`` and nothing
        else, including the default this class ships — so
        ``Config(minimizer = 'condat_vu')`` is refused, and the refusal names
        the three knobs to null out.

        Refused rather than nulled out here, which is the choice this ticket
        had. A ``Config`` is read back as the record of what a run used:
        ``tests/goldens/recording.py`` and ``benchmarks/run_benchmark.py``
        write every field of one into a provenance table. Coercing
        ``step_size = 'HS'`` to None behind the caller would make that table
        right and the caller's belief wrong, and a caller who thinks they
        chose Hestenes-Stiefel on Condat-Vu has misunderstood something an
        error can fix and a silent fix cannot. It is the same failure
        :data:`UNWIRED_KNOBS` exists for, met from the third side.

        Raises
        ------
        ConfigError
            If a knob names an axis the minimizer has not got, or leaves one
            it has got at None.
        """

        absent = [axis for axis, carriers in AXIS_CARRIERS.items()
                  if self.minimizer not in carriers
                  and getattr(self, axis) is not None]
        if absent:
            raise ConfigError(
                'minimizer %r carries no %s axis --> pass None for each; the '
                'minimizers that carry one are %s'
                % (self.minimizer, ', '.join(absent),
                   sorted({carrier for axis in absent
                           for carrier in AXIS_CARRIERS[axis]}))
            )

        missing = [axis for axis, carriers in AXIS_CARRIERS.items()
                   if self.minimizer in carriers
                   and getattr(self, axis) is None]
        if missing:
            raise ConfigError(
                'minimizer %r carries a %s axis --> name a value for each; '
                'None means an axis this minimizer has not got, and it has'
                % (self.minimizer, ', '.join(missing))
            )

    # -- one-knob checks -------------------------------------------------

    def _at_least(self, name: str, bound: int) -> None:
        """Refuse a knob below an inclusive floor — a count, or a scale index."""

        if getattr(self, name) < bound:
            raise ConfigError(
                '%s %r is below %s --> must be %s or more'
                % (name, getattr(self, name), bound, bound)
            )

    def _above(self, name: str, bound: float) -> None:
        """Refuse a knob at or below an exclusive floor.

        Separate from :meth:`_at_least` because zero is a different mistake
        for the two: no workers is one, a convergence threshold of zero is a
        loop that never stops.
        """

        if getattr(self, name) <= bound:
            raise ConfigError(
                '%s %r is not above %s --> must be greater than %s'
                % (name, getattr(self, name), bound, bound)
            )

    def _in_unit_interval(self, name: str, include_zero: bool) -> None:
        """Refuse a fraction that is not one.

        ``include_zero`` is what separates a threshold from a multiplier: a
        contrast of zero is a threshold nothing clears, which is meaningful,
        while a ``gamma`` of zero attenuates every atom to nothing.
        """

        value = getattr(self, name)
        if value > 1 or value < 0 or (value == 0 and not include_zero):
            raise ConfigError(
                '%s %r is outside its range --> must be a fraction in %s'
                % (name, value, '[0, 1]' if include_zero else '(0, 1]')
            )

    def _one_of(self, name: str, allowed: tuple[str, ...]) -> None:
        """Refuse a spelling outside a closed set, and list the set."""

        if getattr(self, name) not in allowed:
            raise ConfigError(
                '%s %r is not a recognised value --> must be one of %s'
                % (name, getattr(self, name), list(allowed))
            )
