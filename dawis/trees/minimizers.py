r"""The minimizer axis — which algorithm solves a tree, and how it is chosen.

:meth:`~dawis.trees.construction.InterscaleTree.CG_minimization` is one
minimizer, the published one. #74 measured four more against it on one fixture
set, #82 kept the axis user-settable with no shortlist, and this module is the
other three landing in ``dawis/``: FISTA, projected gradient and Condat-Vu,
ported from ``restoration_tests/arms/proximal.py`` on
``research/restoration-tests`` — the implementation every #74 and #92 number
was taken with.

The problem all four solve
--------------------------
The same one, written down:

.. math::

    \min_{x \ge 0}  \tfrac{1}{2}
        \bigl\| (W x - w_{\mathrm{obs}}) \odot S \bigr\|^{2}

with :math:`W` the à trous transform to the tree's interscale-maximum level,
:math:`S` the support tube and :math:`w_{\mathrm{obs}}` the wavelet tube. The
tube is already zero off the support, so the residual
``wavelet_tube - analysis * support_tube`` the published solve computes *is*
:math:`(w_{\mathrm{obs}} - Wx) \odot S`, and these arms compute it with the
same expression. Nothing about the problem changes between minimizers; only
what is done with the gradient does.

Why the proximal family is here at all
--------------------------------------
The support mask selects a **linear subspace**, so the mask multiply the
published solve already performs *is* its Euclidean projection, and a
projection onto a closed convex set is a proximity operator. The clip at zero
is the second one. DAWIS therefore already computes both prox operators; what
it does not do is apply them where a convergence argument survives — it builds
conjugate directions out of pre-clip residuals and then clips the iterate,
which voids conjugacy for the constrained problem actually solved. Proximal
gradient is the repair, and these three arms are it.

Which axes each minimizer carries
---------------------------------
:data:`MINIMIZER_AXES`, and it is not the same set for all four. Condat-Vu
carries **no synthesis operator**: a primal-dual method keeps the support
constraint in wavelet space as a dual variable and the positivity in image
space as the primal prox, and it moves between them with :math:`W` and its
genuine transpose — never with a synthesis operator, and so with no choice to
sweep. The line direction and beta belong to the conjugate-gradient recursion
and exist nowhere else.

An axis a minimizer does not carry is **refused**, not defaulted and not
ignored. #96 has met a surface that accepts a knob no run honours three times
— :data:`dawis.config.UNWIRED_KNOBS`, ``restore_object``'s live ``deconv``
default, and #108's module comment — and in all three the code was
self-consistent and only the surface was wrong. ``minimize(minimizer =
'condat_vu', synthesis_operator = 'SUM')`` would be a fourth.

What is deliberately *not* copied from the published loop
---------------------------------------------------------
:meth:`~dawis.trees.construction.InterscaleTree.CG_minimization` reproduces
three quirks of Ellien+2021 on purpose, because that method *is* the published
arm. These are new arms, so they do not inherit a defect:

* ``total_flux`` starts at the initial reconstruction's flux, not at 1.0, so
  the first convergence test compares two real fluxes.
* The loop runs ``max_iter`` steps, not ``max_iter + 1``.
* The clip at zero is part of the step rather than applied after it: here the
  projection *is* the backward half of a forward-backward step.

The convergence *rule* is the published one — relative flux change below
``sigma_flux`` — and the cap and tolerance are the published values, so cost
comparisons are between algorithms rather than between tolerances.

The one split that ships unexplained
------------------------------------
``fista.HAAR.SUM`` diverges on 51 of #81's 54 fixtures with the coarse term
on, where ``projected_gradient.HAAR.SUM`` converges on 54 of 54 — same family,
same cell, same fixtures, and the extrapolation is the only difference between
them. ``SUM`` is not an adjoint of anything, so neither arm has a convergence
argument on that cell and both are outside the theory; what is unexplained is
why the momentum decides it. #96 carries it as a sighting, both arms ship, and
the split ships with them on a cell that is not the default. It is written
down here rather than smoothed away, because a caller who selects that cell
gets a zeroed atom and NaN residuals and is owed the reason.

What a diverged solve costs, and who owns it
--------------------------------------------
Divergence comes back as a zeroed atom with NaN residuals. Two consequences
rode on that, both inherited from the research arm and both reproduced
deliberately here — repairing them would have stopped the port matching the
implementation every #74, #81 and #92 number was taken with, which is the one
thing this ticket had to do. They are listed in :func:`minimize`'s Notes, and
#104 — which is what lets a run reach this door at all — settled the first of
them: ``restore_object`` tests ``norm_wr`` as well as the image, so a zeroed
atom is flagged rather than recorded as converged. That test moves no number,
because the image it zeroes is already zero, and it is the rule
``restoration_tests/arms/base.py`` applies too.

What is still open is the second: the growth test lags one step and never
tests the iterate it returns, so an arm can stop *below* the divergence
factor and hand back an atom far above it. Changing when a solve is called
diverged does move numbers, and that is #110.

What is measured, and what it costs
-----------------------------------
:func:`spectral_radius` is not cheap on this problem. All three operators
have a spectral radius within a thousandth of 1.0 with a clustered spectrum,
so the power iteration converges slowly: on a real tree it takes 68 of a
FISTA solve's 96 analysis transforms, 68 of projected gradient's 87, and it
reaches the 100-iteration cap on ``TRANSPOSE`` and on Condat-Vu. The estimate
lands within 0.25% every time — the accuracy is not the problem, the budget
is, and it is paid once per tree on a run that restores thousands. #74 charged
it apart from solve cost for this reason; it is carried on #96 as Drift rather
than tuned here.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from dawis.config import (
    CG_LINE_DIRECTIONS,
    CG_STEP_SIZES,
    DEFAULT_CG_MAX_ITER,
    MINIMIZERS,
    SYNTHESIS_OPERATORS,
)
from dawis.cubes import CoarseCube, LabelCube, WaveletCube
from dawis.exceptions import DawisValueError
from dawis.trees.construction import InterscaleTree, _analysis_tube
from dawis.wavelets import atrous_adjoint

__all__ = [
    'DEFAULT_DIVERGENCE_FACTOR',
    'DEFAULT_POWER_ITERATIONS',
    'DEFAULT_POWER_TOLERANCE',
    'DEFAULT_STEP_SAFETY',
    'MINIMIZER_AXES',
    'minimize',
    'spectral_radius',
]

#: Which axes each minimizer carries, beside the filter and the coarse term —
#: which every one of them carries. An axis absent here is refused when set,
#: for the reason in the module docstring. Mirrors ``_AXES`` in
#: ``restoration_tests/arms/grid.py``, minus ``wolfe``, which #20 deleted from
#: the 1.0 tree, and minus ``deconv``, which #101 removed.
#:
#: **This is the solver's table, not the public one.** #104 makes the six axes
#: a :class:`~dawis.config.Config` surface and decides what a shipped API does
#: with an illegal *combination*; what is decided here is only that a knob
#: naming an axis the chosen algorithm does not have is an error rather than a
#: value quietly dropped.
MINIMIZER_AXES = {
    'bijaoui_rue': ('synthesis_operator', 'step_size', 'line_direction'),
    'fista': ('synthesis_operator',),
    'projected_gradient': ('synthesis_operator',),
    'condat_vu': (),
}

#: Value each carried axis takes when the caller names none. The published
#: solve's own defaults, so ``minimize(minimizer = 'bijaoui_rue')`` is
#: :meth:`~dawis.trees.construction.InterscaleTree.CG_minimization` with its
#: signature untouched. The defaults that **move** are #104's business and
#: :class:`~dawis.config.Config`'s, not this door's.
_AXIS_DEFAULTS = {
    'synthesis_operator': 'ADJOINT',
    'step_size': 'FR',
    'line_direction': 'as_written',
}

#: The allowed values of each axis, for the refusal message.
_AXIS_VALUES = {
    'synthesis_operator': SYNTHESIS_OPERATORS,
    'step_size': CG_STEP_SIZES,
    'line_direction': CG_LINE_DIRECTIONS,
}

#: Cap on the power iterations used to measure the step size. A cap and not a
#: count: :func:`spectral_radius` stops as soon as the estimate has settled,
#: which on most cells is well before this.
DEFAULT_POWER_ITERATIONS = 100

#: Relative size the estimated remaining climb has to fall below before
#: :func:`spectral_radius` stops.
DEFAULT_POWER_TOLERANCE = 1e-3

#: Iterations :func:`spectral_radius` takes before it is allowed to stop on
#: the tolerance, capped by ``n_iterations`` like everything else.
#:
#: The geometric tail correction needs the increments to *be* geometric, and
#: over the first few steps of a map that is not symmetric they are not: on
#: ``fista.BSPL.ADJOINT`` the ratio looks converged at step 3 and the estimate
#: stops 7.2% low, where the same rule with this floor stops 0.5% low. That is
#: the difference between :data:`DEFAULT_STEP_SAFETY` having no margin over
#: the error and having ten times it.
_MINIMUM_POWER_ITERATIONS = 20

#: The step is this over the measured spectral radius. Below one because the
#: measurement is an estimate and a step above ``1 / L`` costs the
#: forward-backward convergence argument.
DEFAULT_STEP_SAFETY = 0.9

#: Seed of the power iteration's starting vector. Fixed, so two runs of one
#: minimizer on one tree take the same step and a run is reproducible. Not
#: drawn from the run's generator (:func:`dawis.config.make_rng`): this is a
#: measurement of an operator, not a random draw of the science, and it has to
#: give the same number on a rerun of one tree in isolation.
_POWER_SEED = 0

#: A solve is called diverged once its wavelet residual exceeds this multiple
#: of the residual it started from.
#:
#: A guard on finiteness alone is not enough here. Proximal gradient needs a
#: genuine adjoint, and ``SUM`` is not one: with the Haar filter the map
#: ``SUM . support . atrous`` has an eigenvalue no positive step can contract,
#: so ``fista.HAAR.SUM`` grows steadily and reaches ``1e27`` at the iteration
#: cap without ever touching an ``inf``. That atom is finite and meaningless.
DEFAULT_DIVERGENCE_FACTOR = 1e3


def spectral_radius(
    operator: Callable[[np.ndarray], np.ndarray],
    shape: tuple[int, int],
    n_iterations: int = DEFAULT_POWER_ITERATIONS,
    tolerance: float = DEFAULT_POWER_TOLERANCE,
) -> float:
    r"""Measure the spectral radius of a linear map by power iteration.

    What the step size needs. For the symmetric case — Condat-Vu's
    :math:`W^{\mathsf{T}} S W` — this is :math:`\|SW\|^{2}` exactly. For the
    two arms that substitute a synthesis operator for the transpose the map is
    not symmetric, and this is its spectral radius rather than its operator
    norm; that is still the right quantity, because what a gradient step needs
    is the eigenvalue that governs whether :math:`I - \tau M` contracts.

    The estimate climbs to the radius **from below**, which is the dangerous
    direction: an estimate that is low makes the step
    :data:`DEFAULT_STEP_SAFETY` over it too *large*, and a step above
    :math:`1/L` costs the convergence argument the arm was written to have. A
    fixed iteration count is not enough to bound that — on a Haar tube the
    estimate is still 10% low after twelve steps and climbing — so the
    iteration stops on the climb rather than on a count, and adds the climb it
    estimates is left.

    That remainder is the tail of a geometric series: the error of a power
    iteration decays by a roughly constant factor per step, so from two
    successive increments the rest of the climb is
    ``increment * ratio / (1 - ratio)``. Adding it is what turns a systematic
    under-estimate into an estimate that brackets the radius, and it is why
    the tolerance can be loose and the cap rarely reached. The first
    :data:`_MINIMUM_POWER_ITERATIONS` steps are taken regardless, because the
    increments are not yet geometric there and the correction would fire on a
    ratio that means nothing.

    The tail describes the climb left **from the step it was read off**, so it
    is dropped again on any step whose increment did not fall. Carrying one
    forward is what the non-symmetric maps reach: FISTA and projected gradient
    substitute a synthesis operator for the transpose, their increments need
    not be monotone, and they are the two arms that actually exhaust the cap.

    Non-monotone also means the norms can *fall*, and the estimate keeps the
    largest one it saw rather than the one it stops on (#86). Returning a dip
    would put the estimate below a norm already observed, which is the
    dangerous direction: on ``[[0, -3], [1, 0]]`` the iteration oscillates
    ``2.2798, 1.3159, 2.2798, ...`` and never settles, and reporting the dip
    gives 1.316 against a true radius of 1.732 — a step of 0.684 against a
    bound ``1 / L`` of 0.577, 18.5% above it, and 31.6% above the 0.520 the
    safety factor was asking for. Keeping the peak returns 2.280 instead:
    above the true radius, so the step is short rather than long, which is the
    side of the bound the convergence argument needs.

    Parameters
    ----------
    operator : callable
        Takes a 2D array and returns one of the same shape.
    shape : tuple of int
        The array shape the operator acts on.
    n_iterations : int, optional
        Cap on the iterations. Default :data:`DEFAULT_POWER_ITERATIONS`.
    tolerance : float, optional
        Stop once the estimated remaining climb is this small relative to the
        estimate. Default :data:`DEFAULT_POWER_TOLERANCE`.

    Returns
    -------
    float
        The radius, or 1.0 when there is nothing to measure — an operator that
        annihilates the starting vector, or a non-positive iteration cap. A
        step length is still needed in both cases and no step is better than
        another, so the fallback is a number rather than an exception.
    """

    vector = np.random.default_rng(_POWER_SEED).normal(size = shape)
    norm = np.linalg.norm(vector)
    radius = 0.0
    tail = 0.0
    taken = 0
    previous_increment = None

    for _ in range(max(n_iterations, 0)):

        vector = vector / norm
        vector = operator(vector)
        norm = float(np.linalg.norm(vector))

        if not np.isfinite(norm) or norm == 0.0:
            break

        increment = norm - radius
        radius = max(radius, norm)

        if increment <= 0.0:
            # The climb is over, or the map is not symmetric enough for it to
            # be monotone. Either way there is no geometric tail to add.
            #
            # The estimate keeps the largest norm it saw rather than this one
            # (#86). On the branch below `norm > radius`, so the `max` is the
            # identity there and the whole climb is unchanged; it bites only
            # here, where taking `norm` would step *down* onto a dip and
            # return an estimate below one already observed.
            tail = 0.0
            break

        taken += 1

        if previous_increment is not None and increment < previous_increment:
            ratio = increment / previous_increment
            tail = increment * ratio / (1.0 - ratio)
            if taken >= _MINIMUM_POWER_ITERATIONS and tail < tolerance * radius:
                break
        else:
            # The increment did not fall, so the two it would be read off are
            # not geometric and there is no climb this step can predict. The
            # tail an earlier, larger increment predicted describes a climb
            # that is no longer the one left, so it goes rather than stands.
            tail = 0.0

        previous_increment = increment

    radius = radius + tail

    return radius if radius > 0.0 else 1.0


# -- the pieces all three proximal arms share ----------------------------


def _synthesis(
    tree: InterscaleTree,
    wavelet_tube: np.ndarray,
    synthesis_operator: str,
    filter: np.ndarray,
    coarse: bool,
) -> np.ndarray:
    """Rebuild an image from a tube with the named operator.

    The same three branches
    :meth:`~dawis.trees.construction.InterscaleTree.CG_minimization` selects
    between, so a FISTA solve and a conjugate-gradient solve on one cell
    differ in the *method* and in nothing else.

    Parameters
    ----------
    tree : dawis.trees.construction.InterscaleTree
    wavelet_tube : numpy.ndarray
    synthesis_operator : str
        One of :data:`~dawis.config.SYNTHESIS_OPERATORS`.
    filter : numpy.ndarray
    coarse : bool

    Returns
    -------
    numpy.ndarray
        2D, the shape of one tube plane.
    """

    if synthesis_operator == 'ADJOINT':
        return tree.synthesis_adjoint(wavelet_tube, filter = filter,
                                      coarse = coarse)
    if synthesis_operator == 'TRANSPOSE':
        return tree.synthesis_transpose(wavelet_tube, filter = filter,
                                        coarse = coarse)
    return tree.synthesis_sum(wavelet_tube)


def _residual(
    wavelet_tube: np.ndarray,
    support_tube: np.ndarray,
    image: np.ndarray,
    filter: np.ndarray,
    coarse: bool,
) -> np.ndarray:
    """The masked wavelet residual, by the published solve's own expression.

    Parameters
    ----------
    wavelet_tube, support_tube : numpy.ndarray
    image : numpy.ndarray
    filter : numpy.ndarray
    coarse : bool

    Returns
    -------
    numpy.ndarray
        3D, the shape of the tube.
    """

    n_levels = wavelet_tube.shape[2] - int(coarse)

    return wavelet_tube - _analysis_tube(image, n_levels, filter,
                                         coarse) * support_tube


def _diverging(
    image: np.ndarray,
    norm_wr: float,
    first_norm_wr: float,
    factor: float,
) -> bool:
    """Whether a solve has left the numbers it can be scored on.

    Two ways out. The iterate stops being finite, which is what the published
    solve's own handling watches for; or the wavelet residual climbs to
    ``factor`` times where it started, which is what an arm whose step is
    contracting nothing does and which never reaches an ``inf`` — see
    :data:`DEFAULT_DIVERGENCE_FACTOR`.

    Parameters
    ----------
    image : numpy.ndarray
    norm_wr, first_norm_wr : float
    factor : float

    Returns
    -------
    bool
    """

    if not np.all(np.isfinite(image)) or not np.isfinite(norm_wr):
        return True

    return norm_wr > factor * first_norm_wr


def _finish(
    wavelet_tube: np.ndarray,
    support_tube: np.ndarray,
    image: np.ndarray,
    filter: np.ndarray,
    coarse: bool,
    diverged: bool,
    first_norm_wr: float | None,
) -> tuple[np.ndarray, float, float]:
    """Score the final iterate, or report the divergence.

    A diverged solve is zeroed and both residual quantities set to ``nan``,
    which is what :func:`dawis.restoration.synthesis.restore_object` then
    reads: it tests ``norm_wr`` as well as the image, so a zeroed atom comes
    back with ``flag_convergence = False`` rather than as a converged one
    (#104).

    **The growth test is applied here, to the iterate that is returned**
    (#110). Inside the loop it reads a residual measured *before* the step,
    so iterate *k* is only tested at iteration *k + 1* — and a loop that ends
    on ``max_iter`` or on the flux test has no iteration *k + 1*. The last
    iterate was therefore the one nothing scored, and on ``fista.HAAR.SUM``
    capped one step below where the in-loop guard fires, the atom came back
    unflagged with a residual over a thousand times its starting value.
    ``Config.cg_max_iter`` is a user knob and this repo has benchmarked at 3.

    Testing it here costs nothing: ``wr`` is the residual of that same
    iterate, computed for ``sum_wr`` and ``norm_wr`` anyway. With this in
    place every iterate is tested — the earlier ones one step late by the
    loop, the last one here — so the lag is no longer a hole.

    This is a **departure** from ``restoration_tests/arms/proximal.py``, which
    keeps the hole and stays the record of what every #74, #81 and #92 number
    was taken with. It can only turn a returned atom into a flagged one; it
    never changes an atom's pixels and never unflags a solve.

    Parameters
    ----------
    wavelet_tube, support_tube : numpy.ndarray
    image : numpy.ndarray
    filter : numpy.ndarray
    coarse : bool
    diverged : bool
    first_norm_wr : float or None
        The wavelet residual norm the solve started from, or None when
        ``max_iter`` left the loop no iteration to take one in — nothing
        moved, so there is no growth to test.

    Returns
    -------
    reconstructed : numpy.ndarray
    sum_wr : float
    norm_wr : float
    """

    if not diverged:
        wr = _residual(wavelet_tube, support_tube, image, filter, coarse)
        norm_wr = float(np.sqrt(np.sum(wr ** 2)))

        diverged = first_norm_wr is not None and _diverging(
            image, norm_wr, first_norm_wr, DEFAULT_DIVERGENCE_FACTOR)

    if diverged:
        return np.zeros_like(image), float('nan'), float('nan')

    return image, float(np.sum(wr)), norm_wr


def _forward_backward(
    tree: InterscaleTree,
    wavelet_tube: np.ndarray,
    support_tube: np.ndarray,
    synthesis_operator: str,
    filter: np.ndarray,
    coarse: bool,
    momentum: bool,
    sigma_flux: float,
    max_iter: int,
) -> tuple[np.ndarray, float, float]:
    r"""Proximal forward-backward, with or without Nesterov momentum.

    The two arms differ in one line, so they are one function: with the
    momentum the arm is FISTA, without it the arm is projected gradient — the
    floor #74 put in the grid to find out what the conjugacy is buying.

    Parameters
    ----------
    tree : dawis.trees.construction.InterscaleTree
    wavelet_tube, support_tube : numpy.ndarray
    synthesis_operator : str
    filter : numpy.ndarray
    coarse : bool
    momentum : bool
        Whether to extrapolate. True is FISTA.
    sigma_flux : float
    max_iter : int

    Returns
    -------
    reconstructed : numpy.ndarray
    sum_wr : float
    norm_wr : float
    """

    def synthesise(tube):
        return _synthesis(tree, tube, synthesis_operator, filter, coarse)

    def analyse(image):
        n_levels = wavelet_tube.shape[2] - int(coarse)
        return _analysis_tube(image, n_levels, filter, coarse) * support_tube

    radius = spectral_radius(lambda image: synthesise(analyse(image)),
                             wavelet_tube.shape[:2])
    step = DEFAULT_STEP_SAFETY / radius

    image = np.clip(synthesise(wavelet_tube), 0.0, None)
    extrapolated = image
    total_flux = float(np.sum(image))
    momentum_t = 1.0

    diverged = False
    first_norm_wr = None

    for iteration in range(1, max_iter + 1):

        wr = _residual(wavelet_tube, support_tube, extrapolated, filter,
                       coarse)

        # The gradient is taken at ``extrapolated``; the residual that is
        # tested for divergence is taken at ``image``. With the momentum on
        # those are different points, and the extrapolated one is not an
        # iterate: it can sit outside the feasible set with a materially
        # larger residual. On the first step, and with the momentum off
        # throughout, the two points are the same one.
        if momentum and iteration > 1:
            step_norm_wr = float(np.linalg.norm(
                _residual(wavelet_tube, support_tube, image, filter, coarse)))
        else:
            step_norm_wr = float(np.linalg.norm(wr))

        if first_norm_wr is None:
            first_norm_wr = step_norm_wr

        # ``wr`` is the *negative* gradient of the data term under the
        # operator standing in for the transpose, so the descent step adds it.
        updated = np.clip(extrapolated + step * synthesise(wr), 0.0, None)

        if momentum:
            next_t = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum_t ** 2))
            weight = (momentum_t - 1.0) / next_t
            extrapolated = updated + weight * (updated - image)
            momentum_t = next_t
        else:
            extrapolated = updated

        image = updated

        old_total_flux = total_flux or 1.0
        total_flux = float(np.sum(image))
        diff_flux = abs(total_flux - old_total_flux) / abs(old_total_flux)

        if _diverging(image, step_norm_wr, first_norm_wr,
                      DEFAULT_DIVERGENCE_FACTOR):
            diverged = True
            break

        if diff_flux < sigma_flux:
            break

    return _finish(wavelet_tube, support_tube, image, filter, coarse,
                   diverged, first_norm_wr)


def _dual_prox(
    candidate: np.ndarray,
    wavelet_tube: np.ndarray,
    support_tube: np.ndarray,
    sigma: float,
) -> np.ndarray:
    r"""The prox of the conjugate of the wavelet-space data term.

    From Moreau's identity applied to
    :math:`g(w) = \tfrac12 \|(w - w_{\mathrm{obs}}) \odot S\|^{2}`: on the
    support the prox is ``(v - sigma * w_obs) / (1 + sigma)``, and off it the
    prox is zero. The ``support_tube *`` is the projection the published solve
    already performs, applied in the space the constraint lives in.

    Parameters
    ----------
    candidate : numpy.ndarray
        The dual point to project, 3D.
    wavelet_tube, support_tube : numpy.ndarray
    sigma : float
        The dual step.

    Returns
    -------
    numpy.ndarray
        3D, the shape of the tube.
    """

    return support_tube * (candidate - sigma * wavelet_tube) / (1.0 + sigma)


def _condat_vu(
    wavelet_tube: np.ndarray,
    support_tube: np.ndarray,
    filter: np.ndarray,
    coarse: bool,
    sigma_flux: float,
    max_iter: int,
) -> tuple[np.ndarray, float, float]:
    r"""Synthesise a tree by the Condat-Vu primal-dual algorithm.

    The arm that dissolves the ``SUM``/``ADJOINT`` fork. It never rebuilds an
    image from a tube with a synthesis operator: the support constraint stays
    in wavelet space as a dual variable and the positivity stays in image
    space as the primal prox, and the two are coupled by :math:`W` and its
    genuine transpose, :func:`~dawis.wavelets.atrous_adjoint`. There is no
    synthesis operator anywhere in it to choose, which is why
    :data:`MINIMIZER_AXES` gives it no operator axis.

    With no smooth term the iteration is Condat's algorithm with
    :math:`f = 0`:

    .. math::

        x_{n+1} &= \Pi_{\ge 0}
                   \bigl( x_{n} - \tau W^{\mathsf{T}} u_{n} \bigr) \\
        u_{n+1} &= \mathrm{prox}_{\sigma g^{*}}
                   \bigl( u_{n} + \sigma W (2 x_{n+1} - x_{n}) \bigr)

    The steps are :math:`\tau = \sigma =` :data:`DEFAULT_STEP_SAFETY` over
    :math:`\|SW\|`, which satisfies Condat's condition
    :math:`\tau \sigma \|SW\|^{2} \le 1` with the same margin the
    forward-backward arms take.

    Parameters
    ----------
    wavelet_tube, support_tube : numpy.ndarray
    filter : numpy.ndarray
    coarse : bool
    sigma_flux : float
    max_iter : int

    Returns
    -------
    reconstructed : numpy.ndarray
    sum_wr : float
    norm_wr : float
    """

    def analyse(image):
        n_levels = wavelet_tube.shape[2] - int(coarse)
        return _analysis_tube(image, n_levels, filter, coarse) * support_tube

    def transpose(tube):
        return atrous_adjoint(tube, filter, coarse = coarse)

    squared_norm = spectral_radius(lambda image: transpose(analyse(image)),
                                   wavelet_tube.shape[:2])
    tau = sigma = DEFAULT_STEP_SAFETY / np.sqrt(squared_norm)

    image = np.clip(transpose(wavelet_tube), 0.0, None)

    # The dual is warm-started with one prox step at the initial iterate
    # rather than left at zero. From zero the first primal step is
    # ``x - tau * 0`` — the iterate does not move, the relative flux change is
    # exactly 0, and the convergence rule fires on a step that did nothing.
    # Costing one analysis to make the first step a real one is cheaper than a
    # special case in the stopping test, and it is the same ``prox`` the loop
    # applies.
    dual = _dual_prox(sigma * analyse(image), wavelet_tube, support_tube,
                      sigma)
    total_flux = float(np.sum(image))

    diverged = False
    first_norm_wr = None

    for _ in range(1, max_iter + 1):

        # Condat-Vu never forms a wavelet residual: the data term reaches it
        # through the dual variable, so this is a **diagnostic** for this arm
        # and not a step of the algorithm. It is what the divergence guard and
        # nothing else reads.
        step_norm_wr = float(np.linalg.norm(
            _residual(wavelet_tube, support_tube, image, filter, coarse)))
        if first_norm_wr is None:
            first_norm_wr = step_norm_wr

        previous = image
        image = np.clip(previous - tau * transpose(dual), 0.0, None)

        dual = _dual_prox(dual + sigma * analyse(2.0 * image - previous),
                          wavelet_tube, support_tube, sigma)

        old_total_flux = total_flux or 1.0
        total_flux = float(np.sum(image))
        diff_flux = abs(total_flux - old_total_flux) / abs(old_total_flux)

        if (_diverging(image, step_norm_wr, first_norm_wr,
                       DEFAULT_DIVERGENCE_FACTOR)
                or not np.all(np.isfinite(dual))):
            diverged = True
            break

        if diff_flux < sigma_flux:
            break

    return _finish(wavelet_tube, support_tube, image, filter, coarse,
                   diverged, first_norm_wr)


# -- the door ------------------------------------------------------------


def _settle_axis(minimizer: str, axis: str, value: str | None) -> str | None:
    """Refuse an axis the minimizer does not carry; default one it does.

    Parameters
    ----------
    minimizer : str
        One of :data:`~dawis.config.MINIMIZERS`, already checked.
    axis : str
        A key of :data:`_AXIS_DEFAULTS`.
    value : str or None
        What the caller named, or None for *whatever this minimizer takes*.

    Returns
    -------
    str or None
        The value to solve with, and None when the minimizer has no such axis.

    Raises
    ------
    dawis.exceptions.DawisValueError
        If the minimizer does not carry the axis and the caller named a value
        for it, or if the value names no branch the solve has.
    """

    carried = axis in MINIMIZER_AXES[minimizer]

    if not carried:
        if value is not None:
            # What the caller needs back is not this minimizer's own axes —
            # they did not ask about those — but which minimizers do carry the
            # one they asked for, because an axis set on the wrong arm is as
            # often the wrong arm as the wrong axis.
            carriers = [name for name, axes in MINIMIZER_AXES.items()
                        if axis in axes]
            raise DawisValueError(
                'the %s minimizer has no %s axis; the ones that carry it are '
                '%s -->' % (minimizer, axis, carriers), value, None)
        return None

    if value is None:
        return _AXIS_DEFAULTS[axis]

    if value not in _AXIS_VALUES[axis]:
        raise DawisValueError(
            '%s names a branch the solve does not have -->' % axis,
            value, list(_AXIS_VALUES[axis]))

    return value


def minimize(
    tree: InterscaleTree,
    wavelet_cube: WaveletCube,
    label_cube: LabelCube,
    filter: np.ndarray,
    minimizer: str = 'bijaoui_rue',
    synthesis_operator: str | None = None,
    step_size: str | None = None,
    line_direction: str | None = None,
    sigma_flux: float = 1E-3,
    max_iter: int = DEFAULT_CG_MAX_ITER,
    verbose: bool = False,
    coarse_cube: CoarseCube | None = None,
) -> tuple[np.ndarray, float, float]:
    """Synthesise a tree into an image with the named minimizer.

    The minimizer axis. ``'bijaoui_rue'`` is
    :meth:`~dawis.trees.construction.InterscaleTree.CG_minimization`, called
    with its own signature and its own published defaults, so this door adds
    no layer to the path a run already takes. The other three are the proximal
    arms in this module.

    This is the one entry point :class:`~dawis.config.Config` reaches. Since
    #104 :func:`~dawis.restoration.synthesis.restore_object` calls it for
    every tree of every iteration, with the six axes the run was configured
    with, so a stock run comes through here.

    Parameters
    ----------
    tree : dawis.trees.construction.InterscaleTree
        The tree to synthesise.
    wavelet_cube : dawis.cubes.WaveletCube
        Wavelet planes the tree was built on.
    label_cube : dawis.cubes.LabelCube
        Label cube the tree was built on.
    filter : numpy.ndarray
        1D low-band filter of the à trous transform, normally B3-spline.
    minimizer : str, optional
        One of :data:`~dawis.config.MINIMIZERS`. Default ``'bijaoui_rue'``,
        the published conjugate gradient. The default that **moves** is
        #104's and :class:`~dawis.config.Config`'s business, not this door's.
    synthesis_operator : str or None, optional
        Which operator rebuilds an image from a tube —
        :data:`~dawis.config.SYNTHESIS_OPERATORS`. ``None``, the default,
        takes ``'ADJOINT'``, the published one. **Refused** on
        ``'condat_vu'``, which carries no operator axis.
    step_size : str or None, optional
        Formula for the conjugate direction coefficient —
        :data:`~dawis.config.CG_STEP_SIZES`. ``None``, the default, takes the
        published ``'FR'``. **Refused** on every minimizer but
        ``'bijaoui_rue'``: no proximal arm has a conjugate direction.
    line_direction : str or None, optional
        Which reading of the conjugate direction recursion to take —
        :data:`~dawis.config.CG_LINE_DIRECTIONS`. ``None``, the default, takes
        the published ``'as_written'``. Refused like ``step_size``.
    sigma_flux : float, optional
        Relative flux change below which the descent stops. Default 1E-3, the
        published value, taken by all four minimizers so a cost comparison is
        between algorithms rather than between tolerances.
    max_iter : int, optional
        Iteration cap. Default :data:`~dawis.config.DEFAULT_CG_MAX_ITER`.
    verbose : bool, optional
        Print the convergence line when the flux test fires. Read by
        ``'bijaoui_rue'`` alone — the proximal arms log nothing. Default
        False.
    coarse_cube : dawis.cubes.CoarseCube, optional
        Smooth planes of the same decomposition. Given one, the tube carries
        the tree's coarse term :math:`c_{L+1}` as a last channel and every
        minimizer reconstructs it alongside the wavelet planes (#88, #89).
        Default None.

    Returns
    -------
    reconstructed : numpy.ndarray
        The synthesised image of the tree, clipped at zero — the atom. Zeroed
        when the solve diverged.
    sum_wr : float
        Sum of the final wavelet residual over the support of the tree, or
        ``nan`` when the solve diverged.
    norm_wr : float
        Euclidean norm of that residual, or ``nan`` when it diverged.

    Raises
    ------
    dawis.exceptions.DawisValueError
        If ``minimizer`` names an algorithm this module does not have, if an
        axis is set on a minimizer that does not carry it, or if an axis value
        names no branch the solve has.

    Notes
    -----
    A proximal solve can diverge without ever leaving the finite numbers — see
    :data:`DEFAULT_DIVERGENCE_FACTOR` — so divergence is reported as a zeroed
    atom and NaN residuals rather than as an exception, which is what the
    research arm does and what the port has to reproduce.

    **Two hazards rode on that, and both are settled.** Both were inherited
    from ``restoration_tests/arms/proximal.py`` and reproduced bit for bit
    here, because a port that repaired them would have stopped matching the
    implementation every #74, #81 and #92 number was taken with:

    * **The NaN residuals were the only signal, and nothing read them.**
      Settled at #104. A zeroed atom is finite, so
      :func:`~dawis.restoration.synthesis.restore_object`'s
      ``np.isnan(np.sum(image))`` did not fire and ``flag_convergence``
      stayed ``True``: a diverged solve was recorded as a converged one. That
      function tests ``norm_wr`` as well now, which moves no number — the
      image it zeroes is already zero — and agrees with the rule
      ``restoration_tests/arms/base.py`` applies to the same solve.
    * **The growth test lagged one step, and never saw the iterate
      returned.** Settled at #110, and unlike the first this one is a
      **departure** from the reference rather than an agreement with it. The
      residual the loop compares is the one measured before the step, so
      iterate *k* was only scored at iteration *k + 1* and a loop ending on
      ``max_iter`` or on the flux test had no iteration *k + 1*.
      :func:`_finish` now scores the iterate it returns, where that residual
      is computed for ``norm_wr`` anyway, and the two together leave no
      iterate untested. ``Config.cg_max_iter`` is a user knob this repo has
      benchmarked at 3, so a low cap was not hypothetical.

      The departure is narrow in both directions, and
      ``.research/110-divergence-guard/`` is the measurement: it can only turn
      a returned atom into a flagged one, it never moves the pixels of an atom
      that is still returned, and on all 432 of #81's undiverged rows the
      returned residual is at most 1.214 times the one the solve started
      from — three orders of magnitude below the ``1e3`` that flags, so no
      #81, #92 or #94 verdict moves.
    """

    if minimizer not in MINIMIZERS:
        raise DawisValueError(
            'minimizer names the algorithm that solves a tree -->',
            minimizer, list(MINIMIZERS))

    synthesis_operator = _settle_axis(minimizer, 'synthesis_operator',
                                      synthesis_operator)
    step_size = _settle_axis(minimizer, 'step_size', step_size)
    line_direction = _settle_axis(minimizer, 'line_direction', line_direction)

    if minimizer == 'bijaoui_rue':
        return tree.CG_minimization(
            wavelet_cube, label_cube, filter = filter,
            synthesis_operator = synthesis_operator, step_size = step_size,
            line_direction = line_direction, sigma_flux = sigma_flux,
            max_iter = max_iter, verbose = verbose,
            coarse_cube = coarse_cube)

    coarse = coarse_cube is not None
    wavelet_tube, support_tube, _ = tree.tubes(wavelet_cube, label_cube,
                                               coarse_cube = coarse_cube)

    if minimizer == 'condat_vu':
        return _condat_vu(wavelet_tube, support_tube, filter, coarse,
                          sigma_flux, max_iter)

    return _forward_backward(tree, wavelet_tube, support_tube,
                             synthesis_operator, filter, coarse,
                             minimizer == 'fista', sigma_flux, max_iter)
