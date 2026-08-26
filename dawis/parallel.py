"""Parallelism: one map function, and the backends that can run it.

DAWIS is embarrassingly parallel in two places, and only two. Building the
interscale trees of one iteration is independent per root, and synthesising
those trees into atoms is independent per tree. Everything else in the loop is
sequential by construction, because each iteration analyses what the previous
one left behind.

This module is the seam between those two facts and *how* the work is spread.
The science blocks declare what is independent: they cut their work into
patches and hand the patches to :func:`pmap`, together with the arguments
every patch needs. Nothing in them knows whether a patch runs in this process
or on a worker across the machine. *Patch* is the word the blocks already use
— ``size_patch``, ``restore_patch`` — and it is the word used here, so one
concept does not acquire two names at the seam.

Two backends implement the other half.

:class:`SerialBackend` is a plain loop. It is the reference behaviour: any
other backend is correct exactly insofar as it returns what this one would.

:class:`RayBackend` is the adapter for the optional ``[ray]`` extra. It owns
every line of ray in DAWIS — the import, the cluster attach, ``ray.put`` of
the shared arguments, task submission, collection, and shutdown. It attaches
to a cluster that is already up and leaves it alone afterwards; it starts one
only when there is none, and then it is the one that stops it. Because the
import happens inside the adapter and not at module scope, ray is genuinely
optional: DAWIS imports and runs without it, and only asking for the backend
by name raises.

The process-pool and shared-memory backend of #57 drops in here as a third
entry in :data:`BACKENDS`, and touches no science code to do it.

Workers and threads
-------------------

Two kinds of parallelism run underneath DAWIS and they are not the same knob.
A **worker** is a process this module hands a whole patch of trees to. A
**thread** is what numpy and scipy split a *single* array operation across
inside one process, and OpenBLAS, MKL and OpenMP decide how many of them
there are. DAWIS never chose the second one, and that is what #61 fixes: a
sum split over four threads is added in a different association order from
one added on a single thread, so a thread count that moved with the machine —
or with the worker count — moved an atom's flux in its last digits.

The thread count therefore travels with the worker count, through
:class:`parallel_backend`, and it reaches the two kinds of process by two
different routes because they are two different moments in time.

* **This process.** BLAS reads its thread count once, when numpy first loads
  it, which is long before a configuration is parsed. Setting an environment
  variable from inside DAWIS is therefore always too late. ``threadpoolctl``
  talks to the already-loaded library instead, which is the only reason the
  knob can work at all. :class:`parallel_backend` applies it on entry and puts
  the caller's own limits back on exit, exception or not.

* **A ray worker.** A worker is a separate process that has not imported
  numpy yet, so there the environment variables *are* in time — they go into
  the cluster's runtime environment and land before the worker's Python
  starts. No ``threadpoolctl`` call happens inside a worker. This is also
  what lifts ray's own default, which forces every worker it starts to one
  thread whatever the caller asked for.

:data:`THREAD_ENVIRONMENT_VARIABLES` is four names and not one because they
are not synonyms: OpenBLAS prefers ``OPENBLAS_NUM_THREADS`` over the OpenMP
name and MKL prefers ``MKL_NUM_THREADS``, so naming only one leaves a hole a
contributor's shell can walk through.

Threads apply **per process** and do not divide a budget: ``n_threads = 8``
means each process may use eight threads. Dividing a total among workers would
make the thread count depend on the worker count again, which is the exact
coupling being removed. So keep ``n_workers × n_threads`` at or below the core
count — oversubscription costs time, never correctness.

Which backend runs
------------------

**Serial is the default, and asking for one worker always gets it.** A caller
that names no backend and asks for no more than one worker runs in this
process, and that is what the golden tests and every single-worker
configuration get.

Asking for more than one worker is asking for parallelism, and there is only
one thing that can currently answer: a caller that wants several workers and
names no backend gets :data:`DEFAULT_PARALLEL_BACKEND`. That is a deliberate
decision and not an oversight — it is what the pipeline did with its
worker count before the seam existed, and preserving it is what keeps this
ticket a move rather than a science change. :attr:`dawis.config.Config.backend`
names one outright when the guess is not wanted (#56).

Which backend runs is a scheduling decision and only that. A backend is
correct exactly insofar as it returns what :class:`SerialBackend` would, so
swapping one for another — or changing the worker count — cannot move a
result. Restoration is where that was not true before #56, and it is now.

Two ways to use it
------------------

A block that has work to spread calls :func:`pmap` and passes the worker
count it was configured with::

    atoms = pmap(restore_patch, patches, shared=(cube, labels), n_workers=n)

The pipeline decides, once per run, what a worker *is*::

    with parallel_backend(n_workers=config.n_workers):
        ...

Outside such a block a single-worker :func:`pmap` still runs serially and a
multi-worker one starts a backend for the length of the call, so a block
called on its own behaves the way it does inside the pipeline.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from types import ModuleType
from typing import Any, Protocol, runtime_checkable

from threadpoolctl import threadpool_info, threadpool_limits

__all__ = [
    'BACKENDS',
    'DEFAULT_PARALLEL_BACKEND',
    'THREAD_ENVIRONMENT_VARIABLES',
    'Backend',
    'RayBackend',
    'SerialBackend',
    'active_backend',
    'cut_into_patches',
    'make_backend',
    'observed_threads',
    'parallel_backend',
    'pmap',
    'thread_environment',
]

#: Backend :func:`make_backend` picks when more than one worker is asked for
#: and no backend is named. Serial remains what one worker gets, and what any
#: caller that names it gets. See *Which backend runs* above.
DEFAULT_PARALLEL_BACKEND = 'ray'

#: Every environment variable a maths library reads its thread count from.
#:
#: Four names and not one, because they are not synonyms — see *Workers and
#: threads* in the module docstring. Setting these from inside DAWIS only ever
#: reaches a process that has not imported numpy yet, which means a ray worker
#: and never this one; :func:`parallel_backend` uses ``threadpoolctl`` here.
THREAD_ENVIRONMENT_VARIABLES = (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
)


def thread_environment(n_threads: int) -> dict[str, str]:
    """The environment a process must start with to thread at ``n_threads``.

    Parameters
    ----------
    n_threads : int
        Threads each process may use. Applied per process — this is not a
        budget divided among them.

    Returns
    -------
    dict of str to str
        Every name in :data:`THREAD_ENVIRONMENT_VARIABLES`, all set to the
        same count.

    Examples
    --------
    >>> thread_environment(2)['OPENBLAS_NUM_THREADS']
    '2'
    """
    return {name: str(n_threads) for name in THREAD_ENVIRONMENT_VARIABLES}


def observed_threads() -> tuple[int | None, str | None]:
    """What the loaded maths library actually threads at, and which one it is.

    The *observation*, as against the request. A thread count DAWIS asked for
    and a thread count the library gave are two different facts, and the whole
    lesson of #61 is that they can disagree silently — so provenance records
    both, and this is the half that cannot be read off a configuration.

    Returns
    -------
    tuple of (int or None, str or None)
        The effective thread count of the loaded BLAS, and the vendor
        ``threadpoolctl`` found it to be — ``'openblas'``, ``'mkl'`` and so
        on. ``(None, None)`` if no threaded library was found at all, which is
        a real answer and not an error: a numpy built against a reference BLAS
        has no thread count to report.

    Notes
    -----
    BLAS is preferred over OpenMP where both are loaded, because BLAS is what
    sums the vectors whose association order moves an atom's flux.

    ``tests/goldenrun.py`` carries a copy of this, because the recording
    scripts ask it of a ``legacy-seeded`` checkout that predates this module.
    The two must answer alike or a recorded observation stops describing what
    a run reports; a test asserts they do.

    More than one BLAS can be loaded at once — numpy and scipy are built
    against different ones often enough. Reporting whichever
    ``threadpoolctl`` happened to list first would make the record silently
    one of them, so every vendor found is named and the *highest* thread count
    is reported. The highest is the answer that matters: one library still
    threading is enough to reassociate a sum, and a pin that missed one has
    not landed.
    """
    pools = threadpool_info()
    if not pools:
        return None, None
    blas = [pool for pool in pools if pool.get('user_api') == 'blas'] or pools
    counts = [pool['num_threads'] for pool in blas
              if pool.get('num_threads') is not None]
    vendors = sorted({pool['internal_api'] for pool in blas
                      if pool.get('internal_api')})
    return (max(counts) if counts else None), (', '.join(vendors) or None)


@runtime_checkable
class Backend(Protocol):
    """What :func:`pmap` needs of a backend.

    Three things, and the docstrings of this module lean on all three being
    interchangeable: a backend acquires whatever it needs in :meth:`start`,
    releases it in :meth:`stop`, and in between :meth:`map` returns for any
    input exactly what :class:`SerialBackend` would return.
    """

    #: The name a configuration selects this backend by.
    name: str

    def start(self) -> None:
        """Acquire whatever running a patch takes. Idempotent."""
        ...

    def stop(self) -> None:
        """Release whatever :meth:`start` acquired, and nothing else."""
        ...

    def map(
        self,
        function: Callable[..., Any],
        patches: Sequence[Any],
        shared: Sequence[Any],
    ) -> list[Any]:
        """Call ``function(patch, *shared)`` per patch, results in patch order."""
        ...


class SerialBackend:
    """Run every patch in this process, one after another.

    What one worker gets, what the golden tests run, and the behaviour every
    other backend is measured against.

    Parameters
    ----------
    n_workers : int, optional
        Accepted so every backend is constructed the same way. A serial run
        has one worker whatever is asked for, so this is recorded as 1.
        Default is 1.
    n_threads : int or None, optional
        Accepted so every backend is constructed the same way, and unused: a
        serial patch runs in this process, whose thread count
        :class:`parallel_backend` has already set. Default is None.

    Attributes
    ----------
    n_workers : int
        Always 1.
    n_threads : int or None
        As passed.
    """

    name = 'serial'

    def __init__(self, n_workers: int = 1, n_threads: int | None = None):
        self.n_workers = 1
        self.n_threads = n_threads

    def start(self) -> None:
        """Acquire nothing. Present so the backends are interchangeable."""

    def stop(self) -> None:
        """Release nothing. Present so the backends are interchangeable."""

    def map(
        self,
        function: Callable[..., Any],
        patches: Sequence[Any],
        shared: Sequence[Any],
    ) -> list[Any]:
        """Call ``function(patch, *shared)`` for each patch, in order."""
        return [function(patch, *shared) for patch in patches]

    def __repr__(self) -> str:
        return ('<SerialBackend n_workers=%d n_threads=%s>'
                % (self.n_workers, self.n_threads))


class RayBackend:
    """Run patches as ray tasks. The whole of DAWIS's ray usage.

    Ray is imported the first time the backend is started or used, never at
    module scope, so an installation without the ``[ray]`` extra imports and
    runs DAWIS normally and only pays for the absence if it asks for this
    backend by name.

    Cluster ownership follows one rule: attach to what is running, start what
    is not, and stop only what this instance started. A run inside a job that
    already has a head node therefore leaves that node alone, and a run that
    stood its own cluster up tears it down again.

    A worker is a separate process, so it is the one place in DAWIS where
    setting :data:`THREAD_ENVIRONMENT_VARIABLES` is in time to be read: the
    variables go into the runtime environment and land before the worker's
    Python starts, so BLAS reads them at import. That is also what lifts ray's
    own default, which pins every worker it starts to a single thread whatever
    the caller asked for. No ``threadpoolctl`` call happens inside a worker.

    The runtime environment is declared twice, on purpose. Naming it at
    ``ray.init`` covers a cluster this backend starts; naming it again on the
    remote function covers one it merely attaches to, where the job's
    environment was settled by whoever stood the cluster up. Both carry the
    same four values, so the two declarations cannot disagree.

    Parameters
    ----------
    n_workers : int, optional
        CPUs to give a cluster this backend starts. Ignored when attaching to
        a cluster that is already up. Default is 1.
    n_threads : int or None, optional
        Threads each worker may use. ``None``, the default, leaves ray's own
        behaviour alone. Per process, not a budget divided among the workers.
        Default is None.

    Attributes
    ----------
    n_workers : int
        As passed.
    n_threads : int or None
        As passed.

    Notes
    -----
    Shared arguments go into the object store once per :meth:`map` call rather
    than once per task, which is what makes distributing worthwhile: the cubes
    a patch reads are large and identical across the patch list.
    """

    name = 'ray'

    def __init__(self, n_workers: int = 1, n_threads: int | None = None):
        self.n_workers = n_workers
        self.n_threads = n_threads
        self._owns_cluster = False
        self._remote_functions: dict[Any, Any] = {}

    def _runtime_env(self) -> dict[str, Any] | None:
        """The runtime environment a worker must start with, or None."""
        if self.n_threads is None:
            return None
        return {'env_vars': thread_environment(self.n_threads)}

    @staticmethod
    def _import_ray() -> ModuleType:
        """Import ray, or explain which extra provides it."""
        try:
            import ray
        except ImportError as error:
            raise ImportError(
                "the 'ray' parallel backend needs ray, which is not installed. "
                "Install it with: pip install 'dawis[ray]'"
            ) from error
        return ray

    def start(self) -> None:
        """Attach to a running cluster, or start one of ``n_workers`` CPUs."""
        ray = self._import_ray()
        if not ray.is_initialized():
            runtime_env = self._runtime_env()
            if runtime_env is None:
                ray.init(num_cpus=self.n_workers)
            else:
                ray.init(num_cpus=self.n_workers, runtime_env=runtime_env)
            self._owns_cluster = True

    def stop(self) -> None:
        """Shut the cluster down, if and only if this backend started it."""
        if not self._owns_cluster:
            return
        self._owns_cluster = False
        self._import_ray().shutdown()

    def _remote(self, function: Callable[..., Any]) -> Any:
        """The ``@ray.remote`` wrapper of ``function``, made once and kept."""
        remote = self._remote_functions.get(function)
        if remote is None:
            remote = self._import_ray().remote(function)
            runtime_env = self._runtime_env()
            if runtime_env is not None:
                remote = remote.options(runtime_env=runtime_env)
            self._remote_functions[function] = remote
        return remote

    def map(
        self,
        function: Callable[..., Any],
        patches: Sequence[Any],
        shared: Sequence[Any],
    ) -> list[Any]:
        """Submit one task per patch and collect them in patch order."""
        ray = self._import_ray()
        self.start()
        remote = self._remote(function)
        shared_refs = [ray.put(argument) for argument in shared]
        pending = [remote.remote(patch, *shared_refs) for patch in patches]
        return list(ray.get(pending))

    def __repr__(self) -> str:
        return ('<RayBackend n_workers=%d n_threads=%s>'
                % (self.n_workers, self.n_threads))


#: Every backend the seam knows, by the name a configuration uses.
BACKENDS: dict[str, type[Backend]] = {
    SerialBackend.name: SerialBackend,
    RayBackend.name: RayBackend,
}

#: The backend activated by the innermost enclosing :class:`parallel_backend`,
#: or None. Read it through :func:`active_backend`. A plain module global, not
#: a context variable: DAWIS runs one pipeline per process and spreads work by
#: sending it away, never by running the loop itself on several threads.
_ACTIVE: Backend | None = None


def make_backend(name: str | None = None, n_workers: int = 1,
                 n_threads: int | None = None) -> Backend:
    """Build a backend by name, or pick one for a worker count.

    Parameters
    ----------
    name : str or None, optional
        ``'serial'``, ``'ray'``, or None to choose from ``n_workers``. Default
        is None.
    n_workers : int, optional
        Workers the caller wants. Only consulted when ``name`` is None, where
        one worker means serial and more than one means
        :data:`DEFAULT_PARALLEL_BACKEND` — see *Which backend runs* in the
        module docstring. Default is 1.
    n_threads : int or None, optional
        Threads each worker may use, or None to leave the backend's own
        default alone. Never consulted in choosing a backend: which backend
        runs is a worker question, and the two knobs are independent by
        construction (#61). Default is None.

    Returns
    -------
    Backend
        A backend that has not been started.

    Raises
    ------
    ValueError
        If ``name`` is not a known backend.
    """
    if name is None:
        name = SerialBackend.name if n_workers <= 1 else DEFAULT_PARALLEL_BACKEND
    try:
        backend_class = BACKENDS[name]
    except KeyError:
        raise ValueError(
            'unknown parallel backend %r. Known backends: %s'
            % (name, ', '.join(sorted(BACKENDS)))
        ) from None
    return backend_class(n_workers, n_threads)


def active_backend() -> Backend | None:
    """The backend of the innermost enclosing block, or None if there is none."""
    return _ACTIVE


class parallel_backend:
    """Make a backend the one :func:`pmap` uses for the enclosing block.

    Used as a context manager by preference::

        with parallel_backend(n_workers=4):
            ...

    :meth:`start` and :meth:`stop` are also public, because
    ``dawis.run`` sets the backend up at the top of a long
    function and tears it down at the bottom. That call site becomes a plain
    ``with`` when the pipeline is carved (#49).

    Named in lower case, unlike :class:`SerialBackend` and :class:`RayBackend`
    beside it, because it is only ever *called* — the name appears in a
    ``with`` statement, never as a type.

    Blocks nest: the previous backend is restored on exit rather than cleared,
    so an inner block cannot silently turn an outer one off.

    It is also where a run's **thread** count is applied, because this is
    already the one object that knows how a run is spread and #61 wanted no
    second seam for the second kind of parallelism. The two knobs reach their
    two kinds of process by two routes — ``threadpoolctl`` here, environment
    variables on a ray worker — and *Workers and threads* in the module
    docstring is the argument for why. The caller's own limits are put back on
    the way out, including out through an exception.

    Parameters
    ----------
    name : str or None, optional
        Backend to activate, or None to pick one from ``n_workers``. See
        :func:`make_backend`. Default is None.
    n_workers : int, optional
        Workers to run with. Default is 1, which selects the serial backend.
    n_threads : int or None, optional
        Threads each process — this one and every worker — may use. ``None``,
        the default, touches nothing and leaves whatever the caller's
        environment already decided. Per process, not a budget divided among
        the workers, so keep ``n_workers × n_threads`` at or below the core
        count. Default is None.

    Attributes
    ----------
    backend : Backend
        The backend this block activates.
    n_threads : int or None
        As passed.
    """

    def __init__(self, name: str | None = None, n_workers: int = 1,
                 n_threads: int | None = None):
        self.backend = make_backend(name, n_workers, n_threads)
        self.n_threads = n_threads
        self._previous: Backend | None = None
        self._entered = False
        self._limits: Any = None

    def start(self) -> Backend:
        """Acquire the backend's resources and make it the active one."""
        global _ACTIVE
        if self.n_threads is not None:
            self._limits = threadpool_limits(limits=self.n_threads)
        started = False
        try:
            self.backend.start()
            started = True
        finally:
            # A cluster that would not start must not leave this process
            # pinned. `stop` cannot undo it, because the block was never
            # entered, so the only place left to undo it is here.
            if not started:
                self._restore_threads()
        self._previous = _ACTIVE
        self._entered = True
        _ACTIVE = self.backend
        return self.backend

    def stop(self) -> None:
        """Restore the previous backend and release this one's resources."""
        global _ACTIVE
        if not self._entered:
            self._restore_threads()
            return
        self._entered = False
        _ACTIVE = self._previous
        self._previous = None
        try:
            self.backend.stop()
        finally:
            # `finally`, so a backend that fails to shut down still hands the
            # caller their own thread limits back.
            self._restore_threads()

    def _restore_threads(self) -> None:
        """Put the caller's thread limits back, once and only once."""
        limits, self._limits = self._limits, None
        if limits is not None:
            limits.restore_original_limits()

    def __enter__(self) -> Backend:
        return self.start()

    def __exit__(self, *exception: Any) -> bool:
        self.stop()
        return False


def cut_into_patches(work: Sequence[Any], size_patch: int) -> list[list[Any]]:
    """Cut a sequence of independent items into consecutive patches.

    The other half of what a block does before it calls :func:`pmap`: the
    block decides *what* is independent, this decides how many of them travel
    together. Both science blocks that spread work cut it this way, so the
    cut lives here rather than twice over in them.

    Parameters
    ----------
    work : sequence
        Independent items, in the order they should be worked.
    size_patch : int
        Items per patch. The last patch is whatever is left and may be
        shorter. Values below one are read as one — a patch of no items is
        not a unit of work, and rounding up is what stops the knob losing an
        item or asking :func:`range` for a step of zero.

    Returns
    -------
    list of list
        The patches, covering ``work`` exactly once and in order. An empty
        ``work`` gives no patches at all, which is what makes an iteration
        that found nothing cost nothing.

    Examples
    --------
    >>> cut_into_patches([1, 2, 3, 4, 5], 2)
    [[1, 2], [3, 4], [5]]
    >>> cut_into_patches([], 2)
    []
    """
    size_patch = max(size_patch, 1)
    return [list(work[start : start + size_patch])
            for start in range(0, len(work), size_patch)]


def pmap(
    function: Callable[..., Any],
    patches: Iterable[Any],
    shared: Sequence[Any] = (),
    n_workers: int = 1,
) -> list[Any]:
    """Map ``function`` over ``patches``, here or on workers.

    The one function the science blocks call. Each patch becomes one call,
    ``function(patch, *shared)``, and the results come back as a list in the
    order the patches were given whatever backend ran them.

    ``shared`` exists because the patches of one call read the same large
    cubes. A backend that has to move data puts those arguments across once
    for the whole call; the serial backend simply passes them through. Split
    the arguments accordingly: what differs per patch goes in ``patches``,
    what does not goes in ``shared``.

    Parameters
    ----------
    function : callable
        Called as ``function(patch, *shared)``. Must be importable by name on
        a worker — a module-level function, not a closure or a lambda — for
        any backend other than serial.
    patches : iterable
        The independent units of work. Consumed once, in order.
    shared : sequence, optional
        Arguments appended to every call. Default is empty.
    n_workers : int, optional
        Workers this particular map may use. **1 forces the serial backend**,
        whatever is active, so a block's own serial gate is always honoured.
        Default is 1.

    Returns
    -------
    list
        One result per patch, in patch order.

    Notes
    -----
    With no backend active and more than one worker asked for, a backend is
    started for the length of the call and stopped again, so a block called
    outside the pipeline behaves as it does inside it.

    A backend that moves data between processes copies ``shared`` on the way;
    the serial backend passes the same objects through. Nothing here enforces
    the difference, so ``function`` must not mutate what it is handed.
    """
    patches = list(patches)
    if not patches:
        return []

    if n_workers <= 1:
        return SerialBackend().map(function, patches, shared)

    if _ACTIVE is not None:
        return _ACTIVE.map(function, patches, shared)

    with parallel_backend(n_workers=n_workers) as backend:
        return backend.map(function, patches, shared)
