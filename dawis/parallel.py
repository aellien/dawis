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

Which backend runs
------------------

**Serial is the default, and asking for one worker always gets it.** A caller
that names no backend and asks for no more than one worker runs in this
process, and that is what the golden tests and every single-worker
configuration get.

Asking for more than one worker is asking for parallelism, and there is only
one thing that can currently answer: a caller that wants several workers and
names no backend gets :data:`DEFAULT_PARALLEL_BACKEND`. That is a deliberate
decision and not an oversight — it is what the pipeline did with
``n_cpus`` before the seam existed, and preserving it is what keeps this
ticket a move rather than a science change. #53 gives ``Config`` an explicit
backend knob and the guess goes away.

Two ways to use it
------------------

A block that has work to spread calls :func:`pmap` and passes the worker
count it was configured with::

    atoms = pmap(restore_patch, patches, shared=(cube, labels), n_workers=n)

The pipeline decides, once per run, what a worker *is*::

    with parallel_backend(n_workers=n_cpus):
        ...

Outside such a block a single-worker :func:`pmap` still runs serially and a
multi-worker one starts a backend for the length of the call, so a block
called on its own behaves the way it does inside the pipeline.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from types import ModuleType
from typing import Any, Protocol, runtime_checkable

__all__ = [
    'BACKENDS',
    'DEFAULT_PARALLEL_BACKEND',
    'Backend',
    'RayBackend',
    'SerialBackend',
    'active_backend',
    'make_backend',
    'parallel_backend',
    'pmap',
]

#: Backend :func:`make_backend` picks when more than one worker is asked for
#: and no backend is named. Serial remains what one worker gets, and what any
#: caller that names it gets. See *Which backend runs* above.
DEFAULT_PARALLEL_BACKEND = 'ray'


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

    Attributes
    ----------
    n_workers : int
        Always 1.
    """

    name = 'serial'

    def __init__(self, n_workers: int = 1):
        self.n_workers = 1

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
        return '<SerialBackend n_workers=%d>' % self.n_workers


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

    Parameters
    ----------
    n_workers : int, optional
        CPUs to give a cluster this backend starts. Ignored when attaching to
        a cluster that is already up. Default is 1.

    Attributes
    ----------
    n_workers : int
        As passed.

    Notes
    -----
    Shared arguments go into the object store once per :meth:`map` call rather
    than once per task, which is what makes distributing worthwhile: the cubes
    a patch reads are large and identical across the patch list.
    """

    name = 'ray'

    def __init__(self, n_workers: int = 1):
        self.n_workers = n_workers
        self._owns_cluster = False
        self._remote_functions: dict[Any, Any] = {}

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
            ray.init(num_cpus=self.n_workers)
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
        return '<RayBackend n_workers=%d>' % self.n_workers


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


def make_backend(name: str | None = None, n_workers: int = 1) -> Backend:
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
    return backend_class(n_workers)


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

    Parameters
    ----------
    name : str or None, optional
        Backend to activate, or None to pick one from ``n_workers``. See
        :func:`make_backend`. Default is None.
    n_workers : int, optional
        Workers to run with. Default is 1, which selects the serial backend.

    Attributes
    ----------
    backend : Backend
        The backend this block activates.
    """

    def __init__(self, name: str | None = None, n_workers: int = 1):
        self.backend = make_backend(name, n_workers)
        self._previous: Backend | None = None
        self._entered = False

    def start(self) -> Backend:
        """Acquire the backend's resources and make it the active one."""
        global _ACTIVE
        self.backend.start()
        self._previous = _ACTIVE
        self._entered = True
        _ACTIVE = self.backend
        return self.backend

    def stop(self) -> None:
        """Restore the previous backend and release this one's resources."""
        global _ACTIVE
        if not self._entered:
            return
        self._entered = False
        _ACTIVE = self._previous
        self._previous = None
        self.backend.stop()

    def __enter__(self) -> Backend:
        return self.start()

    def __exit__(self, *exception: Any) -> bool:
        self.stop()
        return False


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
