"""Tier 0 checks for the parallelism seam (#47).

The seam is a single function, :func:`dawis.parallel.pmap`, plus the two
backends behind it. What is asserted here is the contract the science blocks
rely on — that a batch is called with its shared arguments appended, that
results come back in the order the batches were given, and that the serial
backend is what runs unless something asks for otherwise.

The ray adapter is exercised against a fake ``ray`` module installed in
``sys.modules``. Ray is the optional ``[ray]`` extra and is not installed in
the test environment; the fake lets the adapter's init, put, submit, get and
shutdown sequence be asserted anyway, which is the part of it that has any
logic in it.

The last two checks are the architectural ones the ticket asks for: nothing
outside :mod:`dawis.parallel` names ``ray``, and the package imports with ray
absent.
"""

from __future__ import annotations

import builtins
import importlib
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

from dawis import parallel


# ---------------------------------------------------------------------------
# pmap, on the serial backend
# ---------------------------------------------------------------------------


def _sum_batch(batch, offset, scale):
    """A stand-in science kernel: one batch, two shared arguments."""
    return [(value + offset) * scale for value in batch]


def test_pmap_calls_the_function_once_per_batch_with_the_shared_arguments():
    result = parallel.pmap(_sum_batch, [[1, 2], [3]], shared=(10, 2))
    assert result == [[22, 24], [26]]


def test_pmap_returns_results_in_the_order_the_batches_were_given():
    order = parallel.pmap(lambda batch: batch, [['a'], ['b'], ['c']])
    assert order == [['a'], ['b'], ['c']]


def test_pmap_accepts_an_iterator_of_batches():
    result = parallel.pmap(lambda batch: sum(batch), iter([[1], [2]]))
    assert result == [1, 2]


def test_pmap_of_nothing_is_nothing():
    assert parallel.pmap(lambda batch: 1 / 0, []) == []


def test_pmap_runs_in_this_process_by_default():
    """Serial is the default: no worker, no cluster, no ray import."""
    sys.modules.pop('ray', None)
    marker = []
    parallel.pmap(lambda batch: marker.append(batch), [[1]])
    assert marker == [[1]]
    assert 'ray' not in sys.modules


def test_pmap_stays_serial_for_one_worker_even_under_an_active_ray_backend(fake_ray):
    """A block's own serial gate wins over whatever the pipeline activated."""
    with parallel.parallel_backend('ray', n_workers=4):
        parallel.pmap(lambda batch: batch, [[1]], n_workers=1)
    assert [name for name, _ in fake_ray.calls if name == 'submit'] == []


# ---------------------------------------------------------------------------
# Choosing a backend
# ---------------------------------------------------------------------------


def test_serial_is_the_default_backend():
    assert parallel.make_backend().name == 'serial'
    assert parallel.make_backend(n_workers=1).name == 'serial'


def test_more_than_one_worker_asks_for_the_default_parallel_backend():
    assert parallel.DEFAULT_PARALLEL_BACKEND == 'ray'
    assert parallel.make_backend(n_workers=4).name == 'ray'


def test_a_named_backend_is_honoured_whatever_the_worker_count():
    assert parallel.make_backend('serial', n_workers=8).name == 'serial'
    assert parallel.make_backend('ray', n_workers=1).name == 'ray'


def test_an_unknown_backend_name_is_refused_by_name():
    with pytest.raises(ValueError, match='dask'):
        parallel.make_backend('dask')


def test_no_backend_is_active_outside_a_parallel_backend_block():
    assert parallel.active_backend() is None
    with parallel.parallel_backend('serial') as backend:
        assert parallel.active_backend() is backend
    assert parallel.active_backend() is None


# ---------------------------------------------------------------------------
# The ray adapter
# ---------------------------------------------------------------------------


class _FakeRef:
    """Stand-in for ``ray.ObjectRef``, holding the value it points at."""

    def __init__(self, value):
        self.value = value


def _install_fake_ray(monkeypatch=None):
    """Put a minimal fake ``ray`` in ``sys.modules`` and return it.

    Records every call the adapter makes as ``(name, payload)`` on
    ``module.calls``, so the init-put-submit-get-shutdown sequence can be
    asserted without a cluster.
    """
    module = types.ModuleType('ray')
    module.calls = []
    state = {'up': False}

    def init(**kwargs):
        state['up'] = True
        module.calls.append(('init', kwargs))

    def is_initialized():
        return state['up']

    def shutdown():
        state['up'] = False
        module.calls.append(('shutdown', None))

    def put(value):
        module.calls.append(('put', value))
        return _FakeRef(value)

    def get(refs):
        module.calls.append(('get', refs))
        if isinstance(refs, list):
            return [ref.value for ref in refs]
        return refs.value

    def remote(function):
        class _RemoteFunction:
            def options(self, **kwargs):
                module.calls.append(('options', kwargs))
                return self

            def remote(self, *args, **kwargs):
                module.calls.append(('submit', args))
                plain = [
                    arg.value if isinstance(arg, _FakeRef) else arg for arg in args
                ]
                return _FakeRef(function(*plain, **kwargs))

        return _RemoteFunction()

    module.init = init
    module.is_initialized = is_initialized
    module.shutdown = shutdown
    module.put = put
    module.get = get
    module.remote = remote
    sys.modules['ray'] = module
    return module


@pytest.fixture
def fake_ray():
    """A fake ``ray``, removed again however the test ends."""
    previous = sys.modules.get('ray')
    module = _install_fake_ray()
    yield module
    if previous is None:
        sys.modules.pop('ray', None)
    else:
        sys.modules['ray'] = previous


def test_the_ray_adapter_puts_the_shared_arguments_once_and_the_batches_never(fake_ray):
    with parallel.parallel_backend('ray', n_workers=2):
        parallel.pmap(_sum_batch, [[1], [2], [3]], shared=(10, 2), n_workers=2)

    put = [payload for name, payload in fake_ray.calls if name == 'put']
    assert put == [10, 2]


def test_the_ray_adapter_returns_what_the_serial_backend_would(fake_ray):
    batches = [[1, 2], [3], [4, 5, 6]]
    serial = parallel.pmap(_sum_batch, batches, shared=(10, 2))
    with parallel.parallel_backend('ray', n_workers=2):
        distributed = parallel.pmap(_sum_batch, batches, shared=(10, 2), n_workers=2)
    assert distributed == serial


def test_the_ray_adapter_starts_the_cluster_it_needs_and_stops_it_again(fake_ray):
    with parallel.parallel_backend('ray', n_workers=3):
        assert ('init', {'num_cpus': 3}) in fake_ray.calls
    assert ('shutdown', None) in fake_ray.calls


def test_the_ray_adapter_leaves_a_cluster_it_did_not_start_running(fake_ray):
    fake_ray.init(num_cpus=8)
    fake_ray.calls.clear()
    with parallel.parallel_backend('ray', n_workers=3):
        pass
    assert fake_ray.calls == []
    assert fake_ray.is_initialized()


@contextmanager
def ray_uninstalled():
    """Make ``import ray`` fail for the duration, as a clean install does."""
    real_import = builtins.__import__

    def refuse_ray(name, *args, **kwargs):
        if name == 'ray':
            raise ImportError("No module named 'ray'")
        return real_import(name, *args, **kwargs)

    previous = sys.modules.pop('ray', None)
    builtins.__import__ = refuse_ray
    try:
        yield
    finally:
        builtins.__import__ = real_import
        if previous is not None:
            sys.modules['ray'] = previous


def test_a_failing_block_still_releases_the_cluster(fake_ray):
    """What ``synthesis_by_analysis``'s try/finally relies on."""
    with pytest.raises(ZeroDivisionError):
        with parallel.parallel_backend('ray', n_workers=2):
            1 / 0
    assert ('shutdown', None) in fake_ray.calls
    assert parallel.active_backend() is None


def test_asking_for_ray_without_the_extra_says_how_to_install_it():
    with ray_uninstalled():
        with pytest.raises(ImportError, match=r"dawis\[ray\]"):
            parallel.parallel_backend('ray', n_workers=2).start()


# ---------------------------------------------------------------------------
# Threads, which are not workers (#61)
# ---------------------------------------------------------------------------
#
# What is asserted is the *effective* thread count the loaded library reports,
# never that an environment variable holds a string. The variable is the
# mechanism and the effective count is the behaviour, and #61 exists because a
# mechanism that looked set did nothing for a year.
#
# The one exception is the ray worker, which is a process that does not exist
# yet: there the environment IS the behaviour, because it is the only thing
# that reaches a process before its Python starts, and the fake cluster is all
# there is to observe.


def _blas_threads():
    """The effective thread count of the loaded BLAS, or None if there is none."""
    threads, _ = parallel.observed_threads()
    return threads


needs_a_threaded_blas = pytest.mark.skipif(
    _blas_threads() is None,
    reason="no threaded maths library is loaded, so there is no thread count "
           "to pin or observe",
)


@needs_a_threaded_blas
def test_a_block_threads_at_what_it_asked_for():
    """The knob's whole point: what was asked for is what the library gives."""
    with parallel.parallel_backend(n_workers=1, n_threads=1):
        assert _blas_threads() == 1


@needs_a_threaded_blas
def test_a_block_gives_the_caller_their_own_limits_back():
    """A run borrows the process's thread count; it does not keep it."""
    before = _blas_threads()
    with parallel.parallel_backend(n_workers=1, n_threads=1):
        pass
    assert _blas_threads() == before


@needs_a_threaded_blas
def test_a_failing_block_still_gives_the_caller_their_limits_back():
    """The run exits through an exception more often than anyone plans for."""
    before = _blas_threads()
    with pytest.raises(ZeroDivisionError):
        with parallel.parallel_backend(n_workers=1, n_threads=1):
            1 / 0
    assert _blas_threads() == before


@needs_a_threaded_blas
def test_naming_no_thread_count_leaves_the_process_alone():
    """``None`` means *touch nothing*, which is what keeps ``pmap`` unchanged."""
    before = _blas_threads()
    with parallel.parallel_backend(n_workers=1):
        assert _blas_threads() == before


@needs_a_threaded_blas
def test_a_block_that_could_not_start_leaves_no_pin_behind():
    """A cluster that refuses to come up must not leave this process pinned."""
    before = _blas_threads()
    with ray_uninstalled():
        with pytest.raises(ImportError):
            parallel.parallel_backend('ray', n_workers=2, n_threads=1).start()
    assert _blas_threads() == before


def test_every_thread_variable_is_named_and_none_is_a_synonym_for_another():
    """Four names, because a library prefers its own over the OpenMP one.

    OpenBLAS reads OPENBLAS_NUM_THREADS in preference to OMP_NUM_THREADS and
    MKL reads MKL_NUM_THREADS, so naming only one leaves a hole a
    contributor's shell can walk through — and the symptom is a spurious
    last-digit difference somewhere else entirely.
    """
    assert set(parallel.THREAD_ENVIRONMENT_VARIABLES) == {
        'OMP_NUM_THREADS',
        'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS',
        'NUMEXPR_NUM_THREADS',
    }
    assert set(parallel.thread_environment(3).values()) == {'3'}


def test_a_worker_is_given_the_thread_count_before_its_python_starts(fake_ray):
    """The ray worker's route, which is the environment and not threadpoolctl.

    A worker is a separate process, so the variables land before it imports
    numpy — the one moment BLAS reads them. This is also what lifts ray's own
    default of one thread per worker.

    Declared twice on purpose: at ``init`` for a cluster this backend starts,
    and on the remote function for one it merely attaches to.
    """
    with parallel.parallel_backend('ray', n_workers=2, n_threads=4):
        parallel.pmap(_sum_batch, [[1], [2]], shared=(10, 2), n_workers=2)

    expected = {name: '4' for name in parallel.THREAD_ENVIRONMENT_VARIABLES}
    init, = [payload for name, payload in fake_ray.calls if name == 'init']
    assert init == {'num_cpus': 2, 'runtime_env': {'env_vars': expected}}

    options = [payload for name, payload in fake_ray.calls if name == 'options']
    assert options == [{'runtime_env': {'env_vars': expected}}]


def test_a_worker_is_left_alone_when_no_thread_count_was_asked_for(fake_ray):
    """``None`` declares no runtime environment, so ray keeps its own default."""
    with parallel.parallel_backend('ray', n_workers=2):
        parallel.pmap(_sum_batch, [[1], [2]], shared=(10, 2), n_workers=2)

    assert ('init', {'num_cpus': 2}) in fake_ray.calls
    assert [name for name, _ in fake_ray.calls if name == 'options'] == []


def test_the_two_thread_observers_answer_alike():
    """The package's and the harness's copies must not drift apart.

    ``goldenrun.observed_threads_at`` exists because the recording scripts ask
    this of a ``legacy-seeded`` checkout, which predates
    :func:`dawis.parallel.observed_threads`. A golden's recorded observation is
    then compared against runs that report through the other copy, so a rule
    changed on one side alone would make the recording describe something
    else — silently, since both still return a plausible number.
    """
    import sys
    from pathlib import Path

    tests_dir = str(Path(__file__).resolve().parent.parent)
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    import goldenrun

    with parallel.parallel_backend(n_workers=1, n_threads=1):
        package = parallel.observed_threads()
    assert goldenrun.observed_threads_at(1) == package


def test_the_thread_count_does_not_choose_a_backend():
    """The two knobs are independent, which is the coupling #61 removed."""
    assert parallel.make_backend(None, 1, 8).name == 'serial'
    assert parallel.make_backend(None, 4, 1).name == parallel.DEFAULT_PARALLEL_BACKEND


def test_the_thread_count_is_per_process_and_not_a_budget_divided_up():
    """``n_threads`` is what *each* process may use, workers included.

    A total-budget design would make the thread count depend on the worker
    count again — the exact coupling being removed — so the same number
    reaches every worker whatever ``n_workers`` says.
    """
    for n_workers in (1, 2, 16):
        backend = parallel.make_backend('ray', n_workers, 4)
        assert backend.n_threads == 4
        assert backend._runtime_env() == {
            'env_vars': parallel.thread_environment(4)
        }


# ---------------------------------------------------------------------------
# The architectural rule the seam exists to make true
# ---------------------------------------------------------------------------


# The static half of the rule — that no module outside this one imports ray —
# moved to `tests/unit/test_import_graph.py` with #58, which walks the source
# tree once for all three of the architecture's dependency rules. What is left
# here is the behaviour that rule buys: every module of the package really does
# import with the [ray] extra absent.
def test_the_package_imports_with_ray_uninstalled():
    """Every module of the package, not only the top-level namespace."""
    package_root = Path(parallel.__file__).parent
    modules = [
        'dawis.' + '.'.join(path.relative_to(package_root).with_suffix('').parts)
        for path in sorted(package_root.rglob('*.py'))
        if path.name != '__init__.py'
    ]
    # dawis.viz needs matplotlib, the [viz] extra, which is a different
    # question from this one.
    modules = [name for name in modules if not name.startswith('dawis.viz')]

    with ray_uninstalled():
        for name in ['dawis'] + modules:
            importlib.import_module(name)


# ---------------------------------------------------------------------------
# The blocks that hand work to the seam
# ---------------------------------------------------------------------------
#
# Cutting a list into patches is the one thing a block does that the seam
# cannot check for it, and getting it wrong loses or duplicates structures
# silently. These drive the tree block's cut with a stand-in connectivity
# function, so what is asserted is the batching alone.


class _FakeRegion:
    """The three attributes ``make_interscale_trees`` reads off a region."""

    def __init__(self, index):
        self.index = index
        self.level = 3
        self.area = 10
        self.norm_max_intensity = 1000.0 - index


class _FakeLabelCube:
    z_size = 9


def _drive_tree_batching(monkeypatch, n_roots, n_workers, size_patch):
    """Run the root-patching of ``make_interscale_trees`` and report the cut."""
    from dawis.trees import deblending

    seen = []

    def record(batch, *shared):
        seen.append((list(batch), shared))
        return list(batch)

    monkeypatch.setattr(deblending, 'interscale_connectivity', record)

    roots = [_FakeRegion(index) for index in range(n_roots)]
    trees = deblending.make_interscale_trees(
        region_list=roots,
        wavelet_cube='wavelet-cube',
        label_cube=_FakeLabelCube(),
        det_err_array='errors',
        tau=0.0,
        size_patch=size_patch,
        n_workers=n_workers,
    )
    return seen, trees


def test_one_worker_hands_the_roots_over_as_a_single_batch(monkeypatch):
    seen, trees = _drive_tree_batching(monkeypatch, n_roots=250, n_workers=1, size_patch=100)
    assert len(seen) == 1
    assert len(trees) == 250


def test_many_workers_cut_the_roots_into_patches_that_rejoin_unchanged(monkeypatch):
    serial, serial_trees = _drive_tree_batching(
        monkeypatch, n_roots=250, n_workers=1, size_patch=100
    )
    _install_fake_ray()
    try:
        cut, cut_trees = _drive_tree_batching(
            monkeypatch, n_roots=250, n_workers=4, size_patch=100
        )
    finally:
        sys.modules.pop('ray', None)

    assert [len(batch) for batch, _ in cut] == [100, 100, 50]
    # Distinct objects across the two runs, so identity says nothing: the
    # roots carry an index precisely so order can be compared.
    assert [tree.index for tree in cut_trees] == [tree.index for tree in serial_trees]
    assert [region.index for region in serial[0][0]] == [
        region.index for batch, _ in cut for region in batch
    ]


def test_the_arguments_every_patch_shares_are_passed_once_each(monkeypatch):
    seen, _ = _drive_tree_batching(monkeypatch, n_roots=10, n_workers=1, size_patch=100)
    (_, shared), = seen
    # region list, wavelet cube, label cube, detection errors, root level, and
    # the four knobs the connectivity reads, plus verbose.
    assert len(shared) == 10
    assert shared[1] == 'wavelet-cube'
    assert shared[3] == 'errors'
    assert shared[4] == 3


def test_only_one_connectivity_function_survived_the_seam():
    """#47 collapsed the plain and ``@ray.remote`` copies into one function.

    The two names the pair answered to before the seam, ``_para`` and
    ``_serial``, lived on the flat ``dawis.make_interscale_trees``, which #50
    deleted. So the check is that neither address survives anywhere a caller
    can reach: not on the block, and not on the package.
    """
    import dawis
    import dawis.trees
    from dawis.trees.deblending import interscale_connectivity

    assert dawis.trees.interscale_connectivity is interscale_connectivity
    for retired in ('interscale_connectivity_para',
                    'interscale_connectivity_serial'):
        assert not hasattr(dawis.trees, retired)
        assert not hasattr(dawis, retired)
