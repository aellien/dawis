"""Placeholders for optional third-party dependencies.

``matplotlib`` and ``ray`` are declared as the optional ``[viz]`` and ``[ray]``
extras, but several core modules import them at module scope. Rather than let a
missing extra break ``import dawis``, those modules fall back to the stubs
defined here. The stub imports cleanly and raises a helpful ``ImportError``
only if the absent library is actually used.

These stubs are scaffolding for the 1.0 refactor: once all matplotlib code
lives in ``dawis.viz`` and all ray code behind the ``dawis.parallel`` seam,
this module can be deleted.
"""

__all__ = ['MissingDependency', 'MissingRay']


class MissingDependency:
    """Stand-in for a third-party module that is not installed.

    Any attribute access or call raises ``ImportError`` naming the extra that
    provides the missing library.

    Parameters
    ----------
    name : str
        Import name of the absent library, e.g. ``'matplotlib'``.
    extra : str
        Name of the dawis extra that installs it, e.g. ``'viz'``.
    """

    def __init__(self, name, extra):
        self._name = name
        self._extra = extra

    def _fail(self, *args, **kwargs):
        raise ImportError(
            "%s is required for this feature but is not installed. "
            "Install it with: pip install 'dawis[%s]'" % (self._name, self._extra)
        )

    def __getattr__(self, attr):
        self._fail()

    def __call__(self, *args, **kwargs):
        self._fail()

    def __repr__(self):
        return "<MissingDependency %s (extra: %s)>" % (self._name, self._extra)


class MissingRay(MissingDependency):
    """Stand-in for ``ray`` that keeps ``@ray.remote`` importable.

    The decorator returns the undecorated function, so the plain serial call
    still works. Only the ``.remote()`` call, which needs a live ray cluster,
    raises ``ImportError``.
    """

    def __init__(self):
        super().__init__('ray', 'ray')

    def remote(self, *args, **kwargs):
        def decorate(function):
            function.remote = self._fail
            return function

        # Support both @ray.remote and @ray.remote(...).
        if len(args) == 1 and not kwargs and callable(args[0]):
            return decorate(args[0])
        return decorate
