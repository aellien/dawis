"""Where a run's own words go, and what DAWIS refuses to do to get them there (#55).

Every module of the package logs through a module-level logger named for
itself, and nothing in the package prints. That is not a style preference: a
run of DAWIS is twenty minutes of a much longer program, and the caller — a
survey pipeline, a notebook, a batch script — owns the decision of where
output lands. A ``print`` takes that decision away, and so does a library that
calls :func:`logging.basicConfig`.

So the blocks emit and configure nothing at all. Their records travel up to
the ``dawis`` logger, and from there to whatever the caller installed.

The one exception, and its size
-------------------------------

If the blocks configure nothing and the caller configured nothing either, a
run says nothing — and the default experience of DAWIS is a twenty-minute
silence, in a notebook most of all. That is worse than the discourtesy it
avoids.

So :func:`run_logging`, entered by the top-level pipeline entry and nowhere
else, does exactly two things for the length of one run:

* attaches a file handler, so the run directory keeps the ``.log.txt`` beside
  the images that DAWIS has always written there;
* attaches a stdout handler **only when the caller has configured no logging
  at all** — no handler on the ``dawis`` logger and none on any ancestor it
  propagates to. When they have, their handlers already receive every record
  and a second stream would double every line.

Both come back off when the run ends, including out through an exception: a
campaign is many runs in one process, and handlers that accumulated would
write the second run into the first run's file.

The level follows the same rule. An unconfigured ``dawis`` logger inherits
the root's default of WARNING, which would drop the start banner and every
iteration line, so an unconfigured run is set to DEBUG and put back to
NOTSET afterwards. A caller who set a level set it on purpose — including
when the effect is a quieter log file — and it is not touched.

The root logger is never touched under any of these conditions.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager

__all__ = ['LINE_FORMAT', 'LOG_EXTENSION', 'PACKAGE_LOGGER', 'run_logging']

#: The logger every module of the package hangs off. Silencing this silences
#: DAWIS and nothing else; it is the one name a caller needs.
PACKAGE_LOGGER = 'dawis'

#: What the run's log file is called, hung off the run's output stem. The name
#: predates 1.0 and is kept so a run directory looks like it always did.
LOG_EXTENSION = '.log.txt'

#: How a line is stamped. This is :data:`logging.BASIC_FORMAT`, which is what
#: :func:`logging.basicConfig` gave the run log before #55 replaced it, so a
#: run directory's ``.log.txt`` reads as it always has. What changed is
#: ``%(name)s``: it used to say ``root`` on every line and now names the
#: module that spoke.
LINE_FORMAT = logging.BASIC_FORMAT


def _is_configured(logger: logging.Logger) -> bool:
    """Whether records from ``logger`` already reach a handler someone installed.

    The walk up the tree is the one :mod:`logging` itself does when it
    dispatches a record: each ancestor in turn, stopping at the first that
    does not propagate.

    Parameters
    ----------
    logger : logging.Logger
        The logger to ask about — in practice the ``dawis`` one.

    Returns
    -------
    bool
        True when some ancestor, or the logger itself, holds a handler.

    Notes
    -----
    :data:`logging.lastResort` is not a handler on anybody and does not count.
    It is what prints an unhandled WARNING to stderr, which is precisely the
    silence this question is asked to avoid.
    """

    current: logging.Logger | None = logger
    while current is not None:
        if current.handlers:
            return True
        if not current.propagate:
            return False
        current = current.parent
    return False


@contextmanager
def run_logging(outpath: str) -> Iterator[logging.Logger]:
    """Give one run its log file, and a voice if the caller left it none.

    The only place in DAWIS that installs a handler. See the module docstring
    for why it is the only one and why it is this small.

    Parameters
    ----------
    outpath : str
        The run's output stem. The log file is this plus
        :data:`LOG_EXTENSION`, which puts it beside the run's images.

    Yields
    ------
    logging.Logger
        The ``dawis`` logger, configured for the length of the block.

    Examples
    --------
    >>> with run_logging('out/ngc1275'):        # doctest: +SKIP
    ...     ...                                 # the run happens here
    """

    logger = logging.getLogger(PACKAGE_LOGGER)
    unconfigured = not _is_configured(logger)

    installed: list[logging.Handler] = [
        logging.FileHandler(filename=''.join((outpath, LOG_EXTENSION)), mode='w')
    ]
    if unconfigured:
        installed.append(logging.StreamHandler(sys.stdout))

    # Only ever raised on a logger nobody has an opinion about, and only for
    # the length of the run. `unconfigured` is what says nobody has one: a
    # caller with a handler somewhere has a level too, and theirs decides how
    # much of this run they hear.
    raise_level = unconfigured and logger.level == logging.NOTSET

    formatter = logging.Formatter(LINE_FORMAT)
    for handler in installed:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    if raise_level:
        logger.setLevel(logging.DEBUG)

    try:
        yield logger
    finally:
        if raise_level:
            logger.setLevel(logging.NOTSET)
        for handler in installed:
            logger.removeHandler(handler)
            handler.close()
