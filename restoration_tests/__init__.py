"""Restoration test battery — the fixture layer and the arms measured on it.

Not part of the DAWIS package. This directory holds the research code of the
reconstruction-algorithm map (#74): the fixtures every arm is measured on
(#75), the metrics and the arm runner (#77), the arms themselves (#78) and
the reporting (#79). It is excluded from the wheel, the same way
``simulations/`` is, and it imports DAWIS rather than being imported by it.

The directory is spelled with an underscore rather than the hyphen #74 and
#75 write, because the tests in ``tests/unit/`` import it and a hyphen is not
a Python identifier. That is the only deviation.
"""
