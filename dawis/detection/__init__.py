"""Detection: finding significant structure in wavelet space.

The second block carved out of the flat package. It depends only on
:mod:`dawis.cubes` and the exception family, so nothing here points
backwards. Its input is what :mod:`dawis.wavelets` produces, but it never
imports it.

Detection answers one question — which coefficients of a wavelet cube are
real structure rather than noise — and then measures what it found. Three
things live here.

The **threshold** is what makes the question answerable: noise shrinks with
scale by a known factor, tabulated in :func:`table_noise_stds`, so one
threshold quoted in sigmas of the image noise becomes a different, correct
threshold at every scale.

The **detectors** apply it. :func:`ms_detect_and_deblend` is the one the
pipeline runs: it segments every plane and splits blended sources in a single
pass, returning labelled regions and the per-scale uncertainty. The older
pair :func:`hard_threshold` and :func:`label_regions` does the same job in two
steps and without deblending, and is kept because the published
Ellien+2021 configuration selects it.

The **measurement** turns labels into numbers: :func:`make_regions_full_props`
builds one :class:`region` per detection, carrying its peak, extent and
intensity — normalised by the noise table so scales can be compared. Regions
are what the tree builder groups.

The old flat modules ``dawis.hard_threshold``, ``dawis.label_regions``,
``dawis.make_regions``, ``dawis.detect_and_deblend`` and
``dawis.table_noise_stds`` held these names until #50 deleted them. This
package is the only address they have.
"""

from dawis.detection.deblending import ms_detect_and_deblend
from dawis.detection.labelling import label_regions
from dawis.detection.noise import table_noise_stds
from dawis.detection.regions import (
    make_regions_crucial_props,
    make_regions_full_props,
    region,
)
from dawis.detection.thresholding import hard_threshold

__all__ = [
    'hard_threshold',
    'label_regions',
    'make_regions_crucial_props',
    'make_regions_full_props',
    'ms_detect_and_deblend',
    'region',
    'table_noise_stds',
]
