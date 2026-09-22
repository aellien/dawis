"""What the one diverging cell of #84's field returns at each cap."""

from dataclasses import replace

import numpy as np

from dawis.restoration.inpainting import subtract_background
from dawis.wavelets import bspl_atrous
from restoration_tests.arms.grid import ArmSpec
from restoration_tests.arms.proximal import fista
from restoration_tests.fixtures.mock import FROZEN_SETTINGS, run_mock_iteration

N_LEVELS = 5
shape = (128, 128)
rows, columns = np.indices(shape).astype(float)
image = np.zeros(shape)
for row, column, amplitude, width in ((30.0, 90.0, 400.0, 3.0),
                                      (95.0, 40.0, 250.0, 4.0)):
    image += amplitude * np.exp(
        -((rows - row) ** 2 + (columns - column) ** 2) / (2.0 * width ** 2))
image += np.random.RandomState(7).normal(loc = 10.0, scale = 1.0, size = shape)

iteration = run_mock_iteration(image, replace(FROZEN_SETTINGS,
                                              n_levels = N_LEVELS))
subtract_background(image, n_sigmas = 3, bins = 200)
bspl_atrous(image, N_LEVELS, None, coarse = True)
tree = max(iteration.trees, key = lambda t: t.interscale_maximum.level)

wavelet_tube, support_tube, _ = tree.tubes(iteration.wavelet_cube,
                                           iteration.label_cube)
spec = ArmSpec('fista', 'HAAR', 'SUM', coarse = False)

for cap in (1, 2, 3, 5, 10, 15, 20, 22, 24, 25, 28, 29, 30, 31, 40):
    records = []
    result = fista(spec, wavelet_tube, support_tube,
                   observer = lambda record, _: records.append(record),
                   max_iter = cap, sigma_flux = 1e-3)
    start = records[0].norm_wr
    print('cap %-4d diverged %-5s  final %-12.5g  growth x%.4g'
          % (cap, result.diverged, result.norm_wr,
             result.norm_wr / start if not result.diverged else float('nan')))
