#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# FILE : DL_table.py (DL_WAV V1.0)
# AUTHOR : Amael Ellien
# DATE (LAST MODIFICATION) :

# Contains gaussian noise conversion coefficients for the 11 first levels of
# wavelet convolution. For levels higher, the value is set to 0.

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import numpy as np
from dawis.exceptions import DawisUnknownType

def table_noise_stds(level_max = 11, wavelet_type = 'BSPL'):
    '''
    Contains gaussian noise conversion coefficients for the 11 first levels of wavelet deconvolution. For levels higher, the value is set to 0.
        Old OV_WAV table
        elif WAV_TYPE=='BSPL':
            TAB_CONVERSION=np.array([0.910276420,\
                         0.0937225640,\
                         0.0462083910,\
                         0.0205422070,\
                         0.0107114720,\
                         0.0060998319,\
                         0.0041865255,\
                         0.0018956766,\
                         0.0009780460,\
                         0.0006189730])
    '''

    if wavelet_type == 'LINEAR':
        table_noise_stds = np.array([0.7984657100,\
                      0.2731163700,\
                      0.1204915100,\
                      0.0588084980,\
                      0.0300016580,\
                      0.0159785950,\
                      0.0091017098,\
                      0.0053883394,\
                      0.0031372000,\
                      0.0013005500,\
                      0.0008742930])

    elif wavelet_type=='BSPL':
        table_noise_stds = np.array([0.869051,\
                     0.19575408,\
                     0.08348428,\
                     0.040284764,\
                     0.019972015,\
                     0.009966966,\
                     0.0050156787,\
                     0.002540318,\
                     0.0013197723,\
                     0.001175349])

    elif wavelet_type == 'HAAR':
        table_noise_stds = np.array([1.118736, \
                                 0.43321106, \
                                 0.21665096, \
                                 0.10852972, \
                                 0.054709565, \
                                 0.027380498, \
                                 0.013997642, \
                                 0.007288606, \
                                 0.0038535022, \
                                 0.0026877765])

    elif wavelet_type == 'LBSPL':
        table_noise_stds = np.array([6.0119975e-01, \
                                     3.4755176e-01, \
                                     1.3366205e-01, \
                                     8.1820749e-02, \
                                     5.2327633e-02, \
                                     3.6003221e-02, \
                                     2.2381091e-02, \
                                     1.4172579e-02, \
                                     8.7727467e-03, \
                                     5.4407278e-03, \
                                     3.3427631e-03, \
                                     2.0711003e-03, \
                                     1.2263433e-03, \
                                     6.6611584e-04, \
                                     3.7057605e-04, \
                                     2.2101894e-04, \
                                     1.3023346e-04, \
                                     6.2509251e-05, \
                                     3.6206282e-05])

    else:
        raise DawisUnknownType('Unknown wavelet type -->', wavelet_type)

    if level_max < np.size(table_noise_stds):
        table_noise_stds = table_noise_stds[0:level_max]

    elif level_max > np.size(table_noise_stds):
        table_noise_stds = np.append(table_noise_stds, np.zeros(level_max - np.size(table_noise_stds)))

    return table_noise_stds
