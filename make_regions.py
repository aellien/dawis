#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# This file is part of DAWIS.
#
# NAME : make regions.py
# AUTHOR : Amael Ellien
# LAST MODIFICATION : 07/2021
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from dawis.table_noise_stds import table_noise_stds
from skimage.measure import regionprops
from matplotlib.offsetbox import AnchoredText
from datetime import datetime
import pdb
import matplotlib.pyplot as plt
import pickle
import numpy as np
import logging

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class region(object):
    """docstring for ."""

    def __init__(self, **kwargs):

        allowed_keys = set(['label', 'area', \
                        'bbox', 'bbox_area', 'centroid', 'local_centroid', 'coords', 'extent', \
                        'eccentricity', 'max_intensity', 'min_intensity', \
                        'norm_max_intensity', 'norm_min_intensity', 'x_max', 'y_max', \
                        'level', 'x_min', 'y_min'])
        # initialize all allowed keys to false
        self.__dict__.update((key, False) for key in allowed_keys)
        # and update the given keys by their given values
        self.__dict__.update((key, value) for key, value in kwargs.items() if key in allowed_keys)

    def plot(self, wavelet_datacube, label_datacube, name = None, show = True, save_path = None, **kwargs):

        fig = plt.figure()

        if name == None:
            fig.suptitle('Region n°%d' %(self.label) )
        else:
            fig.suptitle(name)

        if self.bbox == False:
            coo_region = np.where( label_datacube.array == self.label )
            min_x_axis = np.min(coo_region[0])
            max_x_axis = np.max(coo_region[0])
            min_y_axis = np.min(coo_region[1])
            max_y_axis = np.max(coo_region[1])
            self.bbox = (min_x_axis, min_y_axis, max_x_axis, max_y_axis)

        plt.imshow( wavelet_datacube.array[ self.bbox[0]:self.bbox[2], \
                                            self.bbox[1]:self.bbox[3], \
                                            np.int(self.level) ], \
                    extent = ( self.bbox[1], self.bbox[3], \
                               self.bbox[0], self.bbox[2] ), \
                    origin = 'lower', \
                    **kwargs )

        plt.plot( self.y_max, self.x_max, '+', color = 'black' )

        plt.gca().add_artist(AnchoredText( r'level = %d'%(self.level), \
                                                loc = 'upper right', \
                                                frameon=True, \
                                                pad=0.4, \
                                                borderpad=0.1, \
                                                prop=dict(size=10)))

        try:
            clip = label_datacube.array[ self.bbox[0]:self.bbox[2], \
                                           self.bbox[1]:self.bbox[3], \
                                           np.int(self.level) ]
            clip[np.where(clip != self.label)] = 0.
            plt.contour( clip, \
                    self.label, \
                    extent = ( self.bbox[1], self.bbox[3], \
                               self.bbox[0], self.bbox[2] ), \
                    levels = 0, color = 'black', linewidth = 3 )
        except:
            print('Region too small for countours.')

        if save_path != None:
            plt.savefig(save_path, format = 'pdf')

        #plt.tight_layout()
        if show == True:
            plt.show()
        else:
            plt.close()

def make_regions_full_props(wavelet_datacube, label_datacube, verbose = False):
    '''doc to do'''

    region_list = []
    conv_table = table_noise_stds(wavelet_datacube.z_size, wavelet_type = wavelet_datacube.wavelet_type)

    for level in range(0, label_datacube.z_size):

        props_skimage = regionprops(label_datacube.array[:,:,level].astype(int), wavelet_datacube.array[:,:,level])

        for region_skimage in props_skimage:

            x_max, y_max = region_skimage.coords[np.where(\
                                  wavelet_datacube.array[region_skimage.coords[:,0], \
                                                         region_skimage.coords[:,1], \
                                                         level] == region_skimage.max_intensity)][0]


            x_min, y_min = region_skimage.coords[np.where(\
                                  wavelet_datacube.array[region_skimage.coords[:,0], \
                                                         region_skimage.coords[:,1], \
                                                         level] == region_skimage.min_intensity)][0]


            norm_max_intensity = region_skimage.max_intensity / conv_table[level]
            norm_min_intensity = region_skimage.min_intensity / conv_table[level]

            region_list.append( region( label = region_skimage.label, \
                                        area = region_skimage.area, \
                                        bbox = region_skimage.bbox, \
                                        bbox_area = region_skimage.bbox_area, \
                                        centroid = region_skimage.centroid, \
                                        local_centroid = region_skimage.local_centroid, \
                                        coords = region_skimage.coords, \
                                        extent = region_skimage.extent, \
                                        eccentricity = region_skimage.eccentricity, \
                                        max_intensity = region_skimage.max_intensity, \
                                        min_intensity = region_skimage.min_intensity, \
                                        norm_max_intensity = norm_max_intensity, \
                                        norm_min_intensity = norm_min_intensity, \
                                        x_max = x_max, \
                                        y_max = y_max, \
                                        level = level, \
                                        x_min = x_min, \
                                        y_min = y_min ) )

    if verbose == True:
        log = logging.getLogger(__name__)
        log.info('Found %d regions of significance.' %(len(region_list)))

    return region_list

def make_regions_crucial_props(wavelet_datacube, label_datacube, n_cpus = 1, verbose = False):
    '''/!\ deprecated'''

    conv_table = table_noise_stds(wavelet_datacube.z_size, wavelet_type = wavelet_datacube.wavelet_type)

    region_list = []
    region_tab = np.zeros((np.int(np.sum(label_datacube.lab_counts)), 12), order = 'F')
    wavelet_array = np.copy(wavelet_datacube.array, order='F')
    label_array = np.copy(label_datacube.array, order='F')

    region_tab[:,3] = 1.
    region_tab[:,4] =- 1.
    region_tab[:,10] =- 1.

    local_maxima_regions_para(region_tab, \
                              wavelet_array, \
                              label_array, \
                              label_datacube.lab_counts, \
                              n_cpus, \
                              verbose )

    for i in range(region_tab.shape[0]):

        norm_max_intensity = region_tab[i, 6] / conv_table[np.int(region_tab[i, 2])]
        label = i + 1
        region_list.append( region( label = label, \
                                    area = region_tab[i, 11], \
                                    max_intensity = region_tab[i, 6], \
                                    norm_max_intensity = norm_max_intensity, \
                                    x_max = region_tab[i, 0], \
                                    y_max = region_tab[i, 1], \
                                    level = region_tab[i, 2] ) )

    return region_list

def write_regions_to_pickle(region_list, filename, overwrite = True):

    if overwrite == True:
        mode = 'wb'
    else:
        mode = 'ab'

    with open(filename, mode) as outfile:
        pickle.dump(region_list, outfile)

def read_regions_from_pickle(filename):

    with open(filename, 'rb') as infile:
        region_list = pickle.load(infile)

    return region_list

if __name__ == '__main__':

    from astropy.io import fits
    from datacube import *

    acdc = coarse_datacube.from_fits('/home/ellien/devd/tests/A1365.rebin.acdc.fits')
    cdc = coarse_datacube.from_fits('/home/ellien/devd/tests/A1365.rebin.cdc.fits')
    awdc = wavelet_datacube.from_fits('/home/ellien/devd/tests/A1365.rebin.awdc.fits')
    wdc = wavelet_datacube.from_fits('/home/ellien/devd/tests/A1365.rebin.wdc.fits')
    sdc = support_datacube.from_fits('/home/ellien/devd/tests/A1365.rebin.sdc.fits')
    ldc = label_datacube.from_fits('/home/ellien/devd/tests/A1365.rebin.ldc.fits')

    rl = read_regions_from_pickle('/home/ellien/devd/tests/A1365.rebin.reglist.pkl')
    rl.sort(key = lambda x: x.norm_max_intensity, reverse = True)
    for r in rl[:10]:r.plot(wdc, ldc, cmap = 'hot')
    #region_list_f = make_regions_crucial_props(wdc, ldc, n_cpus = 4, verbose = True)
    #region_list[10001].plot(wdc, ldc, cmap = 'hot')
    #region_list_f[10001].plot(wdc, ldc, cmap = 'hot')
    #write_regions_to_pickle(region_list, '/home/ellien/devd/tests/A1365.rebin.reglist.pkl', overwrite = True)
    #region_list_test = read_regions_from_pickle('/home/ellien/devd/tests/A1365.rebin.reglist.pkl')
