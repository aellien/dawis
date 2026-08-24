"""Regions: what a detection becomes once it is measured.

A label cube says which pixels belong to which detection. A
:class:`region` is that detection turned into numbers — where its peak is,
how bright it is, how elongated, and at which scale it was found. Regions are
the last product of the detection block and the input to the tree builder,
which groups them across scales.

The intensity of a region is recorded twice: raw, and normalised by the noise
conversion coefficient of its scale. The normalised value is the comparable
one, because a coefficient of a given size means something different at every
scale.
"""

from __future__ import annotations

import logging
import pickle

import numpy as np
from skimage.measure import regionprops

from dawis.detection.noise import table_noise_stds

try:
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnchoredText
except ImportError:  # matplotlib is the optional [viz] extra
    from dawis._optional import MissingDependency
    plt = MissingDependency('matplotlib', 'viz')
    AnchoredText = MissingDependency('matplotlib.offsetbox.AnchoredText', 'viz')


class region(object):
    """One connected detection, measured.

    An open bag of properties rather than a fixed record: every name in the
    allowed set is set to ``False`` and then overwritten by whatever the
    caller passes. A property left ``False`` was not measured — the two
    builders below fill different subsets — so code reading a region must
    test for that rather than assume a number.

    Parameters
    ----------
    **kwargs
        Region properties. Names outside the allowed set are dropped
        silently. Values are converted with :func:`numpy.array`.

    Attributes
    ----------
    label : int
        The region's label, unique across the whole wavelet cube.
    level : int
        Index of the wavelet scale the region was detected on.
    area : int
        Number of pixels in the region.
    coords : numpy.ndarray
        Pixel coordinates of the region, of shape ``(area, 2)``.
    bbox : tuple
        Bounding box, as ``(min_row, min_col, max_row, max_col)``.
    max_intensity, min_intensity : float
        Extreme wavelet coefficients inside the region.
    norm_max_intensity, norm_min_intensity : float
        The same, divided by the noise conversion coefficient of the scale,
        so values from different scales are comparable.
    x_max, y_max : int
        Pixel coordinates of the brightest pixel — the region's peak.
    x_min, y_min : int
        Pixel coordinates of the faintest pixel.
    """

    def __init__(self, **kwargs):

        allowed_keys = set(['segmented','label', 'area', 'axis_major_length', 'axis_minor_length', \
                        'bbox', 'bbox_area', 'centroid', 'local_centroid', 'coords', 'extent', \
                        'eccentricity', 'max_intensity', 'min_intensity', \
                        'norm_max_intensity', 'norm_min_intensity', 'x_max', 'y_max', \
                        'level', 'x_min', 'y_min', 'moments', 'moments_central', \
                        'moments_hu', 'moments_normalized', 'moments_weighted', 'moments_weighted_central', \
                        'moments_weighted_hu', 'moments_weighted_normalized', 'orientation', 'perimeter' ])
        # initialize all allowed keys to false
        self.__dict__.update((key, False) for key in allowed_keys)
        # and update the given keys by their given values
        self.__dict__.update((key, np.array(value)) for key, value in kwargs.items() if key in allowed_keys)

    

    def plot(self, wavelet_datacube, label_datacube, name = None, show = True, save_path = None, **kwargs):
        """Show the region's cutout of its wavelet plane, with its outline.

        Parameters
        ----------
        wavelet_datacube : dawis.cubes.wavelet_datacube
            The cube the region was detected on, supplying the pixels.
        label_datacube : dawis.cubes.label_datacube
            The label cube, supplying the outline.
        name : str, optional
            Figure title. Default is None, which titles it by label.
        show : bool, optional
            Display the figure. Default is True; False closes it instead.
        save_path : str, optional
            Write the figure as PDF. Default is None, which writes nothing.
        **kwargs
            Passed through to :func:`matplotlib.pyplot.imshow`.

        Notes
        -----
        A region with no ``bbox`` gets one computed here, by scanning the
        whole label cube for its label, and the result is cached on the
        instance.

        Moves to the ``viz`` subpackage when that is carved.
        """

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
                                            int(self.level) ], \
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
                                           int(self.level) ]
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

def make_regions_full_props(
    wavelet_datacube,
    label_datacube,
    verbose: bool = False,
) -> list[region]:
    """Measure every labelled region of a cube.

    The builder the pipeline runs. Each plane is handed to
    :func:`skimage.measure.regionprops` with the wavelet plane as intensity
    image, and each result becomes a :class:`region`.

    Parameters
    ----------
    wavelet_datacube : dawis.cubes.wavelet_datacube
        The wavelet planes, supplying pixel intensities.
    label_datacube : dawis.cubes.label_datacube
        The label cube naming the regions.
    verbose : bool, optional
        Log the total number of regions found. Default is False.

    Returns
    -------
    list of region
        Every region of every scale, in scale order then label order.

    Notes
    -----
    The peak position is taken as the *first* pixel whose value equals the
    region's maximum. A region with a flat top therefore gets a peak that
    depends on pixel ordering rather than on the data.

    The conversion table is read for ``wavelet_datacube.z_size`` scales, so
    the wavelet cube and the label cube must have the same depth.
    """

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
    """Measure only peak and area, in parallel. Deprecated and broken.

    .. deprecated:: 0.0.7

       Raises ``NameError`` on every call: the Fortran-backed worker
       ``local_maxima_regions_para`` it drives was dropped from the package
       and no replacement exists. Nothing calls it. It is carried through the
       carve unchanged rather than repaired, so that the decision to delete
       or rebuild it is taken on its own ticket and not inside a move.
       Use :func:`make_regions_full_props`.
    """

    conv_table = table_noise_stds(wavelet_datacube.z_size, wavelet_type = wavelet_datacube.wavelet_type)

    region_list = []
    region_tab = np.zeros((int(np.sum(label_datacube.lab_counts)), 12), order = 'F')
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

        norm_max_intensity = region_tab[i, 6] / conv_table[int(region_tab[i, 2])]
        label = i + 1
        region_list.append( region( label = label, \
                                    area = region_tab[i, 11], \
                                    max_intensity = region_tab[i, 6], \
                                    norm_max_intensity = norm_max_intensity, \
                                    x_max = region_tab[i, 0], \
                                    y_max = region_tab[i, 1], \
                                    level = region_tab[i, 2] ) )

    return region_list

def write_regions_to_pickle(
    region_list: list[region],
    filename: str,
    overwrite: bool = True,
) -> None:
    """Write a region list to a pickle file.

    Parameters
    ----------
    region_list : list of region
        The regions to write.
    filename : str
        Path to write to.
    overwrite : bool, optional
        Replace the file. Default is True; False appends a second pickle
        stream to the existing file, which
        :func:`read_regions_from_pickle` will not read back.

    Notes
    -----
    Moves to the ``io`` subpackage when that is carved.
    """

    if overwrite == True:
        mode = 'wb'
    else:
        mode = 'ab'

    with open(filename, mode) as outfile:
        pickle.dump(region_list, outfile)

def read_regions_from_pickle(filename: str) -> list[region]:
    """Read a region list back from a pickle file.

    Parameters
    ----------
    filename : str
        Path written by :func:`write_regions_to_pickle`.

    Returns
    -------
    list of region
        The regions, in the order they were written.

    Notes
    -----
    Reads one pickle stream, so a file appended to holds regions this does
    not return.

    Unpickling executes whatever the file says. Read only files you produced.

    Moves to the ``io`` subpackage when that is carved.
    """

    with open(filename, 'rb') as infile:
        region_list = pickle.load(infile)

    return region_list
