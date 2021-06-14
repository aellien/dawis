#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# This file is part of DAWIS (Detection Algorithm for Intracluster light Studies).
# Author: Amaël Ellien
# Last modification: 07/2021
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# MODULES

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from astropy.io import fits
from dawis.exceptions import DawisDimensionError
import pdb
import copy
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class datacube(object):
    """Base class for DAWIS datacubes. Datacubes can either be wavelet planes,
    label datacubes or multiresolution supports. Each datacube comes in with met-
    hods allowing an easy way of displaying the data and save/read it into fits
    files.

    A datacube instance can either be created by declaring a Numpy array and as-
    signing a datacube type to it (a fits header created ahead of the declararion
    can also be given as an argument), or by reading directly a fits file.

    This class is created mainly for error-control purposes.

    Arguments
    ----------------

            array       The actual data, which is a Numpy array with three dim-
                        ensions. The dtype of the array, as well as the order in
                        which the elements of the array are stored (C or Fortran)
                        is left to the choice of the user. Bad dimensions will
                        raise an error. # Numpy array

            dctype      Datacube type of the datacube. The four different datacu-
                        be types are 'NONE', 'COARSE', 'WAVELET', 'LABELS' and 'SUP-
                        PORT'. # String

            fheader     Fits header associated to the datacube. The attributes
                        of the datacube are written in this fits header. If the
                        fits header was given as an argument, the attributes are
                        simply added at the end of the header, and do not overw-
                        rite any previous information. This header is then used
                        by the method 'to_fits()' when writting the datacube to
                        a fits file. # Header Object Instance

    Examples:

        >>> dc1 = dawis.datacube( np.ones(100, 100, 10), dctype = 'NONE' )
        >>> from astropy.io import fits
        >>> h2  = fits.Header()
        >>> h2['TEST_ARG'] = 10
        >>> dc2 = dawis.datacube( np.ones(100, 100, 10), dctype = 'NONE', header = h2 )

    Attributes
    ----------------

            array       See arguments.

            dctype      See arguments.

            fheader     See arguments.

            shape       Shape of the Numpy array associated to the datacube.

            x_size      Size of the first dimension (X axis, Python order).

            y_size      Size of the second dimension (Y axis, Python order).

            z_size      Size of the third dimension (Z axis, Python order).

    Useful Methods
    ----------------

            to_fits()   Write the datacube and its header to a fits file.

            from_fits() Create a datacube instance and the associated header from
                        a fits file.

            show()      Display information about the datacube.

            help()      Print this help.

    """


    def __init__(self, array, dctype = 'NONE', fheader = None):

        # ATTRIBUTES
        self.array  = array
        ndim = np.ndim(self.array)
        if ndim != 3:
            raise DawisDimensionError('Not a datacube -->', ndim, 3)

        self.shape  = np.shape(array)
        self.dctype = dctype
        self.x_size, self.y_size, self.z_size = self.shape

        # FITS HEADER
        if fheader == None:
            self.fheader = fits.Header()
        else:
            self.fheader = copy.copy(fheader)
        self.fheader['DCTYPE'] = self.dctype
        self.fheader.comments['DCTYPE'] = 'DAWIS datacube type'
        self.fheader['X_SIZE'] = self.x_size
        self.fheader.comments['X_SIZE'] = 'DAWIS X axis size (python order)'
        self.fheader['Y_SIZE'] = self.y_size
        self.fheader.comments['Y_SIZE'] = 'DAWIS Y axis size (python order)'
        self.fheader['Z_SIZE'] = self.z_size
        self.fheader.comments['Z_SIZE'] = 'DAWIS Z axis size (python order)'


    def to_fits(self, filename, overwrite = False):
        '''Write the datacube to a fits file. The fits header associated to the
        datacube is automatically used as header for the new fits file. Note that
        this function only set a Primary Header Data Unit. The output fits file
        will therefore always feature one block only.

        Arguments
        ----------------

                filename        Path/name of the fits file in which the datacube
                                is written. # String

                overwrite       Optional argument (default is False). If value
                                set to True, DAWIS will overwrite the file if it
                                already exists. # Boolean

        Examples:

            >>> dc1 = dawis.datacube( np.ones(100, 100, 10), dctype = 'NONE' )
            >>> dc1.to_fits('dc1.fits')
        '''

        if self.fheader['DCTYPE'] == 'LABEL':
            hdu_dc = fits.PrimaryHDU(self.array, header = self.fheader)
            hdu_lab = fits.ImageHDU(self.lab_counts)
            hdu = fits.HDUList([hdu_dc, hdu_lab])
            hdu.writeto(filename, overwrite = overwrite)
        else:
            hdu = fits.PrimaryHDU(self.array, header = self.fheader)
            hdu.writeto(filename, overwrite = overwrite)


    @classmethod
    def from_fits(cls, filename):
        '''Create a datacube instance from a fits file. Note that only the Prima-
        ry Header Data Unit is read. The associated header is then used as header
        attribute for the datacube instance. If there are more blocks in the fits
        file, the other are ignored.

        Arguments
        ----------------

                filename        Path/name of the fits file which is read. # String

        Examples:

            >>> dc2.from_fits('dc1.fits') # Here 'dc1.fits' is a fits file
                                          # containing a datacube.
        '''

        hdu = fits.open(filename)
        fheader, data = hdu[0].header, hdu[0].data
        try:
            dctype = fheader['DCTYPE']
            if dctype in ['WAVELET', 'COARSE']:
                return cls(data, fheader['WAVTYPE'], fheader)
            elif dctype == 'SUPPORT':
                return cls(data, fheader['TRSHTYPE'], fheader)
            elif dctype == 'LABEL':
                lab_counts = hdu[1].data
                return cls(data, lab_counts, fheader)
        except:
            dctype = 'NONE'

        return cls(data, dctype, fheader)


    def show(self):
        '''Display useful information about the datacube. Version for base data-
        cube is fairly useless.

        Examples:

            >>> dc = dawis.datacube( np.zeros((100,100,10), dctype = 'NONE') )
            >>> dc.show()
        '''
        print('\ndctype : %s\nshape : %d x %d x %d\n' %(self.dctype, \
                                                        self.x_size, \
                                                        self.y_size, \
                                                        self.z_size) )

    def waveplot(self, ncol = 5, name = None, show = True, save_path = None, **kwargs ):
        '''Display the datacube under the form a grid of plots. This function
        uses matplotlib.pyplot.imshow() to plot each plane of which the key args
        can be passed as args here.

        Arguments
        ----------------

                ncol        Number of columns in the grid of plots. # Int

                name        Title of the matplotlib figure. # String or None

                **kwargs    The matplotlib kwargs. See matplotlib.pyplot.imshow()
                            documentation for more information.
        Examples:

            >>> from numpy.random import normal
            >>> dc3 = dawis.datacube( normal(0, 1, (100, 100, 10)), dctype = 'NONE' )
            >>> dc3.waveplot(name = 'Datacube of random values')
            >>> dc3.waveplot(ncol = 2)
            >>> dc3.waveplot(name = 'Datacube of random values with bicubic\
                        interpolation', cmap = 'hot', interpolation = 'bicubic')
        '''

        # Plot make-up.
        if self.z_size > ncol:
            nline = np.int( np.ceil(self.z_size / ncol) )
        else:
            nline = 1

        fig, grid = plt.subplots(nline, ncol, sharex = False, sharey = False, \
                                  gridspec_kw = {'hspace': 0.2, 'wspace': 0.2})

        if name == None:
            fig.suptitle('Datacube of type %s'%(self.dctype) )
        else:
            fig.suptitle(name)

        lvl = 0
        for i in range(nline):
            for j in range(ncol):
                lvl += 1
                if lvl > self.z_size:
                    grid[i][j].set_frame_on(False)
                    grid[i][j].get_xaxis().set_ticks([])
                    grid[i][j].get_yaxis().set_ticks([])
                else:
                    if nline == 1:
                        ax = grid[j]
                    else:
                        ax = grid[i][j]
                    if lvl == 1:
                        norm = ax.imshow(self.array[:, :, lvl - 1], **kwargs)
                    else:
                        im = ax.imshow(self.array[:, :, lvl - 1], **kwargs)
                    cax = plt.colorbar(norm, ax = ax, aspect = 10, pad = 0, \
                                                    orientation = 'horizontal')
                    ax.set_title('z = %d' %(lvl))
                    ax.xaxis.tick_top()
                    if j != 0:
                        ax.get_yaxis().set_ticks([])
                    if i != 0:
                        ax.get_xaxis().set_ticks([])

        if save_path != None:
            plt.savefig(save_path, format = 'pdf')

        #plt.tight_layout()
        if show == True:
            plt.show()
        else:
            plt.close()

    def help(self):
        print(help(datacube))

    def __str__(self):
        return('\nDAWIS datacube object. Use "show()" method for more details.')

class wavelet_datacube(datacube):
    """docstring for ."""

    def __init__(self, array, wavelet_type, fheader = None):

        # Inheritance from parent class datacube
        super().__init__(array, 'WAVELET', fheader)
        self.fheader['WAVTYPE'] = wavelet_type
        self.fheader.comments['WAVTYPE'] = 'DAWIS wavelet type'
        self.wavelet_type = wavelet_type

    def std_plot(self, name = None, show = True, save_path = None, **kwargs ):
        '''doc to do'''

        fig = plt.figure()

        if name == None:
            fig.suptitle('Standard deviation against scale')
        else:
            fig.suptitle(name)

        plt.plot(np.std(self.array, axis = (0,1)), **kwargs)
        plt.yscale('log', basey = 10)

        if save_path != None:
            plt.savefig(save_path, format = 'pdf')

        #plt.tight_layout()
        if show == True:
            plt.show()
        else:
            plt.close()

    def waveplot(self, ncol = 5, name = None, show = True, save_path = None, **kwargs ):
        '''Display the datacube under the form a grid of plots. This function
        uses matplotlib.pyplot.imshow() to plot each plane of which the key args
        can be passed as args here. Version specific to datacube of type 'WAVELET'.

        Arguments
        ----------------

                ncol        Number of columns in the grid of plots. # Int

                name        Title of the matplotlib figure. # String or None

                **kwargs    The matplotlib kwargs. See matplotlib.pyplot.imshow()
                            documentation for more information.
        Examples:

            >>> from numpy.random import normal
            >>> dc3 = dawis.datacube( normal(0, 1, (100, 100, 10)), dctype = 'NONE' )
            >>> dc3.waveplot(name = 'Datacube of random values')
            >>> dc3.waveplot(ncol = 2)
            >>> dc3.waveplot(name = 'Datacube of random values with bicubic\
                        interpolation', cmap = 'hot', interpolation = 'bicubic')
        '''

        # Plot make-up.
        if self.z_size > ncol:
            nline = np.int( np.ceil(self.z_size / ncol) )
        else:
            nline = 1
            ncol = self.z_size

        fig, grid = plt.subplots(nline, ncol, sharex = False, sharey = False, \
                                  gridspec_kw = {'hspace': 0.2, 'wspace': 0.2})

        if name == None:
            fig.suptitle('Datacube of type %s'%(self.dctype) )
        else:
            fig.suptitle(name)

        lvl = 0
        for i in range(nline):
            for j in range(ncol):
                lvl += 1
                if lvl > self.z_size:
                    if nline == 1:
                        grid[j].set_frame_on(False)
                        grid[j].get_xaxis().set_ticks([])
                        grid[j].get_yaxis().set_ticks([])
                    else:
                        grid[i][j].set_frame_on(False)
                        grid[i][j].get_xaxis().set_ticks([])
                        grid[i][j].get_yaxis().set_ticks([])
                else:
                    if nline == 1:
                        ax = grid[j]
                    else:
                        ax = grid[i][j]

                    try:
                        norm = kwargs.pop('norm')
                        if lvl == 1:
                            im_norm = ax.imshow(self.array[:, :, lvl - 1], **kwargs)
                        else:
                            im = ax.imshow(self.array[:, :, lvl - 1], **kwargs)
                        cax = plt.colorbar(im_norm, ax = ax, aspect = 10, pad = 0, \
                                                        orientation = 'horizontal')
                    except:
                        logthresh = 1E-3
                        norm = SymLogNorm(logthresh)
                        if lvl == 1:
                            im_norm = ax.imshow(self.array[:, :, lvl - 1], norm = norm, **kwargs)
                            maxlog = np.log10( np.ceil( np.max( self.array[:, :, lvl - 1] )))
                            minlog = np.log10( np.ceil( - np.min( self.array[:, :, lvl - 1] )))
                        else:
                            im = ax.imshow(self.array[:, :, lvl - 1], norm = norm, **kwargs)

                        tick_locations = (list(- np.logspace(1, minlog, num = 2, base = 10)) \
                                            + [0] \
                                            + list(np.logspace(1, maxlog, num = 2, base = 10)))
                        cax = plt.colorbar(im_norm, ax = ax, aspect = 10, pad = 0, \
                                                    orientation = 'horizontal', \
                                                    ticks = tick_locations)
                    ax.set_title('z = %d' %(lvl))
                    ax.xaxis.tick_top()
                    if j != 0:
                        ax.get_yaxis().set_ticks([])
                    if i != 0:
                        ax.get_xaxis().set_ticks([])

        if save_path != None:
            plt.savefig(save_path, format = 'pdf')

        #plt.tight_layout()
        if show == True:
            plt.show()
        else:
            plt.close()

class coarse_datacube(datacube):
    """docstring for ."""

    def __init__(self, array, wavelet_type, fheader = None):

        # Inheritance from parent class datacube
        super().__init__(array, 'COARSE', fheader)
        # Attributes
        self.fheader['WAVTYPE'] = wavelet_type
        self.fheader.comments['WAVTYPE'] = 'DAWIS wavelet type'
        self.wavelet_type = wavelet_type

class support_datacube(datacube):
    """docstring for ."""

    def __init__(self, array, threshold_type, noise_pixels, fheader = None):

        # Inheritance from parent class datacube
        super().__init__(array, 'SUPPORT', fheader)
        # Attributes
        self.threshold_type = threshold_type
        self.noise_pixels = noise_pixels
        self.fheader['TRSHTYPE'] = threshold_type
        self.fheader.comments['TRSHTYPE'] = 'DAWIS threshold type'

    def histogram_noise(self, name = None, show = True, save_path = None, **kwargs):
        '''doc to do'''

        fig = plt.figure()

        if name == None:
            fig.suptitle('Datacube of type %s'%(self.dctype) )
        else:
            fig.suptitle(name)

        plt.hist(self.noise_pixels.flatten(),  bins = 'auto',  \
                                               histtype = 'bar', \
                                               rwidth = 10. , \
                                               align='left', \
                                               **kwargs)

        if save_path != None:
            plt.savefig(save_path, format = 'png')

        #plt.tight_layout()
        if show == True:
            plt.show()
        else:
            plt.close()

    def test_gaussianity(self, alpha = 1E-3):
        k, pval = normaltest( self.noise_pixels )
        if pval > alpha:
            print('/!\ Warning /!\ noise does not look Gaussian : p = %1.5f alpha = %1.5f'%(pval, alpha))
        else:
            print('Gaussianity test passed : p = %1.5f alpha = %1.5f'%(pval, alpha))
        return k, pval

class label_datacube(datacube):
    """docstring for ."""

    def __init__(self, array, lab_counts, fheader = None):

        # Inheritance from parent class datacube
        super().__init__(array, 'LABEL', fheader)
        # Attributes
        self.lab_counts = lab_counts
        self.fheader['LABC'] = np.sum(self.lab_counts)
        self.fheader.comments['LABC'] = 'DAWIS total label counts'

    def label_counts_plot(self, name = None, show = True, save_path = None, **kwargs ):
        '''doc to do'''

        fig = plt.figure()

        if name == None:
            fig.suptitle('Label counts against scale')
        else:
            fig.suptitle(name)

        plt.plot(self.lab_counts, **kwargs)

        if save_path != None:
            plt.savefig(save_path, format = 'pdf')

        #plt.tight_layout()
        if show == True:
            plt.show()
        else:
            plt.close()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if __name__ == '__main__':

    from numpy.random import normal
    dc3 = datacube( normal(0, 1,(100, 100, 9)), dctype = 'NONE' )
    dc3.waveplot(name = 'Datacube of random values')
    #dc3.waveplot(ncol = 2)
    #dc3.waveplot(cmap = 'hot', interpolation = 'bicubic')
    #FILE = '/home/ellien/devd/tests/datacube_test.fits'

    #dc_test.to_fits(FILE, overwrite = True )

    #dc_test_3 = datacube( np.ones( (10, 10, 3) ), dctype = 'WAVELET DATACUBE' )
    #print(dc_test_3.fheader['DCTYPE'])
    #print(dc_test_3.fheader)

    #dc_test_4 = datacube( np.ones( (10, 10, 3) ), dctype = 'BULLSHIT' )

    #dc_test_2 = datacube.from_fits(FILE)
    #print(dc_test_2.fheader)
