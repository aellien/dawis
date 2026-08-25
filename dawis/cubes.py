#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# This file is part of DAWIS (Detection Algorithm for Intracluster light Studies).
# Author: Amaël Ellien
# Last modification: 07/2021
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# MODULES

import numpy as np
from astropy.io import fits
from dawis.exceptions import DawisDimensionError
import pdb
import copy
# The four display methods that hung off these classes moved to
# :mod:`dawis.viz.cubes` with #46, where the two versions of the plane grid
# became one `plot_cube` dispatching on cube type. Nothing outside
# :mod:`dawis.viz` imports matplotlib.
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class Cube(object):
    """Base class for DAWIS cubes. Cubes can either be wavelet planes,
    label cubes or multiresolution supports. Each cube comes in with met-
    hods allowing an easy way of displaying the data and save/read it into fits
    files.

    A cube instance can either be created by declaring a Numpy array and as-
    signing a cube type to it (a fits header created ahead of the declararion
    can also be given as an argument), or by reading directly a fits file.

    This class is created mainly for error-control purposes.

    Arguments
    ----------------

            array       The actual data, which is a Numpy array with three dim-
                        ensions. The dtype of the array, as well as the order in
                        which the elements of the array are stored (C or Fortran)
                        is left to the choice of the user. Bad dimensions will
                        raise an error. # Numpy array

            dctype      Cube type of the cube. The four different cube types
                        are 'NONE', 'COARSE', 'WAVELET', 'LABELS' and 'SUP-
                        PORT'. # String

            fheader     Fits header associated to the cube. The attributes
                        of the cube are written in this fits header. If the
                        fits header was given as an argument, the attributes are
                        simply added at the end of the header, and do not overw-
                        rite any previous information. This header is then used
                        by the method 'to_fits()' when writting the cube to
                        a fits file. # Header Object Instance

    Examples:

        >>> dc1 = dawis.Cube( np.ones(100, 100, 10), dctype = 'NONE' )
        >>> from astropy.io import fits
        >>> h2  = fits.Header()
        >>> h2['TEST_ARG'] = 10
        >>> dc2 = dawis.Cube( np.ones(100, 100, 10), dctype = 'NONE', header = h2 )

    Attributes
    ----------------

            array       See arguments.

            dctype      See arguments.

            fheader     See arguments.

            shape       Shape of the Numpy array associated to the cube.

            x_size      Size of the first dimension (X axis, Python order).

            y_size      Size of the second dimension (Y axis, Python order).

            z_size      Size of the third dimension (Z axis, Python order).

    Useful Methods
    ----------------

            to_fits()   Write the cube and its header to a fits file.

            from_fits() Create a cube instance and the associated header from
                        a fits file.

            show()      Display information about the cube.

            help()      Print this help.

    """


    def __init__(self, array, dctype = 'NONE', fheader = None):

        # ATTRIBUTES
        self.array  = array
        ndim = np.ndim(self.array)
        if ndim != 3:
            raise DawisDimensionError('Not a cube -->', ndim, 3)

        self.shape  = np.shape(array)
        self.dctype = dctype
        self.x_size, self.y_size, self.z_size = self.shape

        # FITS HEADER
        if fheader == None:
            self.fheader = fits.Header()
        else:
            self.fheader = copy.copy(fheader)
        self.fheader['DCTYPE'] = self.dctype
        self.fheader.comments['DCTYPE'] = 'DAWIS cube type'
        self.fheader['X_SIZE'] = self.x_size
        self.fheader.comments['X_SIZE'] = 'DAWIS X axis size (python order)'
        self.fheader['Y_SIZE'] = self.y_size
        self.fheader.comments['Y_SIZE'] = 'DAWIS Y axis size (python order)'
        self.fheader['Z_SIZE'] = self.z_size
        self.fheader.comments['Z_SIZE'] = 'DAWIS Z axis size (python order)'


    def to_fits(self, filename, overwrite = False):
        '''Write the cube to a fits file. The fits header associated to the
        cube is automatically used as header for the new fits file. Note that
        this function only set a Primary Header Data Unit. The output fits file
        will therefore always feature one block only.

        Arguments
        ----------------

                filename        Path/name of the fits file in which the cube
                                is written. # String

                overwrite       Optional argument (default is False). If value
                                set to True, DAWIS will overwrite the file if it
                                already exists. # Boolean

        Examples:

            >>> dc1 = dawis.Cube( np.ones(100, 100, 10), dctype = 'NONE' )
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
        '''Create a cube instance from a fits file. Note that only the Prima-
        ry Header Data Unit is read. The associated header is then used as header
        attribute for the cube instance. If there are more blocks in the fits
        file, the other are ignored.

        Arguments
        ----------------

                filename        Path/name of the fits file which is read. # String

        Examples:

            >>> dc2.from_fits('dc1.fits') # Here 'dc1.fits' is a fits file
                                          # containing a cube.
        '''

        hdu = fits.open(filename)
        fheader, data = hdu[0].header, hdu[0].data

        # A file with no DCTYPE was not written by DAWIS, which is allowed:
        # it is read as an untyped cube. A file that *says* what it is and
        # then lacks what that kind needs is broken, and the KeyError or
        # IndexError from reading it is left to travel.
        dctype = fheader.get('DCTYPE', 'NONE')

        if dctype in ['WAVELET', 'COARSE']:
            return cls(data, fheader['WAVTYPE'], fheader)
        elif dctype == 'SUPPORT':
            return cls(data, fheader['TRSHTYPE'], fheader)
        elif dctype == 'LABEL':
            lab_counts = hdu[1].data
            return cls(data, lab_counts, fheader)

        return cls(data, dctype, fheader)


    def show(self):
        '''Display useful information about the cube. Version for the base
        cube is fairly useless.

        Examples:

            >>> dc = dawis.Cube( np.zeros((100,100,10), dctype = 'NONE') )
            >>> dc.show()
        '''
        print('\ndctype : %s\nshape : %d x %d x %d\n' %(self.dctype, \
                                                        self.x_size, \
                                                        self.y_size, \
                                                        self.z_size) )

    def help(self):
        print(help(Cube))

    def __str__(self):
        return('\nDAWIS Cube object. Use "show()" method for more details.')

class WaveletCube(Cube):
    """docstring for ."""

    def __init__(self, array, wavelet_type, fheader = None):

        # Inheritance from parent class Cube
        super().__init__(array, 'WAVELET', fheader)
        self.fheader['WAVTYPE'] = wavelet_type
        self.fheader.comments['WAVTYPE'] = 'DAWIS wavelet type'
        self.wavelet_type = wavelet_type

class CoarseCube(Cube):
    """docstring for ."""

    def __init__(self, array, wavelet_type, fheader = None):

        # Inheritance from parent class Cube
        super().__init__(array, 'COARSE', fheader)
        # Attributes
        self.fheader['WAVTYPE'] = wavelet_type
        self.fheader.comments['WAVTYPE'] = 'DAWIS wavelet type'
        self.wavelet_type = wavelet_type

class SupportCube(Cube):
    """docstring for ."""

    def __init__(self, array, threshold_type, noise_pixels, fheader = None):

        # Inheritance from parent class Cube
        super().__init__(array, 'SUPPORT', fheader)
        # Attributes
        self.threshold_type = threshold_type
        self.noise_pixels = noise_pixels
        self.fheader['TRSHTYPE'] = threshold_type
        self.fheader.comments['TRSHTYPE'] = 'DAWIS threshold type'

    def test_gaussianity(self, alpha = 1E-3):
        k, pval = normaltest( self.noise_pixels )
        if pval > alpha:
            print('/!\ Warning /!\ noise does not look Gaussian : p = %1.5f alpha = %1.5f'%(pval, alpha))
        else:
            print('Gaussianity test passed : p = %1.5f alpha = %1.5f'%(pval, alpha))
        return k, pval

class LabelCube(Cube):
    """docstring for ."""

    def __init__(self, array, lab_counts, fheader = None):

        # Inheritance from parent class Cube
        super().__init__(array, 'LABEL', fheader)
        # Attributes
        self.lab_counts = lab_counts
        self.fheader['LABC'] = np.sum(self.lab_counts)
        self.fheader.comments['LABC'] = 'DAWIS total label counts'

