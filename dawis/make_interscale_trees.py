#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# This file is part of DAWIS.
#
# NAME : make_interscale_trees.py
# AUTHOR : Amael Ellien
# LAST MODIFICATION : 07/2021
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from datetime import datetime
from dawis.atrous import atrous
import pdb
import matplotlib.pyplot as plt
import pickle
import numpy as np
import logging
from matplotlib.colors import SymLogNorm

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class interscale_tree(object):
    """docstring for ."""

    def __init__(self, interscale_maximum, region_list, wavelet_datacube, label_datacube, clip = 30):

        self.region_list = region_list
        self.interscale_maximum = interscale_maximum
        self.n_regions = len(region_list)
        self.span_levels = np.size(np.unique([ x.level for x in region_list ]))
        self.clip = clip

        x_min = np.min([ x.bbox[0] for x in self.region_list ])
        y_min = np.min([ x.bbox[1] for x in self.region_list ])
        x_max = np.max([ x.bbox[2] for x in self.region_list ])
        y_max = np.max([ x.bbox[3] for x in self.region_list ])

        # Extent
        clip = np.copy(label_datacube.array[ x_min:x_max, \
                                             y_min:y_max, \
                                             : ])
        self.bbox = (x_min, y_min, x_max, y_max)
        extent_im = np.sum( self.tubes( wavelet_datacube, label_datacube )[1], axis = 2 )
        n_pix = np.size( np.where( extent_im != 0. )[0] )
        self.extent = n_pix / ( x_max - x_min * y_max - y_min )

        # Make vignet larger
        x_min -= np.int( ( x_max - x_min ) / 2. )
        x_max += np.int( ( x_max - x_min ) / 2. )
        if x_min < 0: x_min = 0
        if x_max > label_datacube.x_size: x_max = label_datacube.x_size

        y_min -= np.int( ( y_max - y_min ) / 2. )
        y_max += np.int( ( y_max - y_min ) / 2. )
        if y_min < 0: y_min = 0
        if y_max > label_datacube.y_size: y_max = label_datacube.y_size

        # Clip
        if x_max - x_min < self.clip:
            x_min = int( (x_max + x_min) / 2. - self.clip / 2. + 1.)   # we resize it to the minclip
            x_max = int( (x_max + x_min) / 2. + self.clip / 2. + 1.)   # value.
            if x_min < 0: x_min = 0
            if x_max > label_datacube.x_size: x_max = label_datacube.x_size

        if y_max - y_min < self.clip:
            y_min = int( (y_max + y_min) / 2. - self.clip / 2. + 1.)   # we resize it to the minclip
            y_max = int( (y_max + y_min) / 2. + self.clip / 2. + 1.)   # value.
            if y_min < 0: y_min = 0
            if y_max > label_datacube.y_size: y_max = label_datacube.y_size

        self.bbox = (x_min, y_min, x_max, y_max)
        self.x_size = x_max - x_min
        self.y_size = y_max - y_min

    def plot(self, wavelet_datacube, label_datacube, name = None, show = True, save_path = None, **kwargs):

        fig, ax = plt.subplots(1, 2, sharex = True, sharey = True, \
                                  gridspec_kw = {'hspace': 0.5, 'wspace': 0.5})

        if name == None:
            fig.suptitle('Interscale tree of region n°%d' %(self.interscale_maximum.label) )
        else:
            fig.suptitle(name)

        wavelet_image = np.sum(wavelet_datacube.array[ self.bbox[0]:self.bbox[2], \
                                                       self.bbox[1]:self.bbox[3], \
                                                       0:self.interscale_maximum.level + 1 ], axis = 2)

        try:
            norm = kwargs.pop('norm')
            im = ax[0].imshow( wavelet_image, \
                    extent = ( self.bbox[1], self.bbox[3], \
                               self.bbox[0], self.bbox[2] ), \
                    origin = 'lower', \
                    **kwargs )
            plt.colorbar(im, ax = ax[0], orientation = 'vertical')
        except:
            logthresh = 1E-3
            norm = SymLogNorm(logthresh)
            im = ax[0].imshow( wavelet_image, extent = ( self.bbox[1], self.bbox[3], \
                                                         self.bbox[0], self.bbox[2] ), \
                                              origin = 'lower', \
                                              norm = norm, \
                                              **kwargs )

            maxlog = np.log10( np.ceil( np.max( wavelet_image )))
            minlog = np.log10( np.ceil( - np.min( wavelet_image )))
            tick_locations = (list(- np.logspace(1, minlog, num = 2, base = 10)) \
                                + [0] \
                                + list(np.logspace(1, maxlog, num = 2, base = 10)))
            plt.colorbar(im, ax = ax[0], orientation = 'vertical', \
                                         ticks = tick_locations,
                                         aspect = 10,
                                         shrink = 0.5)


        label_image = np.zeros((self.bbox[2] - self.bbox[0], \
                                self.bbox[3] - self.bbox[1]))

        for i, region in enumerate(self.region_list):

            ax[0].plot( region.y_max, region.x_max, '+' )

            clip = np.copy(label_datacube.array[ self.bbox[0]:self.bbox[2], \
                                                 self.bbox[1]:self.bbox[3], \
                                                 np.int(region.level) ])

            clip[np.where(clip != region.label)] = 0.

            try:
                ax[0].contour( clip, region.label,
                                     extent = ( self.bbox[1], self.bbox[3], \
                                                self.bbox[0], self.bbox[2] ), \
                                     levels = 0, \
                                     color = 'cyan')
            except:
                print('Region too small for contours.')

            clip[np.where(clip == region.label)] = 1
            label_image += clip
            ax[1].plot( region.y_max, region.x_max, '+' )

        ax[1].imshow(label_image, extent = ( self.bbox[1], self.bbox[3], \
                                             self.bbox[0], self.bbox[2] ), \
                                  origin = 'lower',
                                  cmap = 'binary'  )


        if save_path != None:
            plt.savefig(save_path, format = 'pdf')

        if show == True:
            plt.show()
        else:
            plt.close()

    def tubes( self, wavelet_datacube, label_datacube ):

        label_tube = np.zeros((self.bbox[2] - self.bbox[0], \
                               self.bbox[3] - self.bbox[1], \
                               self.interscale_maximum.level + 1))

        wavelet_tube = np.copy(label_tube)

        for region in self.region_list:

            wavelet_tube[ region.coords[:,0] - self.bbox[0], \
                          region.coords[:,1] - self.bbox[1], \
                          region.level] = wavelet_datacube.array[ region.coords[:,0], \
                                                          region.coords[:,1], \
                                                          region.level ]

            label_tube[ region.coords[:,0] - self.bbox[0], \
                        region.coords[:,1] - self.bbox[1], \
                        region.level] = label_datacube.array[ region.coords[:,0], \
                                                              region.coords[:,1], \
                                                              region.level ]

        support_tube = np.copy(label_tube)
        support_tube[support_tube != 0] = 1.

        return wavelet_tube, support_tube, label_tube

    def synthesis_adjoint( self, wavelet_tube, filter ):

        '''
        Tild operator Â associated to the wavelet transform WT. It is given by :

                        Â(W)=SUM_i(H(i)*...*H(N-1)*W(i)) with W a wavelet object
                                                       and H a low-band filter.

        The result is an approximated first reconstruction of the image F with W=WT(F). Here H is given by the 'a trou' algorithm in its smooth only version.
        doc to modif
        '''

        smooth_coeff = np.zeros(wavelet_tube.shape)

        for level in range(wavelet_tube.shape[2]):

            smooth_coeff[:, :, level] = atrous( wavelet_tube[:, :, level], \
                                               n_levels = level + 1, \
                                               filter = filter, \
                                               conditions = 'prolongation' )[0][:, :, level]
        return np.sum( smooth_coeff, axis = 2 )

    def synthesis_sum( self, wavelet_tube ):
        return np.sum( wavelet_tube, axis = 2 )

    def SD_minimization( self, wavelet_datacube, label_datacube, filter, \
                            synthesis_operator = 'ADJOINT', sigma_flux = 1E-3, \
                            max_iter = 200, verbose = False):
        '''
        simple gradient or steepest descent (bijaoui & rué, 1995).
        '''

        wavelet_tube, support_tube, label_tube = self.tubes(wavelet_datacube, label_datacube)

        if synthesis_operator == 'ADJOINT':
            reconstructed = self.synthesis_adjoint(wavelet_tube, filter = filter)
        elif synthesis_operator == 'SUM':
            reconstructed = self.synthesis_sum(wavelet_tube)

        reconstructed[ reconstructed < 0.0 ] = 0.0

        iter = 0
        total_flux = 1.0
        while iter <= max_iter :

            iter += 1
            wr = wavelet_tube - atrous(reconstructed, n_levels = wavelet_tube.shape[2], \
                                                      filter = filter, \
                                                      conditions = 'prolongation' )[1] * support_tube

            if synthesis_operator == 'ADJOINT':
                fr = self.synthesis_adjoint(wr, filter = filter )
            elif synthesis_operator == 'SUM':
                fr = self.synthesis_sum(wr)

            norm_wr = np.sqrt( np.sum( wr**2 ) )
            if norm_wr == 0:
                norm_wr = 1.

            alpha = np.sqrt( np.sum( fr**2 ) ) / norm_wr
            reconstructed = reconstructed + alpha * fr
            reconstructed[ reconstructed < 0.0 ] = 0.0

            old_total_flux = total_flux
            total_flux = np.sum(reconstructed)
            if old_total_flux == 0.:
                old_total_flux = 1.

            diff_flux = abs(total_flux - old_total_flux) / old_total_flux

            if ( diff_flux < sigma_flux ):
                if verbose == True:
                    print( "flux convergence", np.max(reconstructed), np.shape(reconstructed) )
                break

        return reconstructed

    def CG_minimization(self, wavelet_datacube, label_datacube, filter, \
                            synthesis_operator = 'ADJOINT', step_size = 'FR', \
                            sigma_flux = 1E-3, max_iter = 200, verbose = False):
        '''
        conjugate gradient (modified from starck, murtagh & bijaoui, 1998).
        '''

        wavelet_tube, support_tube, label_tube = self.tubes(wavelet_datacube, label_datacube)
        n_levels = wavelet_tube.shape[2]

        if synthesis_operator == 'ADJOINT':
            reconstructed = self.synthesis_adjoint(wavelet_tube, filter = filter)
        elif synthesis_operator == 'SUM':
            reconstructed = self.synthesis_sum(wavelet_tube)

        wr = wavelet_tube - atrous(reconstructed, n_levels = n_levels, \
                                                  filter = filter, \
                                                  conditions = 'prolongation' )[1] * support_tube

        if synthesis_operator == 'ADJOINT':
            fr = self.synthesis_adjoint(wr, filter = filter )
        elif synthesis_operator == 'SUM':
            fr = self.synthesis_sum(wr)

        v = np.copy(fr)

        iter = 0
        total_flux = 1.0
        while iter <= max_iter:

            iter += 1

            norm_wr = np.linalg.norm(wr)
            if norm_wr == 0:
                norm_wr = 1
            delta = np.linalg.norm(fr) / norm_wr
            reconstructed = reconstructed + delta * v
            reconstructed[ reconstructed < 0.0 ] = 0.0

            old_total_flux = total_flux
            total_flux = np.sum(reconstructed)

            if old_total_flux == 0.0:
                old_total_flux = 1.0

            diff_flux = abs( total_flux - old_total_flux ) / old_total_flux
            if diff_flux < sigma_flux:
                if verbose == True:
                    print( "flux convergence", np.max(reconstructed), np.shape(reconstructed) )
                break

            else:

                old_wr = np.copy(wr)
                old_fr = np.copy(fr)

                wr = wavelet_tube - atrous( reconstructed, n_levels = n_levels, \
                                                filter = filter, \
                                                conditions = 'prolongation' )[1] * support_tube

                if synthesis_operator == 'ADJOINT':
                    fr = self.synthesis_adjoint(wr, filter = filter )
                elif synthesis_operator == 'SUM':
                    fr = self.synthesis_sum(wr)

                squared_norm_old_fr = np.linalg.norm( old_fr )**2
                if squared_norm_old_fr == 0.:
                    squared_norm_old_fr = 1.0

                if step_size == 'FR':
                    beta = np.linalg.norm( fr )**2 / squared_norm_old_fr
                elif step_size == 'PRP':
                        beta = np.sum( (fr - old_fr) * fr ) / np.sum( fr * fr )
                elif step_size == 'HS':
                        beta = np.sum( (fr - old_fr) * fr ) / np.sum( (fr - old_fr) * v )
                elif step_size == 'CD':
                        beta = np.linalg.norm( fr )**2 / np.sum( v * old_fr )
                v = fr + beta * old_fr

        return reconstructed

def write_interscale_trees_to_pickle(interscale_tree_list, filename, overwrite = True):

    if overwrite == True:
        mode = 'wb'
    else:
        mode = 'ab'

    with open(filename, mode) as outfile:
        pickle.dump(interscale_tree_list, outfile)

def read_interscale_trees_from_pickle(filename):

    with open(filename, 'rb') as infile:
        interscale_tree_list = pickle.load(infile)

    return interscale_tree_list

def make_interscale_trees(region_list, wavelet_datacube, label_datacube, tau = 0.8, min_span = 3, max_span = 3, lvl_sep_big = 6, min_reg_size = 4, verbose = False):

    interscale_tree_list = []

    region_list.sort(key = lambda x: x.norm_max_intensity, reverse = True)

    i = 0
    levels_rejected = [ 0, label_datacube.z_size ]
    threshold_maximum = region_list[0].norm_max_intensity * tau
    level_maximum = region_list[0].level
    while region_list[i].level in levels_rejected :
        i += 1
        threshold_maximum = region_list[i].norm_max_intensity * tau
        level_maximum = region_list[i].level

    interscale_maximum_list = list(filter( lambda x: ( x.norm_max_intensity >= tau * threshold_maximum ) & \
                                                     ( x.level == level_maximum ) & \
                                                     ( x.area >= min_reg_size ), region_list))
    if verbose == True:
        log = logging.getLogger(__name__)
        log.info('Estimating global interscale maximas: %d found.' %(len(interscale_maximum_list)))

    n_rejected = 0
    for interscale_maximum in interscale_maximum_list:

        if interscale_maximum.level >= lvl_sep_big :
            pmax_span = 1
            pmin_span = 1
        elif interscale_maximum.level < lvl_sep_big :
            pmax_span = max_span
            pmin_span = min_span

        tube = label_datacube.array[interscale_maximum.coords[:,0], interscale_maximum.coords[:,1], :level_maximum + 1]
        covered_label_list = list(np.unique(tube))
        covered_region_list = list(filter( lambda x: x.label in covered_label_list, region_list ))

        connected_region_list = list(filter( lambda x: (x.label in covered_label_list) & \
                                                       ( [x.x_max, x.y_max] in interscale_maximum.coords ) & \
                                                       ( x.area >= min_reg_size) & \
                                                       ( interscale_maximum.level - x.level <= pmax_span - 1 ), region_list ))

        span_levels = np.size(np.unique([ x.level for x in connected_region_list ]))

        if span_levels < pmin_span :
            #log.info('Rejected : level = %d span = %d' %(interscale_maximum.level, span_levels))
            n_rejected += 1
            continue

        interscale_tree_list.append(interscale_tree(interscale_maximum, connected_region_list, wavelet_datacube, label_datacube))

    if verbose == True:
        log.info('%d rejected.' %(n_rejected))
    if not interscale_tree_list:
        log.info("No interscale maximum found. Please consider lowering parameters 'tau' or 'min_span' if this keeps happening at every level. ")
    return interscale_tree_list

if __name__ == '__main__':

    from make_regions import *
    from datacube import *
    from matplotlib.colors import SymLogNorm, LogNorm
    from astropy.io import fits
    from atrous import atrous
    from wavelet_transforms import bspl_atrous, wavelet_datacube
    from hard_threshold import hard_threshold
    from label_regions import label_regions
    from anscombe_transforms import *
    from skimage.metrics import structural_similarity as ssim

    hdu = fits.open('/home/ellien/devd/gallery/A1365.rebin.fits')
    im = hdu[0].data
    header = hdu[0].header
    n_levels = np.int(np.min(np.floor(np.log2(im.shape))))
    print(im.shape, n_levels, 2**n_levels)
    n_levels = 3

    sigma, mean, gain = pg_noise_bissection(im, max_err = 1E-3, n_sigmas = 3)
    aim = anscombe_transform(im, sigma, mean, gain)
    acdc, awdc = bspl_atrous(aim, n_levels, header)
    sdc = hard_threshold(awdc, n_sigmas = 5)
    ldc = label_regions(sdc)
    cdc, wdc = bspl_atrous(im, n_levels, header)
    rl = make_regions_full_props(wdc, ldc, verbose = False)
    lit = make_interscale_trees(rl, wdc, ldc, tau = 0.8, verbose = False)

    for it in lit:

        it.plot(wdc, ldc)
        print(it.extent)

        #bspl = 1 / 16. * np.array([ 1, 4, 6, 4, 1 ])
        #haar = 1 / 2. * np.array([ 1, 0, 1 ])

        #sim = im[it.bbox[0]:it.bbox[2], it.bbox[1]:it.bbox[3]]
        #scores = []
        #ssims = []
        #startTime = datetime.now()
        #rima = it.CG_minimization( wdc, ldc, filter = bspl, synthesis_operator = 'ADJOINT', step_size = 'FR' )
        #scores.append(20 * np.log10(  np.sum(np.sqrt(sim**2)) / ( np.sum(np.sqrt((rima - sim)**2)) ) ))
        #ssims.append( ssim(rima, sim) )
        #print(datetime.now() - startTime)

        #startTime = datetime.now()
        #rimb = it.CG_minimization( wdc, ldc, filter = bspl, synthesis_operator = 'SUM', step_size = 'FR' )
        #scores.append(20 * np.log10(  np.sum(np.sqrt(sim**2)) / ( np.sum(np.sqrt((rimb - sim)**2)) ) ))
        #ssims.append( ssim(rimb, sim) )

        #print(datetime.now() - startTime)
        #startTime = datetime.now()
        #rimc = it.SD_minimization( wdc, ldc, filter = bspl, synthesis_operator = 'ADJOINT' )
        #scores.append(20 * np.log10(  np.sum(np.sqrt(sim**2)) / ( np.sum(np.sqrt((rimc - sim)**2)) ) ))
        #ssims.append( ssim(rimc, sim) )
        #print(datetime.now() - startTime)

        #startTime = datetime.now()
        #rimd = it.SD_minimization( wdc, ldc, filter = bspl, synthesis_operator = 'SUM' )
        #scores.append(20 * np.log10(  np.sum(np.sqrt(sim**2)) / ( np.sum(np.sqrt((rimd - sim)**2)) ) ))
        #ssims.append( ssim(rimd, sim) )
        #print(datetime.now() - startTime)

        #fig, ax = plt.subplots(3, 4)

        #for i in range(4):
        #    ax[0][i].imshow(sim, norm = LogNorm())
        #    ax[0][i].set_title('sn = %2.2f\nssim = %1.2f'%(scores[i], ssims[i]))

        #ax[1][0].imshow(rima, norm = LogNorm())
        #ax[1][1].imshow(rimb, norm = LogNorm())
        #ax[1][2].imshow(rimc, norm = LogNorm())
        #ax[1][3].imshow(rimd, norm = LogNorm())

        #ax[2][0].imshow(sim - rima, norm = SymLogNorm(1E-3))
        #ax[2][1].imshow(sim - rimb, norm = SymLogNorm(1E-3))
        #ax[2][2].imshow(sim - rimc, norm = SymLogNorm(1E-3))
        #ax[2][3].imshow(sim - rimd, norm = SymLogNorm(1E-3))

        #plt.show()
