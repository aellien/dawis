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
from dawis.wavelet_transforms import bspl_atrous
from dawis.deconvolution import pad, unpad, richardson_lucy_np, B2spline, twod_B2spline
from dawis.table_noise_stds import table_noise_stds
import pdb
try:
    import matplotlib.pyplot as plt
except ImportError:  # matplotlib is the optional [viz] extra
    from dawis._optional import MissingDependency
    plt = MissingDependency('matplotlib', 'viz')
import gc
import _pickle as pickle
import numpy as np
import logging
try:
    import ray
except ImportError:  # ray is the optional [ray] extra
    from dawis._optional import MissingRay
    ray = MissingRay()
try:
    from matplotlib.colors import SymLogNorm
except ImportError:  # matplotlib is the optional [viz] extra
    from dawis._optional import MissingDependency
    SymLogNorm = MissingDependency('matplotlib.colors.SymLogNorm', 'viz')
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from scipy.optimize import line_search
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from astropy.stats import sigma_clip

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class interscale_tree(object):
    """Base class for interscale trees. Interscale trees are composed of wavelet 
    coefficient regions that are connected throughout the wavelet planes. Usually
    they are build from a master region called "interscale_maximum".
    """

    def __init__(self, interscale_maximum = None, region_list = None, wavelet_datacube = None, label_datacube = None, det_err_array = None, clip = 30, blank = False):

        if blank == False:
             
            self.region_list = region_list
            self.interscale_maximum = interscale_maximum
            self.n_regions = len(region_list)
            self.span_levels = np.size(np.unique([ x.level for x in region_list ]))
            self.clip = clip
            self.eccentricity = self.interscale_maximum.eccentricity
            self.det_err_array = det_err_array
            
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
            self.extent = n_pix / ( ( x_max - x_min ) * ( y_max - y_min ) )

            # Make vignet larger
            x_min -= int( ( x_max - x_min ) / 2. )
            x_max += int( ( x_max - x_min ) / 2. )
            if x_min < 0: x_min = 0
            if x_max > label_datacube.x_size: x_max = label_datacube.x_size

            y_min -= int( ( y_max - y_min ) / 2. )
            y_max += int( ( y_max - y_min ) / 2. )
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

            # Bayesian parameter bounds
            self.lowb = 0.
            self.highb = 2.
        
    def det_err_tube(self, wavelet_datacube, label_datacube):
        
        wavelet_tube, support_tube, label_tube = self.tubes(wavelet_datacube, label_datacube)
        det_err_tube = np.copy(support_tube)
        
        for level in range(0, det_err_tube.shape[2]):
                det_err_tube[:,:,level] *= self.det_err_array[level]
        
        return det_err_tube
            
    
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
                                                 int(region.level) ])

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
            plt.savefig(save_path, format = 'png')

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

        wr = wavelet_tube - atrous(reconstructed, n_levels = wavelet_tube.shape[2], \
                                                      filter = filter, \
                                                      conditions = 'prolongation' )[1] * support_tube

        norm_wr = np.sqrt( np.sum( wr**2 ) )
        sum_wr = np.sum( wr )

        return reconstructed, sum_wr, norm_wr

    def CG_minimization(self, wavelet_datacube, label_datacube, filter, deconv = False, \
                            synthesis_operator = 'ADJOINT', step_size = 'FR', \
                            sigma_flux = 1E-3, max_iter = 3, verbose = False):
        '''
        conjugate gradient (modified from starck, murtagh & bijaoui, 1998).
        '''

        wavelet_tube, support_tube, label_tube = self.tubes(wavelet_datacube, label_datacube)
        n_levels = wavelet_tube.shape[2]

        if deconv == True:
            if self.interscale_maximum.level <= 2:
                im_psf = twod_B2spline(xsize = self.x_size, ysize = self.y_size, order = 0)

            else:
                im_psf = twod_B2spline(xsize = self.x_size, ysize = self.y_size, order = 2)
    
        if synthesis_operator == 'ADJOINT':
            reconstructed = self.synthesis_adjoint(wavelet_tube, filter = filter)
            if deconv == True:
                reconstructed = richardson_lucy_np(reconstructed, im_psf, 10)
        elif synthesis_operator == 'SUM':
            reconstructed = self.synthesis_sum(wavelet_tube)

        wr = wavelet_tube - atrous(reconstructed, n_levels = n_levels, \
                                                  filter = filter, \
                                                  conditions = 'prolongation' )[1] * support_tube

        if synthesis_operator == 'ADJOINT':
            fr = self.synthesis_adjoint(wr, filter = filter )
            if deconv == True:
                fr = richardson_lucy_np(fr, im_psf, 10)
                
        elif synthesis_operator == 'SUM':
            fr = self.synthesis_sum(wr)

        v = np.copy(fr)

        iter = 0
        total_flux = 1.0
        while iter <= max_iter:

            iter = iter + 1

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
                    print( "flux convergence", diff_flux, np.max(reconstructed), np.shape(reconstructed) )
                break

            else:

                old_wr = np.copy(wr)
                old_fr = np.copy(fr)

                wr = wavelet_tube - atrous( reconstructed, n_levels = n_levels, \
                                                filter = filter, \
                                                conditions = 'prolongation' )[1] * support_tube

                if synthesis_operator == 'ADJOINT':
                    fr = self.synthesis_adjoint(wr, filter = filter )
                    if deconv == True:
                        fr = richardson_lucy_np(fr, im_psf, 10)
                        
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

        wr = wavelet_tube - atrous(reconstructed, n_levels = wavelet_tube.shape[2], \
                                                      filter = filter, \
                                                      conditions = 'prolongation' )[1] * support_tube

        norm_wr = np.sqrt( np.sum( wr**2 ) )
        sum_wr = np.sum( wr )

        return reconstructed, sum_wr, norm_wr

    def CG_minimization_wolfe_synthesis_options(self, wavelet_datacube, label_datacube, filter, 
                                              max_iter=50, gtol=1e-6, verbose=False,
                                              use_sum_synthesis=True, core_weight=2.0):
        '''
        CG minimization with optional sum synthesis for better small source core restoration.
        
        Parameters:
        -----------
        use_sum_synthesis : bool (default=True)
            If True, use sum operator for synthesis instead of adjoint
        core_weight : float (default=2.0)
            Weight factor for fine-scale coefficients to enhance cores
        '''
        
        wavelet_tube, support_tube, label_tube = self.tubes(wavelet_datacube, label_datacube)
        n_levels = wavelet_tube.shape[2]
        
        # Get the correct dimensions from the tubes
        tube_shape = wavelet_tube.shape[:2]  # (y_size, x_size)
        
        if verbose:
            print(f"Wavelet tube shape: {wavelet_tube.shape}")
            print(f"Support tube shape: {support_tube.shape}")
            print(f"Using {'sum' if use_sum_synthesis else 'adjoint'} synthesis")
            print(f"Core weight factor: {core_weight}")
        
        def synthesis_operator(wavelet_coeffs, filter):
            """
            Synthesis operator with sum or adjoint option
            """
            if use_sum_synthesis:
                # Sum synthesis: direct summation of wavelet planes
                # This preserves flux better and enhances small source cores
                reconstruction = np.zeros(tube_shape, dtype=np.float64)
                
                for level in range(wavelet_coeffs.shape[2]):
                    # Weight fine scales more heavily for core enhancement
                    if level == 0:  # Finest scale (contains cores)
                        weight = core_weight
                    else:
                        weight = 1.0
                    
                    # Add weighted wavelet plane
                    reconstruction += weight * wavelet_coeffs[:, :, level]
                
                return reconstruction
            else:
                # Standard adjoint synthesis
                return self.synthesis_adjoint(wavelet_coeffs, filter=filter)
        
        def analysis_operator(image, filter):
            """
            Analysis operator (forward transform)
            """
            atrous_result = atrous(image, n_levels=n_levels, filter=filter,
                                  conditions='prolongation')
            return atrous_result[1]  # Return wavelet coefficients
        
        def objective_and_gradient(x):
            """Combined function with sum synthesis option"""
            # Reshape to the correct tube dimensions
            x_2d = x.reshape(tube_shape)
            x_2d = np.maximum(x_2d, 0.0)  # Non-negativity constraint
            
            try:
                # Forward operator: get wavelet coefficients from reconstruction
                Ax = analysis_operator(x_2d, filter)
                
                # Ensure shapes match
                if Ax.shape != wavelet_tube.shape:
                    if verbose:
                        print(f"Shape mismatch: Ax {Ax.shape} vs wavelet_tube {wavelet_tube.shape}")
                    min_levels = min(Ax.shape[2], wavelet_tube.shape[2])
                    Ax = Ax[:, :, :min_levels]
                    wavelet_tube_local = wavelet_tube[:, :, :min_levels]
                    support_tube_local = support_tube[:, :, :min_levels]
                else:
                    wavelet_tube_local = wavelet_tube
                    support_tube_local = support_tube
                
                # Apply support and compute residual
                Ax_supported = Ax * support_tube_local
                residual = Ax_supported - wavelet_tube_local
                
            except Exception as e:
                if verbose:
                    print(f"Error in forward operator: {e}")
                raise
            
            # Objective function: 0.5 * ||residual||^2
            obj_val = 0.5 * np.sum(residual**2)
            
            # Gradient computation depends on synthesis method
            if use_sum_synthesis:
                # For sum synthesis, gradient is the direct sum of residuals with weights
                grad_2d = np.zeros(tube_shape, dtype=np.float64)
                
                for level in range(residual.shape[2]):
                    if level == 0:  # Finest scale
                        weight = core_weight
                    else:
                        weight = 1.0
                    
                    # Sum weighted residuals for gradient
                    grad_2d += weight * residual[:, :, level] * support_tube_local[:, :, level]
            else:
                # Standard adjoint gradient
                grad_2d = self.synthesis_adjoint(residual, filter=filter)
            
            # Ensure gradient has correct shape
            if grad_2d.shape != x_2d.shape:
                if verbose:
                    print(f"Gradient shape mismatch: {grad_2d.shape} vs {x_2d.shape}")
                from scipy.ndimage import zoom
                zoom_factors = (x_2d.shape[0] / grad_2d.shape[0], 
                              x_2d.shape[1] / grad_2d.shape[1])
                grad_2d = zoom(grad_2d, zoom_factors, order=1)
            
            grad = grad_2d.flatten()
            
            return obj_val, grad
        
        # Initialize reconstruction
        try:
            if use_sum_synthesis:
                # For sum synthesis, start with weighted sum of wavelet coefficients
                initial_recon = synthesis_operator(wavelet_tube * support_tube, filter)
            else:
                # Standard adjoint initialization
                initial_recon = self.synthesis_adjoint(wavelet_tube * support_tube, filter=filter)
            
            # Ensure correct shape
            if initial_recon.shape != tube_shape:
                if verbose:
                    print(f"Resizing initial reconstruction from {initial_recon.shape} to {tube_shape}")
                from scipy.ndimage import zoom
                zoom_factors = (tube_shape[0] / initial_recon.shape[0], 
                              tube_shape[1] / initial_recon.shape[1])
                initial_recon = zoom(initial_recon, zoom_factors, order=1)
            
            reconstructed = np.maximum(initial_recon, 0.0)
            
            # Normalize initial reconstruction to prevent over-brightning
            if np.sum(reconstructed) > 0:
                flux_ratio = np.sum(wavelet_tube * support_tube) / np.sum(reconstructed)
                if flux_ratio > 0.1 and flux_ratio < 10.0:  # Reasonable range
                    reconstructed *= flux_ratio
            
        except Exception as e:
            if verbose:
                print(f"Error in initial reconstruction: {e}")
            reconstructed = np.zeros(tube_shape, dtype=np.float64)
        
        x = reconstructed.flatten()
        
        if verbose:
            print(f"Initial reconstruction shape: {reconstructed.shape}")
            print(f"Initial flux: {np.sum(reconstructed):.6f}")
            print(f"Target flux: {np.sum(wavelet_tube * support_tube):.6f}")
        
        # Initial evaluation
        try:
            f_val, grad = objective_and_gradient(x)
        except Exception as e:
            print(f"Error in initial evaluation: {e}")
            raise
        
        search_direction = -grad
        grad_norm_init = np.linalg.norm(grad)
        
        if verbose:
            print(f"Initial objective: {f_val:.6e}")
            print(f"Initial gradient norm: {grad_norm_init:.6e}")
        
        # Store best solution found
        best_x = x.copy()
        best_f = f_val
        
        # CG iterations
        for iteration in range(max_iter):
            
            grad_norm = np.linalg.norm(grad)
            
            # Convergence check
            if grad_norm < gtol * max(1.0, grad_norm_init):
                if verbose:
                    print(f"Convergence at iteration {iteration}")
                break
            
            # Adaptive step size based on iteration
            max_step = 1.0 if iteration < 10 else 0.5
            
            # Simple but robust line search
            step_size = max_step
            c1 = 1e-4  # Armijo parameter
            rho = 0.5  # Step reduction factor
            
            directional_deriv = np.dot(grad, search_direction)
            
            for line_iter in range(25):
                try:
                    # Test step
                    x_test = x + step_size * search_direction
                    x_2d_test = x_test.reshape(tube_shape)
                    x_2d_test = np.maximum(x_2d_test, 0.0)
                    x_test = x_2d_test.flatten()
                    
                    f_test, _ = objective_and_gradient(x_test)
                    
                    # Armijo condition
                    if f_test <= f_val + c1 * step_size * directional_deriv:
                        break
                        
                    step_size *= rho
                    if step_size < 1e-12:
                        break
                        
                except:
                    step_size *= rho
                    if step_size < 1e-12:
                        break
            
            if step_size < 1e-12:
                if verbose:
                    print(f"Line search failed at iteration {iteration}")
                # Use best solution found so far
                x = best_x.copy()
                break
            
            # Update solution
            x_new = x + step_size * search_direction
            x_2d = x_new.reshape(tube_shape)
            x_2d = np.maximum(x_2d, 0.0)
            x = x_2d.flatten()
            
            # Store previous values
            f_prev = f_val
            grad_prev = grad.copy()
            
            # Update function and gradient
            try:
                f_val, grad = objective_and_gradient(x)
                
                # Track best solution
                if f_val < best_f:
                    best_x = x.copy()
                    best_f = f_val
                    
            except Exception as e:
                if verbose:
                    print(f"Error in iteration {iteration}: {e}")
                x = best_x.copy()
                break
            
            # Polak-Ribiere beta with automatic restart
            grad_prev_norm_sq = np.sum(grad_prev**2)
            if grad_prev_norm_sq > 1e-12:
                beta = max(0.0, np.sum(grad * (grad - grad_prev)) / grad_prev_norm_sq)
            else:
                beta = 0.0
            
            # Restart if beta is too small or too large
            if beta < 1e-8 or beta > 10.0:
                beta = 0.0
            
            # Update search direction
            search_direction = -grad + beta * search_direction
            
            if verbose and (iteration % 5 == 0 or iteration < 3):
                flux_current = np.sum(x.reshape(tube_shape))
                print(f"Iter {iteration:3d}: f = {f_val:.6e}, "
                      f"||grad|| = {grad_norm:.6e}, "
                      f"step = {step_size:.6e}, "
                      f"flux = {flux_current:.6f}")
        
        # Use best solution found
        reconstructed = best_x.reshape(tube_shape)
        
        # Compute final residuals
        try:
            final_wavelet = analysis_operator(reconstructed, filter) * support_tube
            wr = final_wavelet - wavelet_tube
        except:
            wr = np.zeros_like(wavelet_tube)
        
        norm_wr = np.linalg.norm(wr)
        sum_wr = np.sum(wr)
        
        if verbose:
            print(f"\nCG-Synthesis completed in {iteration + 1} iterations")
            print(f"Final objective: {best_f:.6e}")
            print(f"Final flux: {np.sum(reconstructed):.6f}")
            print(f"Target flux: {np.sum(wavelet_tube * support_tube):.6f}")
            print(f"Residual norm: {norm_wr:.6e}")
            print(f"Max pixel value: {np.max(reconstructed):.6f}")
        
        return reconstructed, sum_wr, norm_wr

    def CG_minimization_advanced_synthesis(self, wavelet_datacube, label_datacube, filter, 
                                         max_iter=50, gtol=1e-6, verbose=False,
                                         synthesis_method='adaptive_weight', **kwargs):
        '''
        CG minimization with advanced synthesis operators for optimal source reconstruction.
        
        Parameters:
        -----------
        synthesis_method : str
            'adaptive_weight' - Scale-adaptive weighting
            'regularized_inverse' - Tikhonov regularized inverse
            'morphological' - Morphology-aware synthesis
            'multiscale_fusion' - Multiscale information fusion
            'sparse_promote' - Sparsity-promoting synthesis
            'flux_conserving' - Flux-conserving weighted synthesis
        '''
        
        wavelet_tube, support_tube, label_tube = self.tubes(wavelet_datacube, label_datacube)
        n_levels = wavelet_tube.shape[2]
        tube_shape = wavelet_tube.shape[:2]
        
        def adaptive_weighted_synthesis(wavelet_coeffs, filter):
            """
            Scale-adaptive weighted synthesis based on local content analysis
            """
            reconstruction = np.zeros(tube_shape, dtype=np.float64)
            
            for level in range(wavelet_coeffs.shape[2]):
                coeff_plane = wavelet_coeffs[:, :, level]
                
                # Adaptive weight based on local variance and energy
                local_var = uniform_filter(coeff_plane**2, size=3) - uniform_filter(coeff_plane, size=3)**2
                energy_map = uniform_filter(coeff_plane**2, size=5)
                
                # Higher weights for:
                # - Fine scales (small level index)
                # - High local variance (edges, cores)
                # - High energy regions
                scale_weight = np.exp(-level * 0.5)  # Exponential decay with scale
                content_weight = 1.0 + 2.0 * (local_var / (np.max(local_var) + 1e-10))
                energy_weight = 1.0 + (energy_map / (np.max(energy_map) + 1e-10))
                
                combined_weight = scale_weight * content_weight * energy_weight
                reconstruction += combined_weight * coeff_plane
                
            return reconstruction
        
        def regularized_inverse_synthesis(wavelet_coeffs, filter, lambda_reg=0.01):
            """
            Tikhonov regularized inverse synthesis
            """
            # Create analysis matrix A (conceptual - in practice we use iterative methods)
            def apply_analysis(x):
                return atrous(x, n_levels=n_levels, filter=filter, conditions='prolongation')[1]
            
            def apply_adjoint(w):
                # Standard adjoint
                return self.synthesis_adjoint(w, filter=filter)
            
            # Iterative solution of (A^T*A + λI)x = A^T*b using CG
            b = wavelet_coeffs.flatten()
            x = np.zeros(tube_shape).flatten()
            
            # A few CG iterations for regularized inverse
            r = apply_adjoint(wavelet_coeffs).flatten() - lambda_reg * x
            p = r.copy()
            
            for _ in range(10):  # Inner CG iterations
                x_2d = x.reshape(tube_shape)
                Ap_coeffs = apply_analysis(x_2d)
                Ap = apply_adjoint(Ap_coeffs).flatten() + lambda_reg * p
                
                alpha = np.dot(r, r) / (np.dot(p, Ap) + 1e-12)
                x = x + alpha * p
                r_new = r - alpha * Ap
                
                if np.linalg.norm(r_new) < 1e-6:
                    break
                    
                beta = np.dot(r_new, r_new) / (np.dot(r, r) + 1e-12)
                p = r_new + beta * p
                r = r_new
            
            return x.reshape(tube_shape)
        
        def morphological_synthesis(wavelet_coeffs, filter):
            """
            Morphology-aware synthesis using mathematical morphology
            """
            from scipy.ndimage import binary_opening, binary_closing, binary_dilation
            from skimage.morphology import disk
            
            reconstruction = np.zeros(tube_shape, dtype=np.float64)
            
            for level in range(wavelet_coeffs.shape[2]):
                coeff_plane = wavelet_coeffs[:, :, level]
                
                # Create morphological mask based on significant coefficients
                threshold = np.std(coeff_plane) * 2.0
                mask = np.abs(coeff_plane) > threshold
                
                # Morphological operations based on scale
                if level == 0:  # Finest scale - preserve point sources
                    selem = disk(1)
                    mask = binary_opening(mask, selem)
                    weight_map = 3.0 * mask.astype(float) + 0.5
                elif level <= 2:  # Medium scales - enhance extended structures
                    selem = disk(level + 1)
                    mask = binary_closing(mask, selem)
                    weight_map = 2.0 * mask.astype(float) + 0.7
                else:  # Coarse scales - background
                    selem = disk(level)
                    mask = binary_dilation(mask, selem)
                    weight_map = 1.0 * mask.astype(float) + 0.3
                
                reconstruction += weight_map * coeff_plane
                
            return reconstruction
        
        def multiscale_fusion_synthesis(wavelet_coeffs, filter):
            """
            Multiscale information fusion with cross-scale dependencies
            """
            reconstruction = np.zeros(tube_shape, dtype=np.float64)
            
            # Compute cross-scale correlation
            scale_correlations = np.zeros((n_levels, n_levels))
            for i in range(n_levels):
                for j in range(n_levels):
                    if i != j:
                        # Resize to common size for correlation
                        ci = wavelet_coeffs[:, :, i]
                        cj = wavelet_coeffs[:, :, j]
                        correlation = np.corrcoef(ci.flatten(), cj.flatten())[0, 1]
                        scale_correlations[i, j] = abs(correlation) if not np.isnan(correlation) else 0
            
            for level in range(n_levels):
                coeff_plane = wavelet_coeffs[:, :, level]
                
                # Base weight (fine scales preferred)
                base_weight = np.exp(-level * 0.3)
                
                # Cross-scale influence
                cross_scale_weight = 1.0 + np.sum(scale_correlations[level, :])
                
                # Local structure analysis
                # Detect point sources (high kurtosis)
                local_kurtosis = uniform_filter(coeff_plane**4, size=3) / (uniform_filter(coeff_plane**2, size=3)**2 + 1e-10)
                structure_weight = 1.0 + np.tanh((local_kurtosis - 3.0) / 2.0)  # Enhance high kurtosis regions
                
                combined_weight = base_weight * cross_scale_weight * structure_weight
                reconstruction += combined_weight * coeff_plane
                
            return reconstruction
        
        def sparse_promoting_synthesis(wavelet_coeffs, filter, sparsity_param=0.1):
            """
            Sparsity-promoting synthesis using soft thresholding
            """
            reconstruction = np.zeros(tube_shape, dtype=np.float64)
            
            for level in range(n_levels):
                coeff_plane = wavelet_coeffs[:, :, level]
                
                # Adaptive threshold based on scale and noise estimation
                noise_level = np.median(np.abs(coeff_plane)) / 0.6745  # Robust noise estimate
                scale_factor = 2.0 ** (-level * 0.5)  # Finer scales have lower thresholds
                threshold = sparsity_param * noise_level * scale_factor
                
                # Soft thresholding
                sign = np.sign(coeff_plane)
                magnitude = np.abs(coeff_plane)
                thresholded = sign * np.maximum(magnitude - threshold, 0.0)
                
                # Scale-dependent weight
                scale_weight = (2.0 - level/n_levels)  # Higher weight for finer scales
                
                reconstruction += scale_weight * thresholded
                
            return reconstruction
        
        def flux_conserving_synthesis(wavelet_coeffs, filter):
            """
            Flux-conserving weighted synthesis with energy normalization
            """
            reconstruction = np.zeros(tube_shape, dtype=np.float64)
            total_energy = np.sum(wavelet_coeffs**2)
            
            # Compute energy per scale
            scale_energies = np.array([np.sum(wavelet_coeffs[:, :, i]**2) for i in range(n_levels)])
            scale_weights = scale_energies / (total_energy + 1e-12)
            
            for level in range(n_levels):
                coeff_plane = wavelet_coeffs[:, :, level]
                
                # Combine multiple weighting factors
                # 1. Energy-based weight
                energy_weight = scale_weights[level]
                
                # 2. Resolution-based weight (favor fine scales)
                resolution_weight = np.exp(-level * 0.4)
                
                # 3. Content-based weight (enhance significant coefficients)
                significance_map = np.abs(coeff_plane) / (np.max(np.abs(coeff_plane)) + 1e-10)
                content_weight = 1.0 + significance_map**2
                
                # 4. Spatial coherence weight
                coherence_map = uniform_filter(np.abs(coeff_plane), size=3)
                spatial_weight = 1.0 + coherence_map / (np.max(coherence_map) + 1e-10)
                
                combined_weight = energy_weight * resolution_weight * content_weight * spatial_weight
                reconstruction += combined_weight * coeff_plane
            
            # Flux conservation: normalize to preserve total flux
            original_flux = np.sum(wavelet_coeffs * support_tube)
            if np.sum(reconstruction) > 0:
                reconstruction *= original_flux / np.sum(reconstruction)
                
            return reconstruction
        
        # Dictionary of synthesis methods
        synthesis_methods = {
            'adaptive_weight': adaptive_weighted_synthesis,
            'regularized_inverse': lambda w, f: regularized_inverse_synthesis(w, f, kwargs.get('lambda_reg', 0.01)),
            'morphological': morphological_synthesis,
            'multiscale_fusion': multiscale_fusion_synthesis,
            'sparse_promote': lambda w, f: sparse_promoting_synthesis(w, f, kwargs.get('sparsity_param', 0.1)),
            'flux_conserving': flux_conserving_synthesis,
            'sum': lambda w, f: np.sum(w, axis=2),  # Simple sum for comparison
            'adjoint': lambda w, f: self.synthesis_adjoint(w, filter=f)  # Standard adjoint
        }
        
        if synthesis_method not in synthesis_methods:
            raise ValueError(f"Unknown synthesis method: {synthesis_method}")
        
        synthesis_operator = synthesis_methods[synthesis_method]
        
        # Import required modules
        from scipy.ndimage import uniform_filter
        import numpy as np
        
        # Rest of the CG algorithm using the selected synthesis operator
        def objective_and_gradient(x):
            x_2d = x.reshape(tube_shape)
            x_2d = np.maximum(x_2d, 0.0)
            
            # Forward analysis
            Ax = atrous(x_2d, n_levels=n_levels, filter=filter, conditions='prolongation')[1]
            
            # Handle shape mismatch
            if Ax.shape != wavelet_tube.shape:
                min_levels = min(Ax.shape[2], wavelet_tube.shape[2])
                Ax = Ax[:, :, :min_levels]
                wavelet_tube_local = wavelet_tube[:, :, :min_levels]
                support_tube_local = support_tube[:, :, :min_levels]
            else:
                wavelet_tube_local = wavelet_tube
                support_tube_local = support_tube
            
            # Residual
            Ax_supported = Ax * support_tube_local
            residual = Ax_supported - wavelet_tube_local
            obj_val = 0.5 * np.sum(residual**2)
            
            # Gradient using selected synthesis operator
            grad_2d = synthesis_operator(residual, filter)
            
            # Ensure correct shape
            if grad_2d.shape != x_2d.shape:
                from scipy.ndimage import zoom
                zoom_factors = (x_2d.shape[0] / grad_2d.shape[0], 
                              x_2d.shape[1] / grad_2d.shape[1])
                grad_2d = zoom(grad_2d, zoom_factors, order=1)
            
            grad = grad_2d.flatten()
            return obj_val, grad
        
        # Initialize using the selected synthesis operator
        try:
            initial_recon = synthesis_operator(wavelet_tube * support_tube, filter)
            if initial_recon.shape != tube_shape:
                from scipy.ndimage import zoom
                zoom_factors = (tube_shape[0] / initial_recon.shape[0], 
                              tube_shape[1] / initial_recon.shape[1])
                initial_recon = zoom(initial_recon, zoom_factors, order=1)
            reconstructed = np.maximum(initial_recon, 0.0)
        except Exception as e:
            if verbose:
                print(f"Error in initialization: {e}")
            reconstructed = np.zeros(tube_shape, dtype=np.float64)
        
        x = reconstructed.flatten()
        
        if verbose:
            print(f"Using synthesis method: {synthesis_method}")
            print(f"Initial flux: {np.sum(reconstructed):.6f}")
        
        # Run CG optimization (same as before)
        f_val, grad = objective_and_gradient(x)
        search_direction = -grad
        grad_norm_init = np.linalg.norm(grad)
        
        best_x = x.copy()
        best_f = f_val
        
        for iteration in range(max_iter):
            grad_norm = np.linalg.norm(grad)
            
            if grad_norm < gtol * max(1.0, grad_norm_init):
                break
            
            # Line search
            step_size = 1.0
            directional_deriv = np.dot(grad, search_direction)
            
            for line_iter in range(20):
                try:
                    x_test = x + step_size * search_direction
                    x_2d_test = np.maximum(x_test.reshape(tube_shape), 0.0)
                    f_test, _ = objective_and_gradient(x_2d_test.flatten())
                    
                    if f_test <= f_val + 1e-4 * step_size * directional_deriv:
                        break
                    step_size *= 0.5
                except:
                    step_size *= 0.5
                
                if step_size < 1e-12:
                    break
            
            # Update
            x = x + step_size * search_direction
            x_2d = np.maximum(x.reshape(tube_shape), 0.0)
            x = x_2d.flatten()
            
            f_prev, grad_prev = f_val, grad.copy()
            f_val, grad = objective_and_gradient(x)
            
            if f_val < best_f:
                best_x, best_f = x.copy(), f_val
            
            # Polak-Ribiere update
            beta = max(0.0, np.dot(grad, grad - grad_prev) / (np.dot(grad_prev, grad_prev) + 1e-12))
            search_direction = -grad + beta * search_direction
            
            if verbose and iteration % 10 == 0:
                print(f"Iter {iteration}: f = {f_val:.6e}, ||grad|| = {grad_norm:.6e}")
        
        reconstructed = best_x.reshape(tube_shape)
        
        # Final residuals
        try:
            final_wavelet = atrous(reconstructed, n_levels=n_levels, filter=filter,
                                  conditions='prolongation')[1] * support_tube
            wr = final_wavelet - wavelet_tube
        except:
            wr = np.zeros_like(wavelet_tube)
        
        if verbose:
            print(f"Synthesis method '{synthesis_method}' completed")
            print(f"Final flux: {np.sum(reconstructed):.6f}")
            print(f"Peak value: {np.max(reconstructed):.6f}")
        
        return reconstructed, np.sum(wr), np.linalg.norm(wr)

    from scipy.stats import bootstrap
    from scipy.linalg import inv
    
    def CG_minimization_wolfe_synthesis_options_with_uncertainty(self, wavelet_datacube, label_datacube, filter, 
                                                               max_iter=50, gtol=1e-6, verbose=False,
                                                               use_sum_synthesis=True, core_weight=2.0,
                                                               compute_uncertainty=True, n_bootstrap=100,
                                                               confidence_level=0.68):
        '''
        CG minimization with comprehensive uncertainty estimation.
        
        Parameters:
        -----------
        compute_uncertainty : bool (default=True)
            Whether to compute uncertainty estimates
        n_bootstrap : int (default=100)
            Number of bootstrap samples for uncertainty estimation
        confidence_level : float (default=0.68)
            Confidence level for error bars (0.68 = 1σ, 0.95 = 2σ)
        
        Returns:
        --------
        reconstructed : array
            Reconstructed image
        sum_wr : float
            Sum of weighted residuals
        norm_wr : float
            Norm of weighted residuals
        uncertainty_dict : dict
            Comprehensive uncertainty measurements
        '''
        
        wavelet_tube, support_tube, label_tube = self.tubes(wavelet_datacube, label_datacube)
        n_levels = wavelet_tube.shape[2]
        tube_shape = wavelet_tube.shape[:2]
        
        if verbose:
            print(f"Computing reconstruction with uncertainty analysis")
            print(f"Bootstrap samples: {n_bootstrap}")
            print(f"Confidence level: {confidence_level}")
        
        # Original reconstruction
        def perform_reconstruction(wavelet_data, support_data):
            """Wrapper function for single reconstruction"""
            def synthesis_operator(wavelet_coeffs, filter):
                if use_sum_synthesis:
                    reconstruction = np.zeros(tube_shape, dtype=np.float64)
                    for level in range(wavelet_coeffs.shape[2]):
                        scale_weight = core_weight ** (n_levels - level - 1)
                        reconstruction += scale_weight * wavelet_coeffs[:, :, level]
                    return reconstruction
                else:
                    try:
                        return atrous_adj(wavelet_coeffs, n_levels=n_levels, filter=filter)
                    except:
                        reconstruction = np.zeros(tube_shape, dtype=np.float64)
                        for level in range(wavelet_coeffs.shape[2]):
                            reconstruction += wavelet_coeffs[:, :, level]
                        return reconstruction
            
            def analysis_operator(image, filter):
                return atrous(image, n_levels=n_levels, filter=filter, conditions='prolongation')[1]

            
            # Simplified CG for uncertainty analysis
            x = np.zeros(tube_shape, dtype=np.float64)
            target_wavelet = wavelet_data * support_data
            
            for iteration in range(min(max_iter, 20)):  # Reduced iterations for bootstrap
                try:
                    predicted_wavelet = analysis_operator(x, filter) * support_data
                    residual = target_wavelet - predicted_wavelet
                    
                    # Simple gradient descent step
                    x += 0.1 * synthesis_operator(residual, filter)
                    x = np.maximum(x, 0.0)  # Non-negativity
                    
                    if np.linalg.norm(residual) < gtol:
                        break
                except:
                    break
                    
            return x
        
        # Main reconstruction
        reconstructed = perform_reconstruction(wavelet_tube, support_tube)
        
        # Compute final residuals
        try:
            if use_sum_synthesis:
                predicted_wavelet = np.zeros_like(wavelet_tube)
                for level in range(n_levels):
                    scale_weight = core_weight ** (n_levels - level - 1)
                    # Simple analysis (differentiate the sum synthesis)
                    predicted_wavelet[:, :, level] = reconstructed / scale_weight
            else:
                predicted_wavelet = atrous(reconstructed, n_levels=n_levels, filter=filter,
                                         conditions='prolongation')[1]
            
            final_wavelet = predicted_wavelet * support_tube
            wr = final_wavelet - wavelet_tube
        except:
            wr = np.zeros_like(wavelet_tube)
        
        norm_wr = np.linalg.norm(wr)
        sum_wr = np.sum(wr)
                
        if compute_uncertainty:
            if verbose:
                print("Computing uncertainty estimates...")
            
            # 1. Bootstrap uncertainty
            bootstrap_reconstructions = []
            
            for i in range(n_bootstrap):
                if verbose and i % (n_bootstrap//4) == 0:
                    print(f"  Bootstrap sample {i+1}/{n_bootstrap}")
                
                # Add noise to wavelet coefficients
                conv_table = table_noise_stds(n_levels, wavelet_type = 'BSPL')
                noise_cube = np.zeros(wavelet_tube.shape)
                noise_pixels = wavelet_tube[:,:,0].flatten()
                noise_pixels = sigma_clip(noise_pixels, sigma = 3, sigma_lower = 3, masked = False)
                sigma = np.std(noise_pixels)
                mean = np.mean(noise_pixels)
                for level in range(n_levels):
                    noise_level = sigma * conv_table[level]
                    noise_cube[:, :, level] = np.random.normal(mean, noise_level, noise_cube[:, :, level].shape)
                noisy_wavelet = wavelet_tube + noise_cube
                
                try:
                    bootstrap_recon = perform_reconstruction(noisy_wavelet, support_tube)
                    bootstrap_reconstructions.append(bootstrap_recon)
                except:
                    bootstrap_reconstructions.append(np.zeros_like(reconstructed))
            
            bootstrap_reconstructions = np.array(bootstrap_reconstructions)
            
            # Statistical measures
            bootstrap_mean = np.mean(bootstrap_reconstructions, axis=0)
            bootstrap_std = np.std(bootstrap_reconstructions, axis=0)
            
            # Confidence intervals
            alpha = 1 - confidence_level
            lower_percentile = 100 * alpha / 2
            upper_percentile = 100 * (1 - alpha / 2)
            
            bootstrap_lower = np.percentile(bootstrap_reconstructions, lower_percentile, axis=0)
            bootstrap_upper = np.percentile(bootstrap_reconstructions, upper_percentile, axis=0)
            
            # 2. Analytical uncertainty (diagonal Fisher information approximation)
            try:
                # Simplified Fisher information matrix (diagonal approximation)
                fisher_diag = np.zeros_like(reconstructed)
                
                for level in range(n_levels):
                    support_level = support_tube[:, :, level]
                    noise_pixels = np.copy(wavelet_tube[:,:,0] * (1 - support_tube[:,:,0])).flatten()
                    sigma = np.std(noise_pixels)
                    noise_level = sigma * conv_table[level]
                    if use_sum_synthesis:
                        scale_weight = core_weight ** (n_levels - level - 1)
                        fisher_diag += (scale_weight**2 * support_level) / (noise_level**2)
                    else:
                        fisher_diag += support_level / (noise_level**2)
                
                # Analytical standard errors (diagonal elements only)
                analytical_std = np.sqrt(1.0 / (fisher_diag + 1e-12))
                analytical_std[fisher_diag == 0] = np.inf
                
            except:
                analytical_std = np.full_like(reconstructed, np.inf)
            
            # 3. Residual-based uncertainty
            residual_variance = np.var(wr)
            residual_std = np.sqrt(residual_variance)
            
            # 4. Signal-to-noise ratio
            snr_map = np.divide(reconstructed, bootstrap_std + 1e-12, 
                               out=np.zeros_like(reconstructed), where=bootstrap_std>1e-12)
            
            # 5. Relative uncertainties
            relative_std = np.divide(bootstrap_std, reconstructed + 1e-12,
                                   out=np.full_like(reconstructed, np.inf), 
                                   where=reconstructed>1e-12)
            
            # Summary statistics
            uncertainty_dict= {
                # Bootstrap results
                'bootstrap_std': bootstrap_std,
                'bootstrap_mean': bootstrap_mean,
                'bootstrap_lower': bootstrap_lower,
                'bootstrap_upper': bootstrap_upper,
                'bootstrap_reconstructions': bootstrap_reconstructions,  # For further analysis
                
                # Analytical results
                'analytical_std': analytical_std,
                'fisher_information': fisher_diag,
                
                # Derived quantities
                'snr_map': snr_map,
                'relative_uncertainty': relative_std,
                'residual_std': residual_std,
                
                # Summary statistics
                'total_flux_std': np.std([np.sum(r) for r in bootstrap_reconstructions]),
                'peak_value_std': np.std([np.max(r) for r in bootstrap_reconstructions]),
                'mean_snr': np.mean(snr_map[np.isfinite(snr_map) & (snr_map > 0)]),
                'median_relative_error': np.median(relative_std[np.isfinite(relative_std)]),
                
                # Quality metrics
                'uncertainty_coverage': np.mean((reconstructed >= bootstrap_lower) & 
                                              (reconstructed <= bootstrap_upper)),
                'high_snr_fraction': np.mean(snr_map > 3.0),
                'reliable_pixels': np.sum((bootstrap_std < 0.1 * reconstructed) & 
                                        (reconstructed > 3 * noise_level))
            }
            
            if verbose:
                print(f"\nUncertainty Analysis Results:")
                print(f"Total flux uncertainty: ±{uncertainty_dict['total_flux_std']:.6f}")
                print(f"Peak value uncertainty: ±{uncertainty_dict['peak_value_std']:.6f}")
                print(f"Mean SNR: {uncertainty_dict['mean_snr']:.2f}")
                print(f"Median relative error: {uncertainty_dict['median_relative_error']:.1%}")
                print(f"Coverage probability: {uncertainty_dict['uncertainty_coverage']:.1%}")
                print(f"High SNR pixels (>3σ): {uncertainty_dict['high_snr_fraction']:.1%}")
                print(f"Reliable pixels: {uncertainty_dict['reliable_pixels']}")
        
        if verbose:
            print(f"\nReconstruction completed")
            print(f"Final flux: {np.sum(reconstructed):.6f}")
            print(f"Peak value: {np.max(reconstructed):.6f}")
            print(f"Residual norm: {norm_wr:.6e}")
        
        return reconstructed, sum_wr, norm_wr

@ray.remote
def interscale_connectivity_para( interscale_maximum_list, region_list, wavelet_datacube, label_datacube, det_err_array, level_maximum, min_span = 3, max_span = 3, lvl_sep_big = 6, min_reg_size = 4, verbose = False):

    interscale_tree_list = []
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

        covered_region_list = [x for x in region_list if x.label in covered_label_list]

        connected_region_list = [x for x in region_list if
                         x.label in covered_label_list and
                         [x.x_max, x.y_max] in interscale_maximum.coords and
                         x.area >= min_reg_size and
                         interscale_maximum.level - x.level <= pmax_span - 1]

        span_levels = np.size(np.unique([ x.level for x in connected_region_list ]))

        if span_levels < pmin_span :
            #log.info('Rejected : level = %d span = %d' %(interscale_maximum.level, span_levels))
            n_rejected += 1
            continue

        # Output list of interscale tree and their clipped segmented label array
        interscale_tree_list.append(interscale_tree(interscale_maximum, connected_region_list, wavelet_datacube, label_datacube, det_err_array))

    return interscale_tree_list

def interscale_connectivity_serial( interscale_maximum_list, region_list, wavelet_datacube, label_datacube, det_err_array, level_maximum, min_span = 3, max_span = 3, lvl_sep_big = 6, min_reg_size = 4, verbose = False):

    interscale_tree_list = []
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
        
        covered_region_list = [x for x in region_list if x.label in covered_label_list]

        connected_region_list = [x for x in region_list if
                         x.label in covered_label_list and
                         [x.x_max, x.y_max] in interscale_maximum.coords and
                         x.area >= min_reg_size and
                         interscale_maximum.level - x.level <= pmax_span - 1]
        
        span_levels = np.size(np.unique([ x.level for x in connected_region_list ]))

        if span_levels < pmin_span :
            #log.info('Rejected : level = %d span = %d' %(interscale_maximum.level, span_levels))
            n_rejected += 1
            continue

        interscale_tree_list.append(interscale_tree(interscale_maximum, connected_region_list, wavelet_datacube, label_datacube, det_err_array))
        
    return interscale_tree_list

def make_interscale_trees(region_list, wavelet_datacube, label_datacube, det_err_array, tau = 0.8, min_span = 3, max_span = 3, lvl_sep_big = 6, monomodality = False, threshold_rel = 0.05, min_reg_size = 4, size_patch = 100, n_cpus = 1, verbose = False):

    interscale_tree_list = []

    region_list.sort(key = lambda x: x.norm_max_intensity, reverse = True)

    i = 0
    levels_rejected = [ 0, label_datacube.z_size ]
    threshold_maximum = region_list[0].norm_max_intensity * tau
    level_maximum = region_list[0].level

    if len(region_list) > 1:
        while region_list[i].level in levels_rejected :
            i += 1
            if i >= len(region_list):
                log = logging.getLogger(__name__)
                log.info("No interscale maximum found. Please consider lowering parameters 'tau' or 'min_span' if this keeps happening at every level.")
                return None # Regions are all in rejected levels (0 or last)
                
        threshold_maximum = region_list[i].norm_max_intensity * tau
        level_maximum = region_list[i].level
        
        interscale_maximum_list = [x for x in region_list if 
                            x.norm_max_intensity >= threshold_maximum and
                            x.level == level_maximum and
                            x.area >= min_reg_size]
            
    else:
        interscale_maximum_list = region_list
    

    log = logging.getLogger(__name__)
    log.info('Estimating global interscale maxima: %d found.' %(len(interscale_maximum_list)))

    if (len(interscale_maximum_list) <= size_patch) or (n_cpus == 1):
        log.info('Not activating Ray store.')
        # Connectivity
        interscale_tree_list = interscale_connectivity_serial( interscale_maximum_list = interscale_maximum_list,  \
                                                               region_list = region_list,  \
                                                               wavelet_datacube = wavelet_datacube, \
                                                               label_datacube = label_datacube,  \
                                                               det_err_array = det_err_array, \
                                                               level_maximum = level_maximum,  \
                                                               min_span = min_span,  \
                                                               max_span = max_span,  \
                                                               lvl_sep_big = lvl_sep_big, \
                                                               min_reg_size = min_reg_size, \
                                                               verbose = verbose )

    else:
        logging.info('Activating Ray store with patches of size %d.'%(size_patch))
        id_rl = ray.put(region_list)
        id_wdc = ray.put(wavelet_datacube)
        id_ldc = ray.put(label_datacube)
        id_level_maximum = ray.put(level_maximum)
        id_min_span = ray.put(min_span)
        id_max_span = ray.put(max_span)
        id_lvl_sep_big = ray.put(lvl_sep_big)
        id_min_reg_size = ray.put(min_reg_size)
        id_verbose = ray.put(verbose)
        id_det_err_array = ray.put(det_err_array)
        
        interscale_maximum_patch = []
        interscale_tree_patch = []
        
        for interscale_maximum in interscale_maximum_list:

            interscale_maximum_patch.append(interscale_maximum)
            
            # Launch patches of "size_patch" interscale maxima
            if len(interscale_maximum_patch) >= size_patch:
                interscale_tree_patch.append( interscale_connectivity_para.remote( interscale_maximum_list = interscale_maximum_patch,  \
                                                                                   region_list = id_rl,  \
                                                                                   wavelet_datacube = id_wdc,  \
                                                                                   label_datacube = id_ldc,  \
                                                                                   det_err_array = id_det_err_array, \
                                                                                   level_maximum = id_level_maximum,  \
                                                                                   min_span = id_min_span,  \
                                                                                   max_span = id_max_span,  \
                                                                                   lvl_sep_big = id_lvl_sep_big, \
                                                                                   min_reg_size = id_min_reg_size, \
                                                                                   verbose = id_verbose ) )
                interscale_maximum_patch = []

        # Launch leftovers
        if interscale_maximum_patch:
            interscale_tree_patch.append( interscale_connectivity_para.remote( interscale_maximum_list = interscale_maximum_patch,  \
                                                                               region_list = id_rl,  \
                                                                               wavelet_datacube = id_wdc,  \
                                                                               label_datacube = id_ldc,  \
                                                                               det_err_array = id_det_err_array, \
                                                                               level_maximum = id_level_maximum,  \
                                                                               min_span = id_min_span,  \
                                                                               max_span = id_max_span,  \
                                                                               lvl_sep_big = id_lvl_sep_big, \
                                                                               min_reg_size = id_min_reg_size, \
                                                                               verbose = id_verbose ) )
        
        for id_patch in interscale_tree_patch:
            patch = ray.get( id_patch )
            interscale_tree_list = interscale_tree_list + patch

    if verbose == True:
        log.info('%d rejected.' %(len(interscale_maximum_list) - len(interscale_tree_list)))
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

    hdu = fits.open('/home/aellien/dawis/dawis/gallery/tidal_group_f814w_sci.fits')
    #hdu = fits.open('/home/aellien/Euclid_ERO/data/Euclid-NISP-Stack-ERO-Abell2390.DR3/Euclid-NISP-H-ERO-Abell2390-LSB.DR3.crop.fits')
    im = hdu[0].data
    header = hdu[0].header
    n_levels = int(np.min(np.floor(np.log2(im.shape))))
    print(im.shape, n_levels, 2**n_levels)
    n_levels = 3

    sigma, mean, gain = pg_noise_bissection(im, max_err = 1E-3, n_sigmas = 3)
    aim = anscombe_transform(im, sigma, mean, gain)
    acdc, awdc = bspl_atrous(aim, n_levels, header)
    sdc = hard_threshold(awdc, n_sigmas = 5)
    ldc = label_regions(sdc)
    cdc, wdc = bspl_atrous(im, n_levels, header)
    rl = make_regions_full_props(wdc, ldc, verbose = False)
    print(len(rl))
    lit = make_interscale_trees(rl, wdc, ldc, tau = 0.1, monomodality = True, verbose = False)

    for it in lit[1:]:

        #it.plot(wdc, ldc)
        #print(it.extent)

        bspl = 1 / 16. * np.array([ 1, 4, 6, 4, 1 ])
        haar = 1 / 2. * np.array([ 1, 0, 1 ])

        #sim = im[it.bbox[0]:it.bbox[2], it.bbox[1]:it.bbox[3]]
        #scores = []
        #ssims = []
        startTime = datetime.now()
        rima = it.CG_minimization( wdc, ldc, filter = bspl, synthesis_operator = 'ADJOINT', step_size = 'FR' )
        #scores.append(20 * np.log10(  np.sum(np.sqrt(sim**2)) / ( np.sum(np.sqrt((rima - sim)**2)) ) ))
        #ssims.append( ssim(rima, sim) )
        print(datetime.now() - startTime)

        startTime = datetime.now()
        numba_rima = it.CG_minimization( wdc, ldc, filter = bspl, synthesis_operator = 'ADJOINT', step_size = 'FR' )
        #startTime = datetime.now()
        #rimb = it.CG_minimization( wdc, ldc, filter = bspl, synthesis_operator = 'SUM', step_size = 'FR' )
        #scores.append(20 * np.log10(  np.sum(np.sqrt(sim**2)) / ( np.sum(np.sqrt((rimb - sim)**2)) ) ))
        #ssims.append( ssim(rimb, sim) )

        #print(it.x_size, it.y_size, it.bbox)
        #wt, st, lt = it.tubes(wdc, ldc)
        #it.bayesian_minimization( im, wdc, ldc, bspl, verbose = True )

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
