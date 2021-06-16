import os
import dawis

def test():

    indir = os.path.join(dawis.__path__[0], 'gallery')
    infile = 'tidal_group_f814w_sci.fits'
    outdir = input('DAWIS - In which directory should I write ?\n')
    n_cpus = int(input('DAWIS - How many CPUs available ?\n'))

    tau = 0.1   # Relative Threshold
    gamma = 0.5   # Attenuation factor
    ceps = 1E-4    # Convergence value for epsilon
    n_levels = 9    # Number of wavelet scales
    min_span = 2    # Minimum of wavelet scales spanned by an interscale tree (must be >= 1)
    max_span = 3    # Maximum number of wavelet scales spanned by an interscale tree
    lvl_sep_big = 5     # Scale at wich mix_span & max_span are set to 1
    extent_sep = 0.1    # Ratio n_pix/vignet under which the Haar wavelet is used for restoration
    lvl_sep_lin = 2     # Wavelet scale under which the Haar wavelet can be used for restoration
    max_iter = 500      # Maximum number of iterations
    data_dump = True    # Write data at each iteration /!\ demands lot of space on hardware /!\
    gif = True      # Make gifs of the run (need data_dump = True)
    starting_level = 2 # Starting wavelet scale (this is the third scale - Python convention 0 1 2)

    dawis.synthesis_by_analysis( indir = indir, infile = infile, outdir = outdir, n_cpus = n_cpus, n_levels = n_levels, \
                                        tau = tau, gamma = gamma, ceps = ceps, min_span = min_span, \
                                        max_span = max_span, lvl_sep_big = lvl_sep_big, max_iter = max_iter, \
                                        data_dump = data_dump, gif = gif )
