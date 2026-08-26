# DAWIS — Ubiquitous Language

Glossary of domain terms. Definitions only — no implementation details.

## Terms

### Synthesis by Analysis
The flagship DAWIS pipeline: iterative greedy loop that analyses an image in wavelet space, detects significant structures, restores them as objects, and subtracts them from a running residual until convergence. The main (99%) entry point, but not the only one — DAWIS is a toolbox of wavelet-based analysis methods of which this is one composition.

### Wavelet plane
One scale of the à trous (starlet) decomposition of an image. A set of wavelet planes spans scales from finest to coarsest plus a smooth (coarse) plane.

### Cube
Container stacking per-scale 2D planes into a 3D array (x, y, scale). Four kinds, tracing distinct mathematical objects: wavelet coefficients (WaveletCube), the coarse/smooth planes (CoarseCube, deeply coupled to the WaveletCube it comes from), binary significance masks (SupportCube), and integer region labels (LabelCube). Not multi-wavelength data. Base class: `Cube`. Called "datacube" before 1.0, a name that survives only in pre-1.0 pickles.

### Interscale tree
A hierarchical grouping connecting significant regions across adjacent wavelet scales, rooted at a coarse-scale region. The unit handed to restoration; deblending decides where trees split. Class: `InterscaleTree`. Called `interscale_tree` before 1.0, a name that survives only in pre-1.0 pickles.

### Atom
The output unit of the pipeline: a structure reconstructed (synthesised) from an interscale tree, carrying its image-domain footprint and flux. Atoms are raw synthesis building blocks — not yet astrophysically meaningful sources. Class: `Atom`. Called "restored object" before 1.0, a name that survives only in pre-1.0 pickles.

### Atom selection
Post-processing step that filters and groups atoms — by scale, position, morphology, automatically or interactively — into source models. The step that turns DAWIS output into astrophysical measurements (e.g. separating intracluster light from galaxies).

### Source model
An astrophysical structure modelled as a combination of selected atoms — the product of atom selection, and the level at which catalog-style measurements (positions, fluxes) become meaningful.

### Residual
The running difference between the input image and the sum of all atoms so far; the pipeline iterates on it until convergence.

### Worker
A whole process a run hands a patch of independent work to — interscale trees to build, or trees to synthesise into atoms. Counted by `n_workers`. Scheduling only: the worker count cannot change what a run computes, only how fast it computes it. Not a thread.

### Thread
What a maths library splits a *single* numpy or scipy array operation across, inside one process. Counted by `n_threads`, and owned by OpenBLAS, MKL or OpenMP rather than by DAWIS's own code. A thread is a unit of one process's work; a worker is a whole process. Not scheduling only, as a worker is: a sum split across threads is added in a different association order, so the thread count moves an atom's flux in its last digits — within tolerance, never in a decision like an atom count or a convergence flag.
