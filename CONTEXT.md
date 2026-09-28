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
A hierarchical grouping connecting significant regions across adjacent wavelet scales, rooted at a coarse-scale region. The unit handed to restoration; deblending decides where trees split. Class: `InterscaleTree`. Called "interscale_tree" before 1.0, a name that survives only in pre-1.0 pickles.

### Atom
The output unit of the pipeline: a structure reconstructed (synthesised) from an interscale tree, carrying its image-domain footprint and flux. Atoms are raw synthesis building blocks — not yet astrophysically meaningful sources. Class: `Atom`. Called "restored object" before 1.0, a name that survives only in pre-1.0 pickles.

### Atom selection
Post-processing step that filters and groups atoms — by scale, position, morphology, automatically or interactively — into source models. The step that turns DAWIS output into astrophysical measurements (e.g. separating intracluster light from galaxies).

### Source model
An astrophysical structure modelled as a set of whole atoms — the product of atom selection, and the level at which catalog-style measurements (positions, fluxes) become meaningful. An atom belongs to at most one source model, and carries no weight: an atom is in or out.

### Bootstrap error
The uncertainty on a quantity measured on a group of atoms: the standard deviation of that quantity over N resamplings of the group's atoms, each of size N and drawn **with replacement**. The default error on any source-model measurement.

### Borderline resampling
The alternative uncertainty on a group of atoms, and the one the CFIS papers actually used: draw samples of *variable* size, uniform in `[(1-spread)N, (1+spread)N]`, **without** replacement, swapping in a random count of **borderline** atoms — those one level below a level cut and inside the mask of interest — and report the median with the 5th and 95th percentiles. Not a bootstrap, and named separately so it cannot be mistaken for one.

### Atom store
Where a run keeps its atoms on disk: a table of atom properties plus the atoms' pixels. The pre-1.0 store is one pickle or HDF5 file per iteration, one group per atom.

### Build time
Wall-clock time to synthesise the image of a given selection of atoms from an atom store — the full field being the selection of every atom. The figure a new atom store must lower against the pre-1.0 store, on the same selection of the same run.

### Selection criterion
A test that keeps or drops an atom on one of its properties — wavelet level, bounding-box size, distance from a point, position inside a mask, kurtosis, eccentricity. It is a predicate over the **whole** atom index at once, returning one boolean per atom, so a selection reads no pixels at all: every property it tests lives in the store's index. Thresholds are given in pixels and levels, or in kpc at a redshift the user supplies. Criteria combine with and, or and not. Any callable of that shape is a criterion, so users add their own without changing DAWIS.

### Atom position
The point a position or mask criterion tests: the atom's flux-weighted centroid by default, the centre of its bounding box on request or where the centroid cannot be computed. It is computed by the pipeline and written into the atom store's index, not recomputed at selection time. A mask lookup rounds it to the nearest integer pixel.

### Cascade
An ordered list of (selection criterion, source-model name) rules. Each atom walks the list from the top and the first matching rule claims it, so the order is part of the selection and two orderings of the same criteria give different source models. Atoms no rule claims are unclassified. The other way to compose criteria is a direct split, which separates one group from the rest without any ordering.

### Mask sort
The selection step that assigns each atom to exactly one group by testing its atom position against user-supplied masks, with every atom no mask claims going to an `unclassified` group. The general form of the ICL / BCG / satellites split, and a **named convenience over a cascade of mask criteria** rather than a mechanism of its own.

### Unclassified atoms
The atoms a selection scheme claimed for no source model. They are how a selection is judged — what it missed is visible in them — and that judgement is by eye, so they synthesise to an image and carry no measurements and no catalog row.

### Residual
The running difference between the input image and the sum of all atoms so far; the pipeline iterates on it until convergence.

### Worker
A whole process a run hands a patch of independent work to — interscale trees to build, or trees to synthesise into atoms. Counted by `n_workers`. Scheduling only: the worker count cannot change what a run computes, only how fast it computes it. Not a thread.

### Thread
What a maths library splits a *single* numpy or scipy array operation across, inside one process. Counted by `n_threads`, and owned by OpenBLAS, MKL or OpenMP rather than by DAWIS's own code. A thread is a unit of one process's work; a worker is a whole process. Not scheduling only, as a worker is: a sum split across threads is added in a different association order, so the thread count moves an atom's flux in its last digits — within tolerance, never in a decision like an atom count or a convergence flag.
