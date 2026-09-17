# Provenance audit — what a DAWIS run records, and what it does not

Ticket #62. Written 2026-08-26 against `refactor/1.0` at `b36968b`, on
`amael-XPS-13-9360`.

This is a document, not a code change. It exists because the maths-library
thread count was found by accident — while writing the Tier 2 parallel
consistency test for #57 — and there was no reason to believe it was the only
one. So this looks on purpose for the next one.

It lives here rather than in a `docs/` tree because the thing it audits is
`provenance.toml`, and this repository keeps a document beside what it
describes. `audit_environment.py` beside it is the method, made runnable.

## The standard

**An input counts as recorded only if a run's own outputs state it.**

A value that is knowable in principle is not recorded. Asking the machine
afterwards does not count: the machine may have been reinstalled, and the
answer it gives today is not the answer it gave when the golden was written.
Reading a lock file does not count either — a lock file describes an intended
environment, and the audit's question is what the run actually computed under.

Under that standard the C library version is recorded and scipy's is not. Part
2 shows that split is an accident rather than a decision.

## Method

Re-run this audit at the next release rather than re-inventing it.

1. **Enumerate what is recorded.** Read every field of both
   `tier{1,2}_*/provenance.toml` and every key of a
   `benchmarks/results/*/*/benchmark.json`. Part 1 is that enumeration;
   regenerate it, do not trust this copy.
2. **Enumerate what a run touches.** `grep` the package for third-party
   imports and for every call into one, then sort those call sites into
   *decision paths* — a result that becomes an atom count, a level, a bounding
   box, a convergence flag — and *value paths*, a result that only becomes a
   flux. A dependency on a decision path is the stronger candidate, because a
   decision has no tolerance to absorb it.
3. **Read the candidate facts off the machine.** `python
   tests/goldens/audit_environment.py --pin-threads 1` prints every one of
   them. Everything it prints is readable from inside a run, which is what
   makes each one recordable.
4. **Measure, do not assume.** For each candidate find a lever that changes it
   and nothing else, then run Tier 1 twice. Tier 1 is 40 s and restores 76
   atoms, which is enough structure for a decision to move. The levers used
   here are in the evidence column of Part 2.
5. **Compare at full precision.** Total atom flux to 17 significant digits,
   plus the atom count, the iteration count and the per-atom level sequence. A
   difference in the last digits and a difference in a decision are different
   findings and must not be collapsed into one.
6. **File what you find.** Anything that moves output and is not recorded gets
   its own ticket. Anything that does not gets a row in Part 2 saying so, so
   the next audit does not re-measure it.

Do not use Tier 2 for this. It is 1708 s per run and the audit needs a dozen
runs; every finding below is visible at Tier 1.

## Part 1 — What is recorded today

### Golden provenance, field by field

Counted after #62 moved the package versions into their own table. Both tiers
carry 81 fields. Tier 1 has 8 tables; Tier 2 has 9, because it adds `[budget]`'s
5 while its `[input]` carries 5 fields where Tier 1's carries 10.

| Table | Fields | What they capture | Can they move output? |
|---|---|---|---|
| `[golden]` | `tier`, `recorded_by`, `recorded_on`, `reason`, `ticket`, `wall_s` | which tier, who recorded it, when, why, the ticket, and how long it took | **No.** Bookkeeping. `wall_s` measures the run; it is not an input to it. |
| `[source]` | `repo`, `branch`, `commit`, `describe`, `dirty` | which checkout the measured code came from | **Yes.** The commit *is* the algorithm. `dirty` is the guard that stops a commit from describing code that was not what ran. |
| `[source]` | `stubbed_modules` | imports replaced by do-nothing stand-ins | **No, as recorded.** `ray` and `ultranest` are never reached on the serial path, so stubbing them moves nothing — the fact is recorded rather than argued. |
| `[input]` T1 | `image`, `image_shape`, `image_dtype` | the packaged file the cutout came out of | **Yes.** |
| `[input]` T1 | `cutout_centre_xy`, `cutout_size`, `cutout_half_width`, `cutout_slice_y`, `cutout_slice_x`, `cutout_expression`, `cutout_dtype` | the cutout convention, stated seven ways | **Yes**, and the redundancy is the point: every misreading of `(222, 242)` still lands on a valid source-rich patch, so a silent difference is turned into a loud one. |
| `[input]` T2 | `image`, `image_shape`, `image_dtype`, `image_sha256`, `cutout` | the whole packaged image, by digest | **Yes.** The digest is checked first, so a changed input fails loudly instead of shifting the golden. |
| `[run]` | `seed`, `n_workers`, `n_threads`, `cg_max_iter`, `pipeline_max_iter` | the five knobs the tier fixes by hand | **Yes.** `n_workers` is scheduling only (#57); `n_threads` moves the last digits (#61); the other three move everything. |
| `[run]` | `cg_max_iter_default_in_source` | what the recorded checkout would have used unforced | **No.** It says whether the forced value differed from the shipped one. |
| `[run]` | `n_iterations`, `n_atoms` | 20 and 76 at Tier 1, 27 and 463 at Tier 2 | **No — these are outputs**, recorded beside the inputs because they are the two numbers compared exactly. |
| `[config]` | `starting_level`, `tau`, `n_levels`, `n_sigmas`, `deblend_contrast`, `gamma`, `min_span`, `max_span`, `lvl_sep_big`, `rm_gamma_for_big`, `lvl_deblend`, `extent_sep`, `ecc_sep`, `lvl_sep_lin`, `lvl_sep_op`, `ceps`, `scale_lvl_eps`, `conditions`, `max_iter`, `size_patch`, `inpaint_res`, `iptd_sigma`, `seed`, `n_workers` | the pipeline parameter set, defaults resolved rather than passed | **Yes**, every one. Resolving rather than echoing is what stops a changed default walking in unrecorded. |
| `[config]` | `outfile_format`, `data_dump`, `gif`, `resume` | what the run writes and whether it reads any of it back | **`resume` yes, the other three no.** See F11: `resume` makes a stale output directory an input. It is `false` in both goldens. |
| `[tolerances]` | `residual_sigma`, `image_atol`, `image_rtol`, `atom_rtol`, `atom_atol`, `convergence_rtol`, `convergence_atol`, `exact` | #3's comparison rule, and the sigma this run's own residual produced | **No.** These decide how the golden is *compared*, not what it computed. They belong in provenance regardless: a golden recorded under one rule and compared under another is the failure this prevents. |
| `[machine]` | `hostname` | which machine | **No.** Identification. |
| `[machine]` | `platform` | `platform.platform()` | **Yes, partly** — see F6. The OS, the kernel, the architecture *and*, as a substring on Linux, the glibc version. |
| `[machine]` | `python` | interpreter version | **Yes.** |
| `[dependencies]` | `numpy`, `scipy`, `astropy`, `photutils`, `skimage`, `h5py`, `threadpoolctl`, `ray`, `hdf5` | every package version a run's result can depend on | **Yes.** Added by F4 below, which is why this row describes the table as it is *now*: before #62, `[machine]` carried `numpy` alone. An empty string means the package was absent or stubbed, which `[source].stubbed_modules` distinguishes. |
| `[machine]` | `blas_vendor`, `effective_n_threads` | which maths library loaded, and what thread count it actually gave | **Yes.** Added by #61, and the file says so: these two are load-bearing where the four above are declared diagnostic. |
| `[budget]` T2 | `wall_s`, `benchmark_wall_s`, `benchmark_source`, `marker`, `runs_on` | what the tier costs and when it runs | **No.** Scheduling. |

The `[machine]` table is the one to look hard at. Four of its six fields are
declared "recorded for diagnosis only". That is correct for `hostname`. For
`platform`, `python` and `numpy` it is a claim about reproducibility — and it
is exactly the claim that was wrong about the thread count until #61.

### The benchmark record, key by key

`benchmark.json` has 25 top-level keys.

| Keys | What they capture | Can they move output? |
|---|---|---|
| `label`, `status`, `error`, `started` | what was measured, whether it finished, when | **No.** Bookkeeping. |
| `wall_s`, `loop_s`, `phases_s`, `restore_share`, `outdir_mb`, `peak_rss_total_mb`, `peak_rss_driver_mb`, `max_procs` | the measurement itself — wall time overall, per greedy loop, per pipeline phase, restoration's share, bytes written, peak RSS, process count | **No. These are outputs**, and they are the whole reason the file exists. |
| `n_iterations`, `n_atoms`, `cg_calls` | 27 iterations, 463 atoms, 394 solver calls | **No — outputs.** Recorded so a speed number stays attached to the work that produced it. |
| `image` | path and shape of the reference frame | **Yes.** |
| `seed`, `cg_max_iter` | the two knobs the benchmark sweeps | **Yes.** |
| `seed_supported`, `cg_max_iter_forced`, `cg_max_iter_default_in_source` | whether the measured tree accepted a seed, whether the solver cap was forced, and what it would have been otherwise | **No.** They say how the two knobs above were applied. |
| `config` | the parameter set, defaults resolved | **Yes**, same 29 knobs as `[config]` above. |
| `config_passed` | the subset passed by hand rather than defaulted | **No.** It is a subset of `config`, kept so a reader can see what the script chose. |
| `source` | package directory, git commit, branch, describe, dirty, stubbed modules | **Yes.** Same role as provenance's `[source]`. |
| `machine` | `hostname`, `cpu_model`, `n_cpus_machine`, `ram_total_gb`, `platform`, `python`, and `packages` — itself 9 versions: `numpy`, `scipy`, `astropy`, `h5py`, `photutils`, `skimage`, `matplotlib`, `psutil`, `ray` | **Mixed.** `hostname` and `ram_total_gb` identify; `n_cpus_machine` cannot move output (F13); `cpu_model` moves it indirectly (F1); `platform`, `python` and the numerical entries of `packages` move it directly (F4, F6). |

The benchmark record is not a golden and no comparison rests on it. It is here
because it is the only place in the repository where the dependency versions
are written down at all — and because the fields it has and provenance lacks
are the shortest statement of what provenance should gain.

## Part 2 — Candidate unrecorded inputs

Verdicts are **measured** where a lever was pulled and the output compared, and
**unverified** where they are not. An unverified verdict is an argument about a
code path, not a measurement, and is labelled so a future reader can tell them
apart.

Baseline for every measurement: Tier 1, seed 20260823, `n_threads=1`,
`n_workers=1`, `cg_max_iter=3` — 76 atoms, 20 iterations, total atom flux
`253.93024126701309`.

| # | Candidate | Verdict | Evidence |
|---|---|---|---|
| F1 | **CPU instruction set and dispatch path** | **Does move output. Measured. Unrecorded.** | Re-ran Tier 1 under `MKL_ENABLE_INSTRUCTIONS=SSE4_2`, nothing else changed. Total atom flux `253.93024126701312` against the baseline's `...09`. Same 76 atoms, same 20 iterations, same level sequence. That is the thread count's signature exactly: last digits move, no decision moves. |
| F2 | **BLAS version and threading layer** | **Does move output. Unverified. Unrecorded.** | The same lever as F1 one level up: F1 shows this library's kernel choice moves flux, so a different build of the library is the same class of change. `threadpool_info()` already returns `version: "2025.0-Product"` and `threading_layer: "intel"` in the very call `[machine].blas_vendor` is read from — both are discarded. Not measured directly: it needs two MKL builds side by side. |
| F3 | **numpy's own SIMD dispatch** | **Does *not* move output. Measured.** | Re-ran Tier 1 under `NPY_DISABLE_CPU_FEATURES="AVX2 FMA3 AVX F16C SSE41 SSE42 SSSE3 POPCNT"`, dropping numpy to its SSE3 baseline. Bit-for-bit identical, total flux `253.93024126701309`. A micro-probe explains why: across those levels `sum`, `std`, `cumsum` and `sqrt` are unchanged and only `exp` moves — and `dawis/` calls `sqrt`, `log2` and `power`, never `exp`. numpy's reduction blocking does not depend on SIMD width, so its association order is fixed. F1 is a separate mechanism inside MKL and this lever does not reach it. |
| F4 | **scipy, astropy, photutils, scikit-image versions** | **Does move output. Unverified. Unrecorded in provenance** (recorded in the benchmark record). | These sit on *decision* paths. `astropy.stats.sigma_clip` sets the noise sigma at `detection/deblending.py:98`, which sets every scale's detection threshold, and again at `wavelets/anscombe.py:77,98,181` and `restoration/inpainting.py:58`; `photutils.segmentation.detect_sources` and `deblend_sources` (`detection/deblending.py:112,116`) produce the segmentation itself; `skimage.measure.label` (`detection/labelling.py:66`) labels it, `regionprops` (`detection/regions.py:124`) supplies the `extent` and `eccentricity` that pick an atom's restoration operator, and `skimage.feature.peak_local_max` (`restoration/inpainting.py:60`) picks the mode of the noise histogram; `scipy.stats.sigmaclip` (`detection/thresholding.py:107`) does the sigma job on the simple detector path. A version that changes any of those changes an atom count. Magnitude unverified — F5 bounds how far it has to move. `pyproject.toml` pins none of them by version. |
| F5 | *(margin for F4)* | **Measured.** | Nudged the deblending sigma alone and re-ran Tier 1. **1 and 4 float32 ULPs: bit-for-bit identical** — a rounding-level difference does not survive. **1e-5 relative: 76 atoms, 20 iterations, flux `253.93199307366328`** — values move, decisions hold. **1e-4 relative: 74 atoms, 19 iterations** — a decision moves. 1e-3: 71 atoms. 1e-2: 60 atoms, 17 iterations. The threshold sits between 1e-5 and 1e-4 relative, about a thousand times above float32's last digit. So a dependency version moves DAWIS's *decisions* when it changes an **algorithm** — a default `maxiters`, a centre statistic — and not when it merely rounds differently. |
| F6 | **C library version** | **Does move output. Already recorded, by accident.** | `platform.platform()` returns `Linux-7.0.0-29-generic-x86_64-with-glibc2.43`. The glibc version is in there as a substring, in both provenance and the benchmark record. Only on Linux: the same call on macOS yields no libc version at all. Recorded well enough to need no ticket, and fragile enough to be worth this row. |
| F7 | **Compiler flags of the numerical wheels** | **Does *not* move output on this machine. Measured. Unrecorded.** | Readable at run time, contrary to the assumption that build-time facts are lost: `np.show_config('dicts')` returns the full C compiler args. On the audited machine, `gcc 14.3.0` with `-march=nocona -mtune=haswell -ftree-vectorize -O2` and **no `-ffast-math`** — which is the flag that would have made this a finding. `-march=nocona` also means the wheel's own kernels are conservative, which is consistent with F3 measuring nothing. Only numpy exposes its build this way; for scipy and photutils, F4's versions stay the practical proxy. |
| F8 | **Platform floating-point defaults** | **Does *not* move output on this machine. Measured. Unrecorded.** | Probed inside a `parallel_backend('serial', n_threads=1)` block that runs a matmul first, so a library that sets flush-to-zero on initialisation has done it before the probe reads: subnormals survive a multiply, `1 + 2**-53 == 1`, and `np.geterr()` is unchanged. Neither numpy nor MKL sets flush-to-zero here. Unrecorded — and on a machine where some library does set it, a run would compute different denormals with nothing in its outputs to say so. |
| F9 | **The parallel `backend`** | **Unverified, and deliberately not marked either way.** Outside the spec's minimum list. Unrecorded in the goldens. | `Config.backend` exists on the 1.0 tree and `recording.py` would record it, but both goldens were recorded from `legacy-seeded`, which predates the field — so neither `provenance.toml` names a backend. #57 establishes that the worker *count* cannot move an atom; that the backend handing out the work cannot either is a separate claim and is untested across backends. `ray` is not installed on the audited machine, so this could not be measured here. Re-check once a golden is recorded from the 1.0 tree. |
| F10 | **HDF5 library version** | **Does not move output. Established by inspection.** | `outfile_format` is `pickle` in both goldens, so `h5py` is never reached on the recorded path. `h5py.version.hdf5_version` is `1.14.5` here and recorded nowhere. A live candidate only if a golden is ever recorded at `outfile_format = "hdf5"`. |
| F11 | **Prior contents of the output directory** | **Cannot move output as configured.** | `synthesis_by_analysis` resumes by file: with `resume` true, an iteration whose atoms are already on disk is read back instead of recomputed, which makes a stale directory an unrecorded input of the first order. Both goldens record `resume = false`, and the recording scripts run in a fresh temporary directory. Recorded, and safe. |
| F12 | **`PYTHONHASHSEED` and iteration order** | **Does not move output. Established by inspection.** | Set and dict iteration order would be an input if any ordered result came out of one. The only `set()` in `dawis/` outside `viz` is a literal of allowed key names used for validation (`detection/regions.py:70`), and `io/store.py:361,365` sorts both glob results before zipping them. No result's order depends on hashing. |
| F13 | **`os.cpu_count()`** | **Does not move output. Established by inspection.** | No default in `dawis/` derives from it; `n_workers` and `n_threads` are both explicit and both recorded. The benchmark record's `n_cpus_machine` is identification. |
| F14 | **Maths-library thread environment variables** | **Cannot move output.** | `threadpoolctl` overrides `OMP_NUM_THREADS` and its siblings outright, and `tests/conftest.py` asserts the *effective* count rather than the variable. This is the one candidate that used to be a real hidden input and has been closed: #61 made the thread count a knob DAWIS owns, and `[machine].effective_n_threads` records what the library actually gave. |

## Part 3 — Findings, and what was decided about them

Three candidates move output and were not recorded. One was acted on; two were
weighed and left. The decision was Amaël's, on 2026-08-26, and it is recorded
here with its reasoning because a future reader will otherwise re-derive the
finding and assume it was missed.

### Acted on — the dependency versions are recorded now (F4)

`[machine]` recorded `numpy` and no other dependency, while the detection
threshold comes from astropy, the segmentation from photutils, the region
morphology and the inpainting histogram mode from scikit-image, and the simple
detector's clip from scipy. Every one is on a decision path. The benchmark
record, which measures only speed, already recorded all of them.

This is the finding that matters, and the reason is F5: these four can move an
**atom count**, where F1 and F2 can only move an atom's last digits. A decision
has no tolerance to absorb it.

`recording.py` now writes a `[dependencies]` table — `numpy`, `scipy`,
`astropy`, `photutils`, `skimage`, `h5py`, `threadpoolctl`, `ray` and HDF5's
own library version — observed in the recording process itself. `numpy` moved
out of `[machine]` and into it: what box this is and what was installed on it
are two facts that drift independently.

**Both tiers were re-recorded and both carry the table.**

Neither golden moved. At Tier 1, `atoms.csv`, `convergence.csv`,
`input_cutout.fits`, `residuals.fits` and `restored.fits` are byte-identical
to their previous recording; at Tier 2, the same four files it keeps are too.
`provenance.toml` is the only file that changed in either tier. Same 76 atoms
in 20 iterations at Tier 1, same 463 in 27 at Tier 2, and every tolerance —
including the residual sigma each is derived from — came out unchanged.

That is the outcome this change had to have. Adding a provenance field must
not move a golden, and the byte comparison is what proves it did not rather
than assuming it.

### Weighed and left — the dispatch path and the BLAS build (F1, F2)

Both are real and F1 is measured. Neither is being recorded, and this is a
judgement about significance rather than about evidence.

What F1 actually moves is an atom's flux in its last digits —
`253.93024126701312` against `253.93024126701309`, a relative difference of
about 1e-16. `[tolerances].atom_rtol` is 1e-4, twelve orders of magnitude
wider. The atom count, the iteration count, every level, bounding box, filter
and convergence flag are unchanged. F2 is the same mechanism one level up and
is not even measured.

So they are exactly the class of difference #3 chose float tolerance over
bit-for-bit comparison in order to survive. Recording them would add two
fields that can never explain a golden failure, because no golden failure can
be caused by them.

The distinction worth keeping from #61: the thread count was recorded not
because its effect was large but because *nothing said it existed*, and a
run compared at one thread against a golden recorded at four differed for a
reason no file gave. That argument does not carry to F1 and F2 now, because
`provenance-audit.md` is the file that says they exist and bounds their size.

Re-open this if a golden ever fails a comparison and the dependency versions,
the thread count and the seed all match. That is the situation in which these
two become the remaining explanation, and this section is where to start.

### Watch list — measured, and no difference found

F7 and F8 are unrecorded and both measured **no** difference: no `-ffast-math`
in the numpy build, and no flush-to-zero under the thread pin. Neither is a
demonstrated defect. Written down because "we checked once, on one machine, and
it was fine" is the kind of claim that ages badly.

### Closed, so the next audit skips them

F3, F6, F10, F11, F12, F13 and F14 are negatives on purpose. The point of
writing a negative down is that the next audit does not spend a day
re-measuring it. Two carry an expiry: **F6** should be re-checked on a
non-Linux machine, where `platform.platform()` reports no libc version, and
**F9** once a golden is recorded from the 1.0 tree, which is the first
recording whose `[config]` will name a backend.

## Reproducing the measurements

The Tier 1 harness drives all of them. A run is
`tier1.run_tier1(dawis, tier1.cutout_of(image), work_dir)` and
`goldenrun.atom_catalogue` reads the atoms back; compare the atom count, the
iteration count, the level sequence and the total flux at 17 digits.

- **F1**: prefix the run with `MKL_ENABLE_INSTRUCTIONS=SSE4_2`.
- **F3**: prefix it with `NPY_DISABLE_CPU_FEATURES="AVX2 FMA3 AVX F16C SSE41 SSE42 SSSE3 POPCNT"`.
- **F5**: replace `dawis.detection.deblending`'s module-level `np` with a proxy
  whose `std` returns the real value nudged by a set number of float32 ULPs, or
  by a relative factor. Nudge the sigma and nothing else — patching
  `sigma_clip`'s *return array* instead changes its dtype and confounds the
  measurement.
- **F2's readable half, F7, F8**: `python tests/goldens/audit_environment.py
  --pin-threads 1`.
