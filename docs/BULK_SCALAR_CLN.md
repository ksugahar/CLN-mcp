# Bulk scalar CLN, memory and CUDA

The full comparison is saved in [notebook 23](23_bulk_scalar_cln.ipynb).
The requested GPU implementation is **MATLAB mexcuda**. CuPy measurements
are retained as preliminary reference evidence, not the selected GPU route.
It keeps the internal constant-material wire, zero shift, three CLN stages,
geometry order 3 and p=2,3,4 scope of [notebook 22](22_batched_scalar_cln.ipynb).
Original ICCG A-T and condensed A-T remain independent references.

## Native bulk operations

`BulkSweep` uses NGSolve's discontinuous wrapper of the same oriented HCurl
space. One native LinearForm assembles every element load. Packed local dual
operators map those loads into the same curl coordinates as the archived
per-element `CalcElementVector` calls. NGSolve continues to own orientation,
Piola maps, curved transformations and quadrature. Twelve independent controls
compare both assembly paths for affine/curved meshes, both boundary roles and
p=2,3,4.

Paired squared norms use vector-valued `Integrate`. The native Integral rule
uses `5 + bonus_intorder`, so bonus 8 is retained as **order 13**, rather than
accidentally using order 8. The controls compare the native paired and original
scalar integrals, including a non-polynomial exponential. See the official
[vector Integrate binding](https://raw.githubusercontent.com/NGSolve/ngsolve/master/comp/python_comp.cpp)
and [Integral quadrature implementation](https://raw.githubusercontent.com/NGSolve/ngsolve/master/comp/postproc.cpp).

The three weak residual loads are assembled in one compound HCurl space.
Cumulative physical fields are added directly in their original discontinuous
HDiv coefficient vectors; no interpolation or changed space is introduced.
This avoids progressively nested coefficient-function expressions. Unpacked
local operator copies are released after packing.

All original scalar and physical weak residuals remain below 1e-6, divergence
and recovered physical curl errors below 1e-8, and local curl fit below 1e-10.
Full coefficient and volume vector-field differences must stay below 1e-6.
The scalar route rejects calls to either reference vector solve.

## What memory measurements mean

Element-by-element recovery can avoid a global vector matrix and its factor.
It still needs a compatible global coefficient vector, local operators, loads,
and the scalar trace matrix/factor. Caching every dense element right inverse
costs memory, particularly at higher order. The original A-T reference is
iterative; its memory must be measured rather than replaced by an assumed
large vector Cholesky factor.

Each route/cohort runs in a fresh subprocess with identical imports and mesh.
Windows lifetime peak working set and private commit are recorded alongside
import, pre-solve, post-solve and post-GC snapshots. These peaks include setup,
checks, allocator retention and field samples. They are not obtained by
subtracting unrelated process peaks. Explicit unique NumPy operator storage
is reported separately; it excludes native FE matrices/factors.
One observation per fresh process establishes this bounded comparison, not
a statistical memory scaling law. GPU allocator reserved bytes are reported
separately and are not described as peak VRAM.
The CUDA experiment retains CPU local operators for load mapping/checks and
adds device copies; GPU offload does not automatically reduce host memory.

## Actual GPU experiment

### Selected mexcuda route

The standalone CUDA MEX keeps local operators and work buffers on the GPU
across stages. One MEX call executes the complete colored iteration, with a
single device reduction and scalar transfer per convergence check. MATLAB
Engine transfers are explicit. Scalar FE solves, load assembly and integrated
checks remain on CPU; there is no implicit CPU solver fallback.

The MEX registry uses checked uint64 generation handles rather than pointers.
It checks array types, finite values, integer products before indexing,
operator ranges and both write/write and read/write conflicts inside a batch.
Destroy, stale handles and unload/recreate behavior have native controls.
The Engine's scalar uint64 return is explicitly rewrapped as `matlab.uint64`
before it is passed back into the checked MEX boundary.

Build and accuracy validation use the upgraded 100 host in an isolated
environment. The complete four-cohort CLN validation retains every coefficient,
field and original residual gate. This is a non-timing validation: public
results contain no solve or phase times. The saved source and binary hashes
link it to the native MEX build and ten rejection/lifecycle controls.
See [MathWorks mexcuda](https://www.mathworks.com/help/parallel-computing/mexcuda.html).

Formal performance comparisons require an otherwise idle **mdx** host under
the user's updated host policy. The available mdx host is busy and exposes a
virtual display adapter without an available NVIDIA CUDA device. Therefore
**mexcuda full-CLN speed is unverified**, not inferred from the CuPy experiment.
Timing on 100 is not used as performance acceptance. GPU deployment to A100
also remains unverified while that host is unavailable.

Memory validation includes the separately owned MATLAB process. Each
route/cohort uses a fresh Python process and, for mexcuda, a fresh MATLAB
session. Native per-process peaks and simultaneous Python+MATLAB RSS sampled
every 20 ms are saved separately. The sampled joint peak can miss brief
spikes; it is not an OS-provided exact process-tree peak. MATLAB session and
GPU-context overhead are part of the complete-method memory cost.

### Preliminary CuPy reference

The measured NGSolve build reports the host reference backend, so native
NGSolve GPU FE execution is unavailable in this environment. See
[NGSolve GPU backend documentation](https://docu.ngsolve.org/latest/i-tutorials/unit-5.6-gpu/commonGPU.html).
An optional CuPy engine instead runs explicit CUDA FP64 element projection
and algebraic fit kernels. Each block owns one element and each batch has
disjoint free supports. Colors execute sequentially. Scalar FE solves,
assembly and integrated physical checks remain on CPU.

This is a **CUDA-local / CPU-FE hybrid**, not an entirely GPU CLN. Missing
CUDA raises; there is no host fallback. Setup, initial compilation/context,
operator upload, per-stage transfers, synchronization and all checks are
included in complete timings. First observations are kept separately from
three warmed runs. Device, CUDA/CuPy versions, raw observations, transfer
times and device operator storage are saved. A separate correctness smoke run
primes the on-disk CUDA kernel cache before the full study; the first full
observation includes a fresh process/context but is not a cold compiler-cache
measurement. Results on the measured GPU
do not establish performance on newer hardware or larger meshes.

## Reproduce

Build and non-timing validation on the upgraded 100 host:

```powershell
python validation/build_mexcuda.py --directory mex-build
Copy-Item mex-build/build.json docs/data/mexcuda_build.json
python validation/mexcuda_controls.py --directory mex-build --output docs/data/mexcuda_controls.json
python validation/mexcuda_cln_verify.py --directory mex-build --output docs/data/mexcuda_cln_verify.json
python validation/mexcuda_memory_study.py --directory mex-build --output docs/data/mexcuda_memory_study.json --workers mex-memory-workers
python validation/validate_mexcuda_evidence.py
```

The built binary stays in the external build directory. On an otherwise idle
mdx host with the required MATLAB/CUDA runtime, use the same hash-verified MEX
with `validation/mexcuda_cln_study.py` for formal first/warmed timing. This
formal study has not run; its mere presence is not performance evidence.

The historical CPU/CuPy reference producers are retained for reproduction:

```powershell
python validation/bulk_cln_study.py --output docs/data/bulk_cln_study.json
python validation/bulk_controls.py --output docs/data/bulk_controls.json
# Optional CUDA environment with CuPy; no changes to the native FE runtime.
python validation/gpu_cln_study.py --output docs/data/gpu_cln_study.json
python validation/bulk_memory_study.py --gpu --output docs/data/bulk_memory_study.json --directory memory-workers
python validation/validate_bulk_cln.py
```

Run timing and memory jobs sequentially on an idle host, with caller-owned
TaskManager, four NGSolve threads and one BLAS thread. The saved evidence
includes normalized source hashes. Existing producers and their evidence
remain unchanged. These additions are Codex self-review, not independent
scientific-review acceptance or a general preference for V/Phi over A-T.
The CPU and CUDA studies are separate alternating-order blocks. Compare
GPU/CPU speed ratios within the CUDA block; observed timing ranges, especially
the CPU block's small p=2 cohort, preclude treating cross-block differences as
an algorithmic speedup. No cause is assigned to unexplained variability.
These LAB observations predate the clarified idle-mdx-only timing policy and
are historical exploratory results, not formal performance acceptance.

The repository publishes only the specifically requested CUDA MEX source in
`validation/cln3d/mexcuda/colored_projection.cu`. Compiled MEX files and build
logs stay outside public source. The format lint exception is restricted to
that path; it does not permit arbitrary native sources or binaries.
