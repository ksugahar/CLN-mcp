# Formal mexcuda timing on NGSolve

This follow-up closes the pending GPU timing question in notebook 23 using the
mdx GPU machine reached through the `NGSolve` SSH alias. Its Windows host name
is NGSOLVE; the GPU is an NVIDIA A100-SXM4-40GB. This is distinct from the old
`A100` SSH destination and from the CPU-only `mdx` host, which was busy.

See [executed notebook 24](24_mexcuda_timing.ipynb) for full tables, raw ranges,
startup and transfer accounting. The physics and recovery implementation are
unchanged from [the bulk scalar CLN study](BULK_SCALAR_CLN.md).

## Recorded result

The current mexcuda scalar route does **not** beat existing A-T for these
four cohorts. Warmed complete-recursion medians (seconds) are:

| Cohort | Original A-T | Condensed A-T | CPU bulk V/Phi | mexcuda V/Phi |
|---|---:|---:|---:|---:|
| p=2, 3 mm | 1.265 | 0.952 | 2.685 | 2.553 |
| p=3, 3 mm | 3.562 | 2.294 | 6.780 | 6.209 |
| p=3, 2 mm | 8.816 | 6.619 | 22.133 | 17.621 |
| p=4, 3 mm | 8.741 | 4.599 | 17.216 | 15.306 |

mexcuda improves CPU bulk V/Phi by 1.05–1.26x, but takes 1.74–2.02x the
original A-T time. Fresh Engine/configuration/GPU initialization has median
5.501 s (observed range 5.362–6.453 s), additional to recursion. Initializing
for one short solve therefore increases the disadvantage; shared sessions
amortize this cost without changing the warmed ordering.

The seven MEX calls together take warmed medians 0.276, 0.380, 1.055 and
1.069 s, respectively. Their bridge/conversion remainders are 0.031–0.043 s;
target-upload plus value-download warmed medians are below 0.002 s per recursion.
Operator setup/upload is separate and already included in full setup, whose
medians are 0.892, 2.354, 6.713 and 6.705 s. CPU setup, scalar FE work,
load/field/residual checks and coefficient integration dominate the total;
stage transfer is not the principal bottleneck in these cases.

All 64 full calculations pass, including 112 mexcuda local recovery steps.
Maximum mexcuda/original-A-T coefficient and volume-vector differences are
1.21173e-10 and 5.34785e-09; maximum recovered original-equation residual is
4.19971e-10. No numerical tolerance or physical check was relaxed. The accepted
monitor reports zero foreign compute jobs throughout the measured experiment.
All observations, ranges and nested phase times remain in the JSON and notebook.

## Measurement contract

- Four matched curved-mesh cohorts: p=2/3/4 at 3 mm and p=3 at 2 mm;
  geometry order 3, quadrature bonus 8, three stages and seven physical fields.
- Four routes: original ICCG A-T, condensed A-T, CPU bulk V/Phi and mexcuda
  V/Phi. Each has one first observation plus three warmed observations;
  execution order reverses on alternate repeats.
- Four NGSolve threads, one BLAS thread and one MATLAB computational thread.
  CPU FE assembly and scalar solves remain part of the GPU route.
  The runtime catalog also reports an OpenMP pool of 18; this protocol does
  not impose a four-thread process-wide OpenMP cap or CPU affinity. The same
  runtime settings apply to every route; all pool metadata is saved.
- Complete recursion includes FE/local setup, operator conversion/allocation
  and upload, all seven target uploads and coefficient downloads, MATLAB
  bridge calls and every physical residual/compatibility check. Mesh generation,
  reporting samples and independent cross-route comparisons are outside the
  solver timer. Setup and phase times are nested inside the total.
- Route-object destruction, MEX-handle cleanup after return and reporting
  samples are outside each solver timer, consistently with the retained A-T
  producer. The host wrapper's whole-experiment wall time includes them and
  cross-route validation; it is not a per-solve latency.
- A separate four-fresh-process study includes Engine launch, path/thread
  configuration, MATLAB version queries, GPU-context initialization and device
  queries. Shutdown is separate. Python/module imports are excluded common
  overhead. Driver/filesystem caches may be warm: no cold-boot claim is made.

Engine `quit()` acknowledges asynchronously. The initialization study records
that call separately and waits on the exact owned Windows process handle
until MATLAB actually exits before starting the next fresh session. Shutdown
wall time includes this wait; it is not misreported as just the acknowledgement.

The host gate requires no foreign Python/MATLAB/pytest/nvcc work, zero GPU
utilization and at most 5% CPU busy during a three-second preflight sample.
The first attempt was interrupted by a host restart before any benchmark
started. After restart, the wrapper waited for background startup activity
to end and accepted a fresh idle sample. During the experiment it samples
foreign compute processes every 30 seconds, excluding its own process tree.
Observed ownership is retained as PID plus creation time, so an operation-owned
MATLAB process is recognized after its Python parent exits without incorrectly
trusting a reused PID. Portable orphan/reuse controls check this lifecycle rule.
The first completed calculation was rejected because another GPU validation
job appeared in the monitor. Its timings remain private diagnostics, not
accepted performance evidence. The accepted study is a fresh rerun after
that job ended and the user reserved an exclusive measurement window.
An earlier monitor's shutdown false positive is also retained in private
diagnostics; accepted evidence comes from the corrected ownership monitor.
This cannot rule out short intervening jobs or hypervisor contention. The
measured ranges remain part of the evidence.

## Interpreting the accounting

`seconds` is measured complete recursion with all checks. The initial study's
Engine/GPU startup is saved separately in `matlab_engine`. The supplementary
initialization JSON measures the complete configuration sequence in four fresh
processes. Notebook 24 adds its median to warmed recursion only as a labelled
recombined estimate; it is not a measured cold solve for every cohort.

`mex_setup_transfer_seconds` measures each complete setup call, including
conversion, bridge, CUDA allocation and upload. Count it once per T/A object,
not seven times from the repeated step metadata. It is already in setup.
Per-step `mex_call_seconds` includes target conversion/upload, projection,
fit checks, result download and conversion/bridge overhead. The latter is a
wall-time remainder, not a direct measurement of pure IPC. Projection and fit
times use host wall clocks including launches/synchronization, not CUDA-event
kernel-only timings. These overlapping categories must not be added to total
recursion again.

## Runtime, reproducibility and release status

The calculation uses the released Radia 5.2.3 wheel, NGSolve 6.2.2607 and
MATLAB R2026a. CLN Python producers are the repository sources recorded by
normalized SHA-256; they are not obtained from a newly released package.
The MEX is the exact artifact previously built on 100, with binary hash
checked against `mexcuda_build.json`; it is not rebuilt on the compute host.
The round-wire study does not require the separate unpublished C2 topology API.
Repository integration and package publication remain separate operations.

An isolated, job-owned environment supplies FE dependencies without changing
the host's existing Python/MATLAB installation. The existing official MATLAB
Engine installation is reused. All Engine sessions are operation-owned and
closed. Native controls reject invalid inputs and stale handles before timing.

```powershell
python validation/mexcuda_controls.py --directory mex-build --output controls.json
python validation/mexcuda_host_timing.py --directory mex-build --output host-timing.json
python validation/mexcuda_initialization.py --directory mex-build --output initialization.json
python validation/validate_mexcuda_timing.py
```

The wrapper writes `formal.json` and `formal.log` inside the MEX directory.
Copy the accepted result, host gate, controls and initialization JSON into
the named `docs/data/mexcuda_*` paths consumed by the portable validator.
Public evidence contains source identities, all observations, coefficients,
sampled and volume-vector parity, unchanged true residuals and rejection gates.
Private durable laboratory storage retains runtime freeze, commands, exact
binary, raw operational logs and restart/preflight diagnostics. Disposable
job inputs/environments are removed only after hash-verified recovery.

The accuracy scope remains constant-material internal round wire, zero shift,
three stages and these meshes. Faster local projections do not prove a faster
complete CLN method, large-problem scaling or a GPU-memory advantage. This is
Codex self-review, not independent scientific acceptance.
