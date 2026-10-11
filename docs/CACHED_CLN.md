# Reusing CPU load and validation operators

This follow-up to [formal mexcuda timing](MEXCUDA_TIMING.md) targets repeated
CPU assembly and checking through a complete three-stage CLN recursion.
See [executed notebook 25](25_cached_cln.ipynb), the complete native
[timing JSON](data/cached_cln_timing.json),
[host monitor](data/cached_cln_host.json) and
[differential controls](data/cached_cln_controls.json).

## Recorded result

All 96 complete calculations pass, including 224 optimized scalar/local steps.
The accepted host monitor records zero foreign compute jobs. Warmed medians
(seconds), including operator setup and every physical check, are:

| Cohort | Original A-T | Condensed A-T | Bulk CPU | Cached CPU | Bulk mexcuda | Cached mexcuda |
|---|---:|---:|---:|---:|---:|---:|
| p=2, 3 mm | 1.264 | 1.252 | 2.727 | 2.402 | 2.592 | 2.210 |
| p=3, 3 mm | 3.178 | 2.240 | 6.795 | 5.872 | 6.241 | 5.132 |
| p=3, 2 mm | 8.960 | 6.625 | 22.265 | 18.499 | 17.752 | 14.223 |
| p=4, 3 mm | 8.708 | 4.914 | 17.411 | 15.760 | 15.363 | 13.756 |

CPU reuse improves the bulk CPU route by 1.10–1.20x; mexcuda with CPU reuse
improves the previous mexcuda route by 1.12–1.25x. The optimized mexcuda route
still takes 1.58–1.75x the original A-T time. These are medians of three warmed
observations, not universal speed bounds; first values and ranges are saved.

Maximum optimized/original-A-T coefficient and volume-vector differences are
1.21172e-10 and 5.34967e-09; maximum recovered original physical residual is
4.19972e-10. Original scalar residuals stay below 1.511e-14. Added logical CSR
storage is 27.69, 136.82, 404.33 and 476.69 MiB for the four cohorts. No
whole-method memory reduction is claimed. Initial MATLAB/GPU startup remains
an additional cost, separately measured in notebook 24.

For mexcuda, warmed physical-check sums drop from 0.443–3.773 s to
0.0026–0.0209 s per recursion, and curl-target assembly from 0.174–1.439 s
to 0.0078–0.0521 s. Scalar solve/check sums also decrease. Setup instead
increases from 0.932–6.868 s to 1.357–11.165 s; integrated curl-field checks
are slightly slower, at 0.375–1.095 s versus 0.280–0.993 s. The full-run
improvement includes these increases. Local operator construction and added
assembly now dominate; accelerating individual check phases is not an equal
speedup of the complete method. Maximum native differential vector error is
6.66e-16 across the twelve controls.

## Implementation

`validation/cln3d/CLN_VPhi_cached.py` retains the compatible hybrid-mixed
V/Phi scalar trace solve and colored element curl recovery. Its CPU recovery
and caller-owned mexcuda recovery use the same cached FE work. No scalar,
curl, boundary, coefficient or field acceptance limit is relaxed.

`validation/cln3d/cached_checks.py` assembles fixed-mesh operators once per
complete recursion: potential-to-scalar-load, flux-to-broken-curl-load,
uncondensed scalar-equation checks, and physical forcing/curl-check operators.
Each subsequent stage uses native NGSolve matrix-vector operations. Reused
scalar solution and norm buffers do not own saved modes: each accumulated
flux and recovered potential retains independent storage.

This speed optimization **adds assembled sparse load/check matrices**,
including a rectangular vector curl-curl validation operator. They are never
factored or solved as a global A-T equation. `global_matrix_assembled=False`
describes the unchanged element recovery; each new physical check explicitly
records `validation_global_vector_operator=True`. This route therefore does
not preserve the earlier claim of avoiding every global vector matrix.

The additional operators exchange memory for fewer repeated quadrature
evaluations. Their logical CSR array bytes are reported for both roles;
these exclude native graph/allocator overhead, other FE storage and MATLAB.
They are not whole-process peak memory measurements or a memory advantage.
All added assembly is inside the reported full-run setup and total time.

## Quadrature and acceptance

NGSolve's [symbolic integrators](https://github.com/NGSolve/ngsolve/blob/v6.2.2607/fem/symbolicintegrator.cpp)
select different default integration orders for linear and bilinear forms.
The cached rectangular load/check operators explicitly match the original
LinearForm volume rule, `2 * test FE order + 8`, using NGSolve integration
rules. Uniform tetrahedral spaces are required and checked. Boundary loads
retain their original rule. Integrated divergence and curl-field norms
retain order 13, including direct integration of the vector difference;
no cancellation-prone subtraction of separate energies replaces that check.

Twelve native controls cover flat/curved meshes, orders 2–4 and both boundary
roles. They compare cached scalar loads, broken curl loads, complete physical
check vectors and uncondensed scalar operator application against independent
uncompiled LinearForms and matrix-free application. Repeated updates include
sign/scale changes and clearing previous/forcing fields. The unchanged
original and condensed A-T routes independently verify all seven coefficients
and volume/sample vector fields in every complete benchmark run.

The scalar, physical and cross-route relative gates remain 1e-6; divergence
and integrated curl error remain 1e-8; local fit remains 1e-10. The guarded
scalar route still rejects global A-T/vector-trace solver fallback.

## Timing contract and reproduction

The mdx GPU machine is reached through `NGSolve`, distinct from the old
`A100` destination and the busy CPU-only `mdx`. An idle CPU/GPU preflight and
process monitor accompany the sequential experiment. Four cohorts each run
six routes, once first-observed and three times warmed, with alternating
route order. Setup, all load/check assembly and every stage transfer are
included. No operator cache survives a repeat. Imports, mesh creation,
cross-route comparisons, reporting samples and route-object teardown are
outside the solver timer, consistently with notebook 24. MATLAB/Engine
initialization remains separately reported; its earlier median is not a new
measurement in this study. Warm timings do not establish cold-boot latency.

NGSolve uses four configured threads and dense BLAS one; runtime pool metadata
is retained rather than claiming process-wide CPU affinity. The released
Radia 5.2.3 wheel and unchanged, hash-identified MEX binary are used. No new
package release or unpublished Radia API is implied.

With the native runtime and previously built MEX directory available:

```text
python validation/cached_host_timing.py --directory <built-mex-directory> --output <host-result.json>
python validation/cached_controls.py --output <control-result.json>
python validation/validate_cached_cln.py
```

The host wrapper runs `cached_cln_study.py` and saves `formal.json` and its
operational log in the supplied directory. Portable CI validates the saved
source identities, full physics, native controls, storage accounting, host
isolation and six corrupted-evidence controls. It does not rerun native FE.

Scope remains a constant-material, simply connected, zero-shift internal wire.
The C2 independent T–Omega comparison, loss FEM convergence and F2/Radia API
migration remain open. These timings do not close those scientific/API gates.
