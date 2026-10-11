# Compact within-run CLN operators

This follows the cached-operator experiment in notebook 25. The new validation route shares six identical assembled operators between T and A within one complete recursion: the original uncondensed scalar residual operator, scalar potential load, broken curl load, and three original physical-equation check operators. Before reuse, native element-to-global DOF layouts and dimensions are compared. Boundary masks, scalar trace factorizations, boundary right-hand sides, recovery inverses and coloring remain role-specific.

The scalar potential load is assembled directly into the discontinuous HDiv flux space. Previously its test space was the entire hybrid compound space, whose unused cell/trace rows enlarged the sparse graph. The full compound right-hand side is zeroed and its flux prefix filled explicitly on every update.

The pool belongs to one fixed mesh/order/quadrature run. It retains native spaces and compact layout hashes; it is destroyed between repetitions. Its keys describe fixed integrands inside this workflow, not a general arbitrary-form cache. No global vector equation is solved or factored. Global vector check matrices are still retained and disclosed.

## Reproduction and acceptance

- `validation/compact_controls.py`: independent original LinearForm and uncondensed scalar applications, flat/curved meshes, p=2–4, both roles, repeated sign/scale/buffer changes.
- `validation/compact_cln_study.py`: four curved-mesh cohorts, all three stages and seven fields; original and condensed A-T, previous cached CPU/mexcuda and compact CPU/mexcuda. One first and three warmed runs, alternating route order.
- `validation/compact_host_timing.py`: idle Windows GPU-host preflight and foreign-job monitoring; retains binary/wheel identities and whole-experiment wall time.
- `validation/compact_memory_study.py`: reproducible fresh-process CPU memory lane. The accepted formal comparison finished first; this separate memory audit was interrupted by the user shutting down the metered VM. Six console observations are saved, but full worker JSON/private counters remain unrecovered. No complete whole-process memory advantage is established.
- `validation/validate_compact_cln.py`: portable source/provenance, physics, vector, memory and corruption gates.
- [Notebook 26](26_compact_cln.ipynb): executed results, first/warmed ranges, setup/operator breakdown and memory comparison.

The original acceptance limits remain: coefficients and volume-vector parity 1e-6; original scalar and physical equation residuals 1e-6; divergence and direct curl-field errors 1e-8; local curl fit 1e-10; homogeneous boundary DOFs exactly zero. Quadrature explicitly matches the original LinearForm rule, and direct field-error integration remains order 13.

Every solver timer includes setup, scalar solves, local projections, transfers and all mandatory checks. Mesh creation, imports, cross-route comparisons and reporting samples are excluded consistently. MATLAB/GPU startup is separately recorded; destruction and Engine shutdown are outside solver timers but inside the whole experiment. The unchanged CUDA binary and released Radia 5.2.3 are identified by hashes. This is validation source publication, not a new numerical package release.

CSR bytes count unique logical retained arrays, excluding native graph/allocator overhead. Per-role reference counts overlap for shared operators and must not be summed as physical compact storage. Native memory measurements include construction transients, factors and local projection arrays, but cover CPU processes only; they do not estimate the MATLAB/GPU whole-method peak.

Scope remains homogeneous, simply connected, zero-shift internal wire with curved tetrahedra. This evidence does not close C2 independent T–Omega verification, loss FEM convergence or F2/Radia API migration, and it is Codex self-review rather than independent scientific acceptance.

NGSolve uses four threads, dense BLAS and MATLAB compute use one each. Runtime metadata records other loaded pools; this is not a process-wide four-thread affinity claim. Timing is specific to this otherwise idle mdx NGSolve machine.

Remote recovery status: the VM is user-shut-down. The task-owned computation stage and setup helper await recovery, hash verification and literal-path cleanup at the next user-started session. Formal timing, host isolation and native precision-control JSON are already recovered.

## Saved matched results

| Cohort | A-T [s] | Condensed A-T [s] | Previous CPU [s] | Compact CPU [s] | Previous mexcuda [s] | Compact mexcuda [s] |
|---|---:|---:|---:|---:|---:|---:|
| p=2, h=3 mm | 1.2055 | 1.1419 | 2.5156 | 2.5011 | 2.1771 | 2.4080 |
| p=3, h=3 mm | 3.3327 | 2.2864 | 5.7522 | 5.4909 | 5.1336 | 4.7663 |
| p=3, h=2 mm | 8.7592 | 6.3413 | 18.5026 | 17.5637 | 14.1726 | 13.1610 |
| p=4, h=3 mm | 8.7001 | 4.5173 | 15.5721 | 13.9335 | 13.7422 | 11.8714 |

All values are warmed complete-run medians on NGSolve. First observations, warmed ranges, per-operator setup times and partial native memory observations are retained in notebook 26 and the four adjacent JSON evidence files. Lower-order GPU overhead can outweigh the saved assembly; memory reduction is not a universal speedup. Original and condensed A-T remain explicit controls.

Measured tradeoff: p=3/4 setup and full recursion improve, but p=2 mexcuda is 10.6% slower and CPU nearly unchanged. Both A-T controls remain faster throughout. Additional CSR storage falls 55.8–57.0%; complete process-memory superiority remains unverified after the interrupted audit.
