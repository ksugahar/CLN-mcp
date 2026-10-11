# Whole-method CLN memory on 100

This completes the memory follow-up to [notebook 26](26_compact_cln.ipynb) using 100 (T7920), as requested. [Notebook 27](27_whole_memory100.ipynb) compares original A-T, condensed A-T, cached V/Phi CPU/mexcuda and compact V/Phi CPU/mexcuda in four curved-mesh cohorts, three stages and seven field modes. Each of the 24 cases uses a fresh Python process and, for mexcuda, a fresh operation-owned MATLAB Engine session.

## Measurement boundary

The worker samples its own process and observed descendants every requested 20 ms, starting before MATLAB startup and continuing through actual owned MATLAB exit. Existing caller sessions and other jobs are outside the ancestry tree and remain untouched. Per-process native lifetime peak working-set/private-commit counters are retained independently. The common benchmark orchestration launcher and unrelated system services are excluded. All routes use the same Python imports and mesh construction.

Summed working sets can double-count shared pages. Sampled simultaneous maxima can miss short transients; sums of observed process native peaks are not simultaneous maxima and do not bound unseen short-lived processes. Private commit and resident RAM are different measures. The device operator allocation counts the two persistent owned MEX operator sets only: CUDA context, library pools and unmeasured VRAM transients are excluded. Host RAM and VRAM are reported separately.

The 100 host is shared; this is memory/accuracy validation, not an idle-host timing test. An isolated venv uses NGSolve 6.2.2607, released Radia 5.2.3, official MATLAB Engine and the unchanged SHA-identified mexcuda binary. Host/runtime, source, mesh and binary identities are saved. The GPU is 100's Quadro RTX 5000; the saved mdx timing device is a different machine.

## Time and precision

Time comparisons exclusively load the unchanged `docs/data/compact_cln_timing.json` accepted mdx NGSolve experiment. Its raw host-local byte hash is recorded, allowing Git newline normalization during portable checks. Memory-run elapsed-time fields are stripped recursively from saved solver records. No 100-host speed claim or cross-host speed ratio is introduced.

Each memory run retains the original mandatory equation, divergence, curl-fit, integrated curl-error and boundary gates. The portable gate compares all seven coefficients, sampled vector RMS and integrated mode energies against the same-mesh saved mdx original A-T reference (relative limit 1e-6). Existing mdx full volume-vector parity remains unchanged. Six corruptions reject altered residuals, fields, memory, MATLAB ownership, accidental timing fields and mdx identity.

## Reproduction

Run `validation/whole_memory100.py --directory <built-mex-directory> --workers <private-worker-directory> --output <result-json>` with the isolated native Python and official MATLAB Engine. This launches only owned sessions and awaits their exit. Retain raw worker JSON/logs outside tracked source; the aggregate result belongs in `docs/data/whole_memory100.json`.

Run `validation/validate_whole_memory100.py` and the complete portable documentation gates. Execute and save notebook 27. The numerical producers used for previous timing and interrupted memory evidence are unchanged.

The earlier user-shut-down mdx stage still awaits raw-log recovery and task-owned scratch cleanup at the next user-started VM session. This separate completed 100 audit does not claim to have recovered or removed those remote files. C2 independent T–Omega verification, loss FEM convergence and F2/Radia API migration remain open. Results are Codex self-review rather than independent scientific acceptance.

## Saved simultaneous RSS results

| Cohort | A-T MiB | Condensed A-T MiB | Previous CPU MiB | Compact CPU MiB | Previous mexcuda MiB | Compact mexcuda MiB |
|---|---:|---:|---:|---:|---:|---:|
| p=2, h=3 mm | 163.62 | 144.98 | 186.36 | 171.36 | 2194.21 | 2177.69 |
| p=3, h=3 mm | 211.83 | 174.39 | 373.12 | 296.96 | 2384.60 | 2308.50 |
| p=3, h=2 mm | 295.68 | 305.23 | 858.14 | 629.44 | 2868.70 | 2636.52 |
| p=4, h=3 mm | 316.89 | 285.50 | 926.43 | 660.11 | 2933.57 | 2665.42 |

Compact CPU reduces previous V/Phi peak RSS by 8.1–28.7%, and mexcuda by 0.8–9.1%, while original A-T remains lower in all four cohorts. These are one fresh run per condition and sampled summed working sets, not unique physical RAM or exact instantaneous peaks.
