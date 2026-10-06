"""Three-dimensional round-conductor CLN: A-T.

Based on the student notebook by N. Tanimoto. Public reproduction uses
NGSolve and the public Radia sparsesolv ICCG, with explicit gradient gauge
handling and checked residuals. Boundary conditions are retained from the
student formulation; scalar potentials use order+1 for gauge reconstruction.

Verified scope: order 1, s0=0, R0/L1/R2 only. These are normalized
coefficients for the initial axial electric field 1 V/m. Multiply R and L
by cylinder length squared to obtain the corresponding terminal circuit
coefficients. The magnetic model represents internal conductor energy;
it does not include the external return path or its inductance.

Run: python CLN_AT.py --maxh 0.003 --stages 1 --output result.json
Dependencies: ngsolve, numpy, scipy, radia.
See docs/06_3d_round_wire.ipynb for convergence and limitations.
"""

import argparse
import math

try:
    from ._iccg import gradient_kernel, solve_iccg, print_solve, rel_err, write_json, sample_fields
except ImportError:  # run as a script
    from _iccg import gradient_kernel, solve_iccg, print_solve, rel_err, write_json, sample_fields

# geometry and material (original values)
R_WIRE = 0.01      # radius r
H_WIRE = 0.01      # length h
SIGMA = 1e6
MU = 4 * math.pi * 1e-7
SHIFT = 1.1        # ICCG acceleration used by the original for every solve


def make_mesh(maxh, curve):
    from netgen.occ import Cylinder, OCCGeometry, Z
    from ngsolve import Mesh

    conductor = Cylinder((0, 0, 0), Z, r=R_WIRE, h=H_WIRE)
    conductor.maxh = maxh
    conductor.faces.name = "conductorBND"
    conductor.faces.Max(Z).name = "out"
    conductor.faces.Min(Z).name = "in"
    conductor.mat("sig")
    geo = OCCGeometry(conductor)
    return Mesh(geo.GenerateMesh(maxh=maxh)).Curve(curve)


def run(maxh=1e-3, order=1, stages=1, curve=3, tol=1e-10, maxiter=10000, bonus_intorder=4):
    from ngsolve import (BilinearForm, Cross, HCurl, Integrate, LinearForm,
                         curl, ds, dx, specialcf)

    if isinstance(bonus_intorder, bool) or not isinstance(bonus_intorder, int) or bonus_intorder < 0:
        raise ValueError("bonus_intorder must be a nonnegative integer")
    r, h, sigma, mu = R_WIRE, H_WIRE, SIGMA, MU
    import warnings
    if stages != 1:
        warnings.warn('Only R0/L1/R2 (one stage) is validated; higher-stage coefficients are unverified', RuntimeWarning)
    # Match quadrature across coupled scalar/vector forms on curved elements.
    dx = dx(bonus_intorder=bonus_intorder)
    mesh = make_mesh(maxh, curve)
    print(f"mesh: nv={mesh.nv} nedge={mesh.nedge} nface={mesh.nface} ne={mesh.ne}")

    fesA = HCurl(mesh, order=order, nograds=True, dirichlet="in|out|conductorBND", complex=False)
    fesT = HCurl(mesh, order=order, nograds=True, complex=False)
    print(f"ndof: HCurl(A)={fesA.ndof} HCurl(T)={fesT.ndof}")
    A, N = fesA.TnT()
    T, W = fesT.TnT()
    kerA = gradient_kernel(fesA)  # null spaces of the singular curl-curl matrices
    kerT = gradient_kernel(fesT)

    Rn, Ln, R_th, L_th, R_all, solves = [], [], [], [], [], []
    solve_kw = dict(shift=SHIFT, tol=tol, maxiter=maxiter)

    # stage 0: T from the surface source E_s = e_z
    Es = (0, 0, 1)
    n = specialcf.normal(mesh.dim)
    a = BilinearForm(fesT)
    a += 1 / sigma * curl(T) * curl(W) * dx
    f = LinearForm(fesT)
    f += -Cross(Es, W.Trace()) * n * ds("conductorBND", bonus_intorder=bonus_intorder)
    a.Assemble()
    f.Assemble()
    gfT, info = solve_iccg(a, f, fesT, kernel=kerT, label="T stage 0", **solve_kw)
    print_solve(info)
    solves.append(info)

    J = curl(gfT)
    E = 1. / sigma * J
    R = 1 / Integrate(sigma * E * E * dx, mesh)
    Rn.append(R)
    R_theory = (2 * 0 + 1) / (math.pi * r * r * sigma * h)
    R_th.append(R_theory)
    R_all.append({"index": 0, "R": R, "R_theory": R_theory, "rel_err": rel_err(R, R_theory)})
    print(f"R[0] = {R:.10e}  theory {R_theory:.10e}  rel.err {rel_err(R, R_theory):.3e}")

    E0 = E
    B = None
    for nStage in range(stages):
        print(f"{nStage + 1}-stage")
        # magnetic stage: vector potential A driven by R J
        a = BilinearForm(fesA)
        a += 1 / mu * curl(A) * curl(N) * dx
        f = LinearForm(fesA)
        f += N * (R * J) * dx
        a.Assemble()
        f.Assemble()
        gfA, info = solve_iccg(a, f, fesA, kernel=kerA, label=f"A stage {nStage + 1}", **solve_kw)
        print_solve(info)
        solves.append(info)

        if nStage == 0:
            B = curl(gfA)
        else:
            B = B + curl(gfA)
        L = Integrate(B * B / mu * dx, mesh)
        Ln.append(L)
        L_theory = mu / (4 * 2 * (nStage + 1) * math.pi * h)
        L_th.append(L_theory)
        print(f"L[{2 * nStage + 1}] = {L:.10e}  theory {L_theory:.10e}  "
              f"rel.err {rel_err(L, L_theory):.3e}")

        # electric stage: current vector potential T driven by -B/L
        a = BilinearForm(fesT)
        a += 1. / sigma * curl(T) * curl(W) * dx
        f = LinearForm(fesT)
        f += W * (-B / L) * dx
        a.Assemble()
        f.Assemble()
        gfT, info = solve_iccg(a, f, fesT, kernel=kerT, label=f"T stage {nStage + 1}", **solve_kw)
        print_solve(info)
        solves.append(info)

        J = J + curl(gfT)
        E = J / sigma
        R = 1 / Integrate(sigma * E * E * dx, mesh)
        R_theory = (2 * nStage + 3) / (math.pi * r * r * sigma * h)
        Rn.append(R)
        R_th.append(R_theory)
        R_all.append({"index": 2 * nStage + 2, "R": R, "R_theory": R_theory,
                      "rel_err": rel_err(R, R_theory)})
        print(f"R[{2 * nStage + 2}] = {R:.10e}  theory {R_theory:.10e}  "
              f"rel.err {rel_err(R, R_theory):.3e}")

    return {
        "script": "CLN_AT",
        "formulation": "A-T",
        "params": {"maxh": maxh, "order": order, "stages": stages, "curve": curve,
                   "tol": tol, "maxiter": maxiter, "bonus_intorder": bonus_intorder, "shift": SHIFT,
                   "r": r, "h": h, "sigma": sigma, "mu": mu},
        "mesh": {"nv": mesh.nv, "ne": mesh.ne, "ndof_A": fesA.ndof, "ndof_T": fesT.ndof},
        "Rn": Rn, "Ln": Ln, "R_theory": R_th, "L_theory": L_th,
        "R_rel_err": [rel_err(x, y) for x, y in zip(Rn, R_th)],
        "L_rel_err": [rel_err(x, y) for x, y in zip(Ln, L_th)],
        "R_all": R_all,
        "solves": solves,
        "field_profiles": sample_fields(mesh,E0,B,E,r,h) if stages == 1 else None,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--maxh", type=float, default=1e-3, help="mesh size (default 1e-3)")
    p.add_argument("--order", type=int, default=1, help="FE order (default 1)")
    p.add_argument("--stages", type=int, default=1, help="number of CLN stages (verified: 1)")
    p.add_argument("--curve", type=int, default=3, help="geometry curving order (default 3)")
    p.add_argument("--bonus-intorder", type=int, default=4, help="extra volume/boundary quadrature order")
    p.add_argument("--tol", type=float, default=1e-10, help="ICCG relative tolerance")
    p.add_argument("--maxiter", type=int, default=10000, help="ICCG iteration limit")
    p.add_argument("--quick", action="store_true", help="coarse smoke run (maxh 3e-3; higher stages unverified)")
    p.add_argument("--output", default=None, help="JSON output path")
    args = p.parse_args(argv)
    if args.quick:
        args.maxh = 3e-3
    from ngsolve import TaskManager
    with TaskManager():
        data = run(args.maxh, args.order, args.stages, args.curve, args.tol, args.maxiter, args.bonus_intorder)
    write_json(args.output or f"AT_order{args.order}.json", data)


if __name__ == "__main__":
    main()
