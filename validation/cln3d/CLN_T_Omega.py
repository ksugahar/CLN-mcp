"""Three-dimensional round-conductor CLN: T-Omega.

Based on the student notebook by N. Tanimoto. Public reproduction uses
NGSolve and the public Radia sparsesolv ICCG, with explicit gradient gauge
handling and checked residuals. Boundary conditions are retained from the
student formulation; scalar potentials use order+1 for gauge reconstruction.

Verified scope: order 1, s0=0, R0/L1/R2 only. These are normalized
coefficients for the initial axial electric field 1 V/m. Multiply R and L
by cylinder length squared to obtain the corresponding terminal circuit
coefficients. The magnetic model represents internal conductor energy;
it does not include the external return path or its inductance.

Run: python CLN_T_Omega.py --maxh 0.003 --stages 1 --output result.json
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


def run(maxh=1e-3, order=1, stages=1, curve=3, tol=1e-10, maxiter=10000):
    from ngsolve import (BilinearForm, CoefficientFunction, Cross, H1, HCurl, Integrate, LinearForm, curl, ds, dx, grad,
                         specialcf)

    r, h, sigma, mu = R_WIRE, H_WIRE, SIGMA, MU
    mesh = make_mesh(maxh, curve)
    print(f"mesh: nv={mesh.nv} nedge={mesh.nedge} nface={mesh.nface} ne={mesh.ne}")

    fesT = HCurl(mesh, order=order, nograds=True, complex=False)
    fesOmega = H1(mesh, order=order + 1, definedon="sig", complex=False)
    print(f"ndof: HCurl(T)={fesT.ndof} H1(Omega)={fesOmega.ndof}")
    Omega, psi = fesOmega.TnT()
    T, W = fesT.TnT()
    kerT = gradient_kernel(fesT)  # null space of the singular curl-curl matrix

    Rn, Ln, R_th, L_th, R_all, solves = [], [], [], [], [], []
    solve_kw = dict(shift=SHIFT, tol=tol, maxiter=maxiter)

    # stage 0: T from the surface source E_s = e_z
    Es = CoefficientFunction((0, 0, 1))
    n = specialcf.normal(mesh.dim)
    a = BilinearForm(fesT)
    a += 1 / sigma * curl(T) * curl(W) * dx
    f = LinearForm(fesT)
    f += -Cross(Es, W.Trace()) * n * ds("conductorBND")
    a.Assemble()
    f.Assemble()
    gfT, info = solve_iccg(a, f, fesT, kernel=kerT, label="T stage 0", **solve_kw)
    print_solve(info)
    solves.append(info)

    Tpot = gfT
    J = curl(gfT)
    E = J / sigma
    R = 1 / Integrate(sigma * E * E * dx, mesh)
    Rn.append(R)
    R_theory = (2 * 0 + 1) / (math.pi * r * r * sigma * h)
    R_th.append(R_theory)
    R_all.append({"index": 0, "R": R, "R_theory": R_theory, "rel_err": rel_err(R, R_theory)})
    print(f"R[0] = {R:.10e}  theory {R_theory:.10e}  rel.err {rel_err(R, R_theory):.3e}")

    E0 = E
    H = None
    for nStage in range(stages):
        print(f"{nStage + 1}-stage")
        # magnetic stage: reduced scalar potential Omega
        a = BilinearForm(fesOmega)
        a += mu * grad(Omega) * grad(psi) * dx
        f = LinearForm(fesOmega)
        f += -mu * grad(psi) * (Tpot) * dx
        a.Assemble()
        f.Assemble()
        gfOmega, info = solve_iccg(a, f, fesOmega, label=f"Omega stage {nStage + 1}", **solve_kw)
        print_solve(info)
        solves.append(info)

        if nStage == 0:
            H = R * (Tpot + grad(gfOmega))
        else:
            H = H + R * (Tpot + grad(gfOmega))
        B = mu * H
        L = Integrate(B * B / mu * dx, mesh)
        Ln.append(L)
        L_theory = mu / (4 * 2 * (nStage + 1) * math.pi * h)
        L_th.append(L_theory)
        print(f"L[{2 * nStage + 1}] = {L:.10e}  theory {L_theory:.10e}  "
              f"rel.err {rel_err(L, L_theory):.3e}")

        # electric stage: current vector potential T
        a = BilinearForm(fesT)
        a += 1. / sigma * curl(T) * curl(W) * dx
        f = LinearForm(fesT)
        f += W * (-B / L) * dx
        a.Assemble()
        f.Assemble()
        gfT, info = solve_iccg(a, f, fesT, kernel=kerT, label=f"T stage {nStage + 1}", **solve_kw)
        print_solve(info)
        solves.append(info)

        Tpot = Tpot + gfT
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
        "script": "CLN_T_Omega",
        "formulation": "T-Omega",
        "params": {"maxh": maxh, "order": order, "stages": stages, "curve": curve,
                   "tol": tol, "maxiter": maxiter, "shift": SHIFT,
                   "r": r, "h": h, "sigma": sigma, "mu": mu},
        "mesh": {"nv": mesh.nv, "ne": mesh.ne, "ndof_T": fesT.ndof, "ndof_Omega": fesOmega.ndof},
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
    p.add_argument("--tol", type=float, default=1e-10, help="ICCG relative tolerance")
    p.add_argument("--maxiter", type=int, default=10000, help="ICCG iteration limit")
    p.add_argument("--quick", action="store_true", help="coarse smoke run (maxh 3e-3, 1 stage)")
    p.add_argument("--output", default=None, help="JSON output path")
    args = p.parse_args(argv)
    if args.quick:
        args.maxh, args.stages = 3e-3, 1
    from ngsolve import TaskManager
    with TaskManager():
        data = run(args.maxh, args.order, args.stages, args.curve, args.tol, args.maxiter)
    write_json(args.output or f"TOmega_order{args.order}.json", data)


if __name__ == "__main__":
    main()
