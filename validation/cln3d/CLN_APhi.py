"""Three-dimensional round-conductor CLN: A-phi.

Based on the student notebook by N. Tanimoto. Public reproduction uses
NGSolve and the public Radia sparsesolv ICCG, with explicit gradient gauge
handling and checked residuals. Boundary conditions are retained from the
student formulation; scalar potentials use order+1 for gauge reconstruction.

Verified scope: order 1, s0=0, R0/L1/R2 only. These are normalized
coefficients for the initial axial electric field 1 V/m. Multiply R and L
by cylinder length squared to obtain the corresponding terminal circuit
coefficients. The magnetic model represents internal conductor energy;
it does not include the external return path or its inductance.

Run: python CLN_APhi.py --maxh 0.003 --stages 1 --output result.json
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
SHIFT_PHI = 1.0    # ICCG acceleration of the original phi solves
SHIFT_A = 1.1      # ICCG acceleration of the original A solves


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
    from ngsolve import (BilinearForm, GridFunction, H1, HCurl, Integrate,
                         LinearForm, curl, dx, grad)

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

    fesA = HCurl(mesh, order=order, nograds=False, dirichlet="in|out|conductorBND", complex=False)
    fesPhi = H1(mesh, order=order + 1, definedon="sig", dirichlet="in|out", complex=False)
    print(f"ndof: HCurl(A)={fesA.ndof} H1(phi)={fesPhi.ndof}")
    A, N = fesA.TnT()
    phi, psi = fesPhi.TnT()

    Rn, Ln, R_th, L_th, R_all, solves = [], [], [], [], [], []
    kw = dict(tol=tol, maxiter=maxiter)

    # stage 0: phi = h on "in", 0 on "out" (Dirichlet lifting)
    a = BilinearForm(fesPhi)
    a += sigma * grad(phi) * grad(psi) * dx
    a.Assemble()
    gfPhi = GridFunction(fesPhi)
    gfPhi.Set(h, definedon=mesh.Boundaries("in"))
    f = LinearForm(fesPhi)
    f.Assemble()
    fr = f.vec.CreateVector()
    fr.data = f.vec - a.mat * gfPhi.vec
    gfPhi, info = solve_iccg(a, fr, fesPhi, gf=gfPhi, shift=SHIFT_PHI,
                             label="phi stage 0", **kw)
    print_solve(info)
    solves.append(info)

    E = -grad(gfPhi)
    J = sigma * E
    R = 1 / Integrate(sigma * E * E * dx, mesh)
    Rn.append(R)
    R_theory = (2 * 0 + 1) / (math.pi * r * r * sigma * h)
    R_th.append(R_theory)
    R_all.append({"index": 0, "R": R, "R_theory": R_theory, "rel_err": rel_err(R, R_theory)})
    print(f"R[0] = {R:.10e}  theory {R_theory:.10e}  rel.err {rel_err(R, R_theory):.3e}")

    E0 = E
    Apot = B = None
    for nStage in range(stages):
        print(f"{nStage + 1}-stage")
        # magnetic stage: vector potential A driven by R J
        a = BilinearForm(fesA)
        a += 1 / mu * curl(A) * curl(N) * dx
        f = LinearForm(fesA)
        f += N * (R * J) * dx
        a.Assemble()
        f.Assemble()
        gfA, info = solve_iccg(a, f, fesA, kernel=gradient_kernel(fesA, a.mat), shift=SHIFT_A,
                               label=f"A stage {nStage + 1}", **kw)
        print_solve(info)
        solves.append(info)

        if nStage == 0:
            Apot = gfA
            B = curl(gfA)
        else:
            Apot = Apot + gfA
            B = B + curl(gfA)
        L = Integrate(B * B / mu * dx, mesh)
        Ln.append(L)
        L_theory = mu / (4 * 2 * (nStage + 1) * math.pi * h)
        L_th.append(L_theory)
        print(f"L[{2 * nStage + 1}] = {L:.10e}  theory {L_theory:.10e}  "
              f"rel.err {rel_err(L, L_theory):.3e}")

        # electric stage: scalar potential phi driven by -A_pot/L
        a = BilinearForm(fesPhi)
        a += sigma * grad(phi) * grad(psi) * dx
        f = LinearForm(fesPhi)
        f += sigma * grad(psi) * (-Apot / L) * dx
        a.Assemble()
        f.Assemble()
        gfPhi, info = solve_iccg(a, f, fesPhi, shift=SHIFT_PHI,
                                 label=f"phi stage {nStage + 1}", **kw)
        print_solve(info)
        solves.append(info)

        E = E - Apot / L - grad(gfPhi)
        J = sigma * E
        R = 1 / Integrate(sigma * E * E * dx, mesh)
        R_theory = (2 * nStage + 3) / (math.pi * r * r * sigma * h)
        Rn.append(R)
        R_th.append(R_theory)
        R_all.append({"index": 2 * nStage + 2, "R": R, "R_theory": R_theory,
                      "rel_err": rel_err(R, R_theory)})
        print(f"R[{2 * nStage + 2}] = {R:.10e}  theory {R_theory:.10e}  "
              f"rel.err {rel_err(R, R_theory):.3e}")

    return {
        "script": "CLN_APhi",
        "formulation": "A-phi",
        "params": {"maxh": maxh, "order": order, "stages": stages, "curve": curve,
                   "tol": tol, "maxiter": maxiter, "bonus_intorder": bonus_intorder,
                   "shift_phi": SHIFT_PHI, "shift_A": SHIFT_A,
                   "r": r, "h": h, "sigma": sigma, "mu": mu},
        "mesh": {"nv": mesh.nv, "ne": mesh.ne, "ndof_A": fesA.ndof, "ndof_phi": fesPhi.ndof},
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
    p.add_argument("--stages", type=int, default=1, help="number of CLN stages (default 1)")
    p.add_argument("--curve", type=int, default=3, help="geometry curving order (default 3)")
    p.add_argument("--bonus-intorder", type=int, default=4, help="extra volume/boundary quadrature order")
    p.add_argument("--tol", type=float, default=1e-10, help="ICCG relative tolerance")
    p.add_argument("--maxiter", type=int, default=10000, help="ICCG iteration limit")
    p.add_argument("--quick", action="store_true", help="coarse smoke run (maxh 3e-3)")
    p.add_argument("--output", default=None, help="JSON output path")
    args = p.parse_args(argv)
    if args.quick:
        args.maxh = 3e-3
    from ngsolve import TaskManager
    with TaskManager():
        data = run(args.maxh, args.order, args.stages, args.curve, args.tol, args.maxiter, args.bonus_intorder)
    write_json(args.output or f"Aphi_order{args.order}.json", data)


if __name__ == "__main__":
    main()
