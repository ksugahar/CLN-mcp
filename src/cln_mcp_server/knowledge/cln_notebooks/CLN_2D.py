"""2D Cauer ladder network (CLN) of a round conductor (cross-section), A_z formulation.

What it computes
    The CLN parameters R_0, L_1, R_2, L_3 (two stages by default, as in the
    original) per unit length of an infinitely long round wire (radius
    r = 0.01 m, sigma = 1e6 S/m, mu = mu_0) and compares them with the
    analytic values
        R_theory(k) = (2k+1) / (pi r^2 sigma),     k = 0, 1, 2, ...
        L_theory(k) = mu / (8 pi (k+1)),           k = 0, 1, 2, ...
    With two or more stages it also prints the inner products between the
    stage fields used to check the orthogonality of the CLN modes
    (A_1.A_3, B_1.H_3, J_2.J_2, J_2.E_4, J_2.J_4, and 1/int J_2^2/sigma).

Formulation
    Scalar A_z in H1 (order 1), A_z = 0 on the outer circle:
      (1/mu) grad A . grad v = v J,
    A_pot accumulates R A, B accumulates R grad A, L = int R J A_pot,
    J <- J - sigma A_pot / L,  R = 1 / int J^2/sigma, starting from E = 1.
    Linear systems: NGSolve CG with a BDDC preconditioner (tol 1e-8, as in
    the original); non-convergence raises RuntimeError.

How to run
    python CLN_2D.py                      # original settings (maxh 1e-3, 2 stages)
    python CLN_2D.py --quick              # coarse smoke run
    python CLN_2D.py --maxh 5e-4 --stages 4 --output 2D.json
    Results are written as JSON (default 2D_order<order>.json).

Dependencies
    ngsolve, netgen (OCC geometry), numpy (no sparsesolv needed).

Original notebook by N. Tanimoto; ported to the public Radia sparsesolv.
"""

import argparse
import math

try:
    from ._iccg import solve_cg, print_solve, rel_err, write_json
except ImportError:  # run as a script
    from _iccg import solve_cg, print_solve, rel_err, write_json

R_WIRE = 0.01
SIGMA = 1e6
MU = 4 * math.pi * 1e-7
CG_TOL = 1e-8
CG_MAXITER = 10000


def make_mesh(maxh, curve):
    from netgen.occ import Circle, OCCGeometry
    from ngsolve import Mesh

    cir = Circle((0, 0), R_WIRE).Face()
    cir.edges.name = "outer"
    geo = OCCGeometry(cir, dim=2)
    return Mesh(geo.GenerateMesh(maxh=maxh)).Curve(curve)


def run(maxh=1e-3, order=1, stages=2, curve=3):
    from ngsolve import (BilinearForm, CoefficientFunction, GridFunction, H1,
                         InnerProduct, Integrate, LinearForm, Preconditioner,
                         dx, grad)

    r, sigma, mu = R_WIRE, SIGMA, MU
    mesh = make_mesh(maxh, curve)
    print(f"mesh: nv={mesh.nv} ne={mesh.ne}")
    fes = H1(mesh, order=order, dirichlet="outer")
    u = fes.TrialFunction()
    v = fes.TestFunction()

    E = CoefficientFunction(1)
    J = sigma * E
    Rn, Ln, R_th, L_th, solves = [], [], [], [], []
    fields = {}
    Apotential = B = None

    for nStage in range(stages):
        print(f"{nStage + 1}-stage")
        R = 1 / Integrate(J * J / sigma * dx(bonus_intorder=4), mesh)
        Rn.append(R)
        R_theory = (2 * nStage + 1) / (math.pi * r * r * sigma)
        R_th.append(R_theory)
        print(f"R[{2 * nStage}] = {R:.10e}  theory {R_theory:.10e}  "
              f"rel.err {rel_err(R, R_theory):.3e}")

        # magnetostatic stage
        a = BilinearForm(fes)
        a += 1 / mu * grad(u) * grad(v) * dx(bonus_intorder=4)
        Precon = Preconditioner(a, type="bddc")
        f = LinearForm(fes)
        f += v * J * dx(bonus_intorder=4)
        gfA = GridFunction(fes)
        a.Assemble()
        f.Assemble()
        info = solve_cg(a.mat, Precon.mat, f.vec, gfA.vec, fes.FreeDofs(),
                        tol=CG_TOL, maxiter=CG_MAXITER, label=f"A stage {nStage + 1}")
        print_solve(info)
        solves.append(info)

        if nStage == 0:
            Apotential = R * (gfA)
            B = R * grad(gfA)
            fields["A1"], fields["B1"] = Apotential, B
        else:
            Apotential = Apotential + R * (gfA)
            B = B + R * grad(gfA)
            if nStage == 1:
                fields["A3"], fields["B3"] = Apotential, B

        L = Integrate(R * J * Apotential * dx(bonus_intorder=4), mesh)
        Ln.append(L)
        L_theory = mu / (4 * 2 * (nStage + 1) * math.pi)
        L_th.append(L_theory)
        print(f"L[{2 * nStage + 1}] = {L:.10e}  theory {L_theory:.10e}  "
              f"rel.err {rel_err(L, L_theory):.3e}")

        J = J - sigma * Apotential / L
        if nStage == 0:
            fields["J2"] = J
        elif nStage == 1:
            fields["J4"] = J

    inner = {}
    if stages >= 2:
        A1, A3, B1, B3 = fields["A1"], fields["A3"], fields["B1"], fields["B3"]
        J2, J4 = fields["J2"], fields["J4"]
        inner = {
            "A_InnerProduct": Integrate(InnerProduct(A1, A3), mesh),
            "B1_H3_InnerProduct": Integrate(InnerProduct(B1, B3 / mu), mesh),
            "J2_J2_InnerProduct": Integrate(InnerProduct(J2, J2), mesh),
            "J2_E4_InnerProduct": Integrate(InnerProduct(J2, J4 / sigma), mesh),
            "J_InnerProduct": Integrate(J2 * J4, mesh),
            "R2": 1 / Integrate(J2 * J2 / sigma, mesh),
        }
        for k, val in inner.items():
            print(f"{k}: {val:.10e}")

    return {
        "script": "CLN_2D",
        "formulation": "2D A_z",
        "params": {"maxh": maxh, "order": order, "stages": stages, "curve": curve,
                   "cg_tol": CG_TOL, "cg_maxiter": CG_MAXITER, "preconditioner": "bddc",
                   "r": r, "sigma": sigma, "mu": mu},
        "mesh": {"nv": mesh.nv, "ne": mesh.ne, "ndof": fes.ndof},
        "Rn": Rn, "Ln": Ln, "R_theory": R_th, "L_theory": L_th,
        "R_rel_err": [rel_err(x, y) for x, y in zip(Rn, R_th)],
        "L_rel_err": [rel_err(x, y) for x, y in zip(Ln, L_th)],
        "inner_products": inner,
        "solves": solves,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--maxh", type=float, default=1e-3, help="mesh size (default 1e-3)")
    p.add_argument("--order", type=int, default=1, help="FE order (default 1)")
    p.add_argument("--stages", type=int, default=2, help="number of CLN stages (default 2)")
    p.add_argument("--curve", type=int, default=3, help="geometry curving order (default 3)")
    p.add_argument("--quick", action="store_true", help="coarse smoke run (maxh 3e-3)")
    p.add_argument("--output", default=None, help="JSON output path")
    args = p.parse_args(argv)
    if args.quick:
        args.maxh = 3e-3
    data = run(args.maxh, args.order, args.stages, args.curve)
    write_json(args.output or f"2D_order{args.order}.json", data)


if __name__ == "__main__":
    main()
