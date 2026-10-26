"""3D Cauer ladder network (CLN) of a round conductor, A formulation with gauge correction.

What it computes
    The CLN parameters R_0, L_1, R_2, L_3, ... of a conducting disc-shaped
    cylinder (radius r = 1 m, height h = 0.2 m, sigma = 3.4e7 S/m,
    mu = mu_0) driven by a uniform axial field, and compares them with the
    analytic values of the round wire,
        R_theory(k) = (2k+1) / (pi r^2 sigma h),   k = 0, 1, 2, ...
        L_theory(k) = mu / (8 pi (k+1) h),         k = 0, 1, 2, ...

Formulation
    Mesh: OCC cylinder, maxh 0.1, end faces identified (CLOSESURFACES) and
    refined in z at 0.25/0.5/0.75 (three extra prism layers), curved order 5.
    Magnetic stage: A in HCurl (type1, n x A = 0 on all faces),
      (1/mu) curl A . curl v = v . J,
    solved with shifted ICCG (shift acc[stage] = 1.01, tol 1e-12, at most
    10000 iterations).  Gauge correction: H1 potential u (zero on all faces),
      grad u . grad v = -grad v . A,
    solved with NGSolve CG and the "local" (Jacobi) preconditioner
    (tol 1e-12, at most 10000 iterations).
    A_pot accumulates R (A + grad u), B accumulates R curl A,
    L = int R J . A_pot,  J <- J - sigma A_pot / L,
    R = 1 / int J.J/sigma, starting from E = e_z.
    The A system is singular (gradient null space: lowest-order vertex
    gradients plus the gradient-type higher-order basis functions of the
    type1 space, detected as zero columns of the curl-curl matrix); the
    null-space component of its right-hand side is projected out before the
    solve and reported per solve.  This projection changes the load; it is not a choice
    of gauge.  With dx(bonus_intorder=4) on every volume integral the removed
    component is small (measured 2026-10-06: <= 4.3e-7, default meshes <= 1.4e-8)
    and is gated by _iccg.KERNEL_FRACTION_BOUND;
    the residual against the original load is stored as original_rhs_residual.
    A gauge term alpha G G^T makes the gradient directions nonsingular for ICCG.  ICCG at tol 1e-12 stops at the
    floating-point floor (about 2e-12); the helper accepts residuals within
    10 tol and checks the explicit residual against 1e3 tol.

How to run
    python A_ICCG_production.py           # original settings (maxh 0.1, 10 stages)
    python A_ICCG_production.py --quick   # coarse smoke run
    python A_ICCG_production.py --maxh 0.2 --stages 3 --output A.json
    Results are written as JSON (default A_ICCG_order<order>.json).

Dependencies
    ngsolve, netgen (OCC geometry), numpy, scipy, and radia (for the public
    sparsesolv ICCG, radia.sparsesolv_ngsolve).

Original notebook by N. Tanimoto; ported to the public Radia sparsesolv.
"""

import argparse
import math

try:
    from ._iccg import gradient_kernel, solve_cg, solve_iccg, print_solve, rel_err, write_json
except ImportError:  # run as a script
    from _iccg import gradient_kernel, solve_cg, solve_iccg, print_solve, rel_err, write_json

R_DISC = 1.0
H_DISC = 0.2
SIGMA = 3.4e7
MU = 4 * math.pi * 1e-7
ACC = 1.01          # ICCG acceleration of the original (acc[nStage] = 1.01 for all stages)
ICCG_TOL = 1e-12
ICCG_MAXITER = 10000
GAUGE_TOL = 1e-12
GAUGE_MAXITER = 10000


def make_mesh(maxh, curve):
    from netgen.occ import Cylinder, IdentificationType, OCCGeometry, Z
    from ngsolve import Mesh

    cyl = Cylinder((0, 0, 0), Z, r=R_DISC, h=H_DISC, bottom="bot", top="top").bc("S_side")
    cyl.faces.Min(Z).name = "S_bottom"
    cyl.faces.Max(Z).name = "S_top"
    cyl.faces.Min(Z).Identify(cyl.faces.Max(Z), "bot-top", type=IdentificationType.CLOSESURFACES)
    ngmesh = OCCGeometry(cyl).GenerateMesh(maxh=maxh)
    ngmesh.ZRefine("bot-top", [0.25, 0.5, 0.75])
    mesh = Mesh(ngmesh)
    mesh.Curve(curve)
    return mesh


def run(maxh=0.1, order=1, stages=10, curve=5, acc=None):
    from ngsolve import (BilinearForm, CoefficientFunction, GridFunction, H1,
                         HCurl, Integrate, LinearForm, Preconditioner, curl,
                         dx, grad)

    r, h, sigma, mu = R_DISC, H_DISC, SIGMA, MU
    if acc is None:
        acc = [ACC] * stages
    mesh = make_mesh(maxh, curve)
    print(f"mesh: nv={mesh.nv} nedge={mesh.nedge} nface={mesh.nface} ne={mesh.ne}")

    fes = HCurl(mesh, order=order, dirichlet="S_side|S_bottom|S_top", type1=True)
    gauge = H1(mesh, order=order, dirichlet="S_side|S_bottom|S_top")
    print(f"ndof: HCurl={fes.ndof} H1={gauge.ndof} total={fes.ndof + gauge.ndof}")
    u = fes.TrialFunction()
    v = fes.TestFunction()
    uu = gauge.TrialFunction()
    vv = gauge.TestFunction()
    ker = None  # null space of the singular curl-curl matrix (built from the first matrix)

    E = CoefficientFunction((0, 0, 1))
    J = sigma * E
    Rn, Ln, R_th, L_th, solves = [], [], [], [], []
    Apotential = B = None

    for nStage in range(stages):
        print(f"{nStage + 1}-stage")
        R = 1 / Integrate(J * J / sigma * dx(bonus_intorder=4), mesh)
        Rn.append(R)
        R_theory = (2 * nStage + 1) / (math.pi * r * r * sigma * h)
        R_th.append(R_theory)
        print(f"R[{2 * nStage}] = {R:.10e}  theory {R_theory:.10e}  "
              f"rel.err {rel_err(R, R_theory):.3e}")

        # magnetostatic stage (ICCG)
        a = BilinearForm(fes)
        a += 1 / mu * curl(u) * curl(v) * dx(bonus_intorder=4)
        f = LinearForm(fes)
        f += v * J * dx(bonus_intorder=4)
        a.Assemble()
        f.Assemble()
        if ker is None:
            ker = gradient_kernel(fes, a.mat)
        gfA, info = solve_iccg(a, f, fes, kernel=ker, shift=acc[nStage], tol=ICCG_TOL,
                               maxiter=ICCG_MAXITER, label=f"A stage {nStage + 1}")
        print_solve(info)
        solves.append(info)

        # gauge correction (NGSolve CG, local preconditioner)
        aa = BilinearForm(gauge)
        aa += grad(uu) * grad(vv) * dx(bonus_intorder=4)
        cc = Preconditioner(aa, "local")
        ff = LinearForm(gauge)
        ff += grad(vv) * gfA * dx(bonus_intorder=4)
        aa.Assemble()
        ff.Assemble()
        gfu = GridFunction(gauge)
        rhs = ff.vec.CreateVector()
        rhs.data = -ff.vec
        info = solve_cg(aa.mat, cc.mat, rhs, gfu.vec, gauge.FreeDofs(), tol=GAUGE_TOL,
                        maxiter=GAUGE_MAXITER, label=f"gauge stage {nStage + 1}")
        print_solve(info)
        solves.append(info)

        if nStage == 0:
            Apotential = R * (gfA + grad(gfu))
            B = R * curl(gfA)
        else:
            Apotential = Apotential + R * (gfA + grad(gfu))
            B = B + R * curl(gfA)

        L = Integrate(R * J * Apotential * dx(bonus_intorder=4), mesh)
        Ln.append(L)
        L_theory = mu / (4 * 2 * (nStage + 1) * math.pi * h)
        L_th.append(L_theory)
        print(f"L[{2 * nStage + 1}] = {L:.10e}  theory {L_theory:.10e}  "
              f"rel.err {rel_err(L, L_theory):.3e}")

        J = J - sigma * Apotential / L

    return {
        "script": "A_ICCG_production",
        "formulation": "A with H1 gauge correction",
        "params": {"maxh": maxh, "order": order, "stages": stages, "curve": curve,
                   "acc": list(acc), "iccg_tol": ICCG_TOL, "iccg_maxiter": ICCG_MAXITER,
                   "gauge_tol": GAUGE_TOL, "gauge_maxiter": GAUGE_MAXITER,
                   "zrefine": [0.25, 0.5, 0.75],
                   "r": r, "h": h, "sigma": sigma, "mu": mu},
        "mesh": {"nv": mesh.nv, "ne": mesh.ne, "ndof_HCurl": fes.ndof, "ndof_H1": gauge.ndof},
        "Rn": Rn, "Ln": Ln, "R_theory": R_th, "L_theory": L_th,
        "R_rel_err": [rel_err(x, y) for x, y in zip(Rn, R_th)],
        "L_rel_err": [rel_err(x, y) for x, y in zip(Ln, L_th)],
        "solves": solves,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--maxh", type=float, default=0.1, help="mesh size (default 0.1)")
    p.add_argument("--order", type=int, default=1, help="FE order (default 1)")
    p.add_argument("--stages", type=int, default=None, help="number of CLN stages (default 10; 3 with --quick)")
    p.add_argument("--curve", type=int, default=5, help="geometry curving order (default 5)")
    p.add_argument("--quick", action="store_true", help="coarse smoke run (maxh 0.2, 3 stages)")
    p.add_argument("--output", default=None, help="JSON output path")
    args = p.parse_args(argv)
    if args.stages is None:
        args.stages = 3 if args.quick else 10
    if args.quick:
        args.maxh = 0.2
    from ngsolve import TaskManager
    with TaskManager():
        data = run(args.maxh, args.order, args.stages, args.curve)
    write_json(args.output or f"A_ICCG_order{args.order}.json", data)


if __name__ == "__main__":
    main()
