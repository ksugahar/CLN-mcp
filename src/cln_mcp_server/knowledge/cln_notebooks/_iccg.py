"""Checked CLN solves using the public Radia sparsesolv ICCG.

For curl-curl systems, project the load onto the complement of the supplied
gradient kernel and report the removed fraction. This changes the load; it
must not be described as merely choosing a gauge. Reject a fraction above
1e-4 before solving. Add alpha G G^T, with positive diagonal-scaled alpha,
to make the gradient directions nonsingular. Verify A G = 0 on free dofs.
Check the residual against the original curl-curl matrix and projected load;
also report the residual against the original load. Do not silently change
solver backend or increase the IC shift. The tested shift is 1.0 for phi,
1.1 for vector potentials. Gauge weight is independent of the IC shift.

Boundary diagonals are filled only on constrained dofs, whose solution is
zero in this correction solve. Nonzero boundary lifting is supplied by the
caller in the right-hand side and preserved in the output GridFunction.

Validated scope: simply connected cylindrical conductor, order 1, one
magnetic/electric stage, s0=0. No general multiply connected gauge claim.
"""

import math

import numpy as np

RESIDUAL_BOUND_FACTOR = 1e3
CG_RESIDUAL_FLOOR = 1e-6
DEFAULT_MAXITER = 10000
KERNEL_FRACTION_BOUND = 1e-4
STALL_FACTOR = 10


def _sparsesolv_solver_class():
    # sparsesolv is distributed inside the public Radia package; radia must be
    # imported before its compiled submodule.
    import radia  # noqa: F401
    from radia.sparsesolv_ngsolve import SparseSolvSolver
    return SparseSolvSolver


def _free_mask(freedofs, ndof):
    return np.fromiter(freedofs, dtype=bool, count=ndof)


def explicit_residual(mat, u, rhs, free):
    """Relative residual ||rhs - mat*u|| / ||rhs|| on the free dofs ``free``."""
    r = rhs.CreateVector()
    r.data = rhs - mat * u
    rv = r.FV().NumPy()[free]
    fv = rhs.FV().NumPy()[free]
    fnorm = float(np.linalg.norm(fv))
    rnorm = float(np.linalg.norm(rv))
    if fnorm == 0.0:
        return 0.0 if rnorm == 0.0 else math.inf
    return rnorm / fnorm


def gradient_kernel(fes, amat=None):
    """Kernel basis of a curl-curl matrix on the HCurl space ``fes``.

    Returns ``(G, kfree)``: the discrete gradient ``G`` (scipy CSR, HCurl dofs x
    H1 dofs, from ``fes.CreateGradient()``) and the free-dof mask of the
    matching H1 space.  ``G[:, kfree]`` spans the null space of
    ``curl . curl`` restricted to the free HCurl dofs.  If the assembled
    curl-curl matrix ``amat`` is given, dofs whose matrix column vanishes
    (pure gradient basis functions not covered by CreateGradient, as for
    ``type1=True`` on prisms) are appended as unit columns.
    """
    import scipy.sparse as sp

    G, h1 = fes.CreateGradient()
    # G.CSR() returns views into NGSolve memory: copy before G is released
    Gs = sp.csr_matrix(G.CSR()).copy()
    kfree = _free_mask(h1.FreeDofs(), h1.ndof)
    if amat is None:
        return Gs, kfree

    # Some spaces keep higher-order basis functions that are exact gradients
    # (e.g. HCurl type1 on prisms: gradient-type quad-face functions) while
    # CreateGradient only returns the lowest-order part.  Every dof whose
    # curl-curl column vanishes is such a kernel function; add it as a unit
    # column.  (Lowest-order edge functions never have a zero column, so the
    # added columns are independent of G.)
    A = sp.csr_matrix(amat.CSR()).copy()
    amax = float(abs(A).max())
    colmax = np.asarray(abs(A).max(axis=0).todense()).ravel()
    extra = np.flatnonzero(colmax <= 1e-12 * amax)
    if len(extra) == 0:
        return Gs, kfree
    E = sp.csr_matrix((np.ones(len(extra)), (extra, np.arange(len(extra)))),
                      shape=(fes.ndof, len(extra)))
    return sp.hstack([Gs, E], format="csr"), np.concatenate([kfree, np.ones(len(extra), bool)])


def _project_out(b, G):
    """Euclidean projection of ``b`` onto the orthogonal complement of range(G)."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    G = G[:, np.asarray(G.getnnz(axis=0)).ravel() > 0]
    if G.shape[1] == 0:
        return b.copy()
    M = (G.T @ G).tocsc()
    # G may have a constant null vector (pure-Neumann H1); a relative
    # regularisation of 1e-14 removes the singularity without affecting G c.
    M = M + 1e-14 * float(M.diagonal().max()) * sp.identity(M.shape[0], format="csc")
    solve = spla.factorized(M)
    # two passes (one step of iterative refinement): a single pass leaves a
    # kernel remainder of ~1e-8 relative, which already stalls ICCG at 1e-8
    for _ in range(2):
        b = b - G @ solve(G.T @ b)
    return b


def solve_iccg(a, f, fes, shift=1.1, tol=1e-10, maxiter=None, gf=None,
               kernel=None, residual_bound=None,
               kernel_fraction_bound=KERNEL_FRACTION_BOUND,
               diagonal_scaling=False, auto_shift=False, label="", gauge_weight=1.0):
    """Solve ``a u = f`` on the free dofs of ``fes`` with sparsesolv ICCG.

    Args:
        a: assembled BilinearForm (its ``.mat`` is used).
        f: assembled LinearForm, or an assembled right-hand-side BaseVector
           (e.g. a Dirichlet-lifted ``f.vec - a.mat * g.vec``).
        fes: the finite element space (supplies ``FreeDofs()``).
        shift: IC shift (acceleration) parameter.
        tol: relative residual tolerance.
        maxiter: iteration limit (default ``DEFAULT_MAXITER``).
        gf: GridFunction to receive the solution on the free dofs; a new one is
            created if None.  Non-free entries are not modified.
        kernel: optional ``(G, kfree)`` from :func:`gradient_kernel` for a
            singular curl-curl system.  The component of the rhs in
            range(G[:, kfree]) = ker(A) is removed before solving, so that the
            singular system is consistent; its relative size is reported as
            ``rhs_kernel_fraction``.
        residual_bound: explicit-residual bound (default
            ``RESIDUAL_BOUND_FACTOR * tol``).
        kernel_fraction_bound: raise if the removed kernel component exceeds
            this fraction of the rhs (default ``KERNEL_FRACTION_BOUND``).
        diagonal_scaling, auto_shift: sparsesolv options; both are off by
            default, matching the fixed-shift, unscaled ICCG of the original
            scripts.
        label: name used in error messages and in the returned info.

    Returns:
        (gf, info) where ``info`` is a JSON-serialisable dict.

    Raises:
        RuntimeError: on non-convergence, an explicit residual that is not
            finite or above ``residual_bound``, or a kernel fraction above
            ``kernel_fraction_bound``.
        gauge_weight: positive multiplier of the gradient gauge matrix; its
            independence is tested separately on physical circuit coefficients.
    """
    from ngsolve import GridFunction

    SparseSolvSolver = _sparsesolv_solver_class()
    rhs = f.vec if hasattr(f, "vec") else f
    if maxiter is None:
        maxiter = DEFAULT_MAXITER
    if residual_bound is None:
        residual_bound = RESIDUAL_BOUND_FACTOR * tol
    if gf is None:
        gf = GridFunction(fes)

    freedofs = fes.FreeDofs()
    free = _free_mask(freedofs, fes.ndof)
    b_orig = rhs.FV().NumPy()[free].copy()
    b = b_orig
    kernel_fraction = None
    if kernel is not None:
        G, kfree = kernel
        b = _project_out(b_orig, G[free][:, kfree])
        nb = float(np.linalg.norm(b_orig))
        kernel_fraction = float(np.linalg.norm(b_orig - b)) / nb if nb else 0.0

    solve_mat = a.mat
    gauge_null_defect = None
    gauge_scale = None
    if kernel is not None:
        if not math.isfinite(gauge_weight) or gauge_weight <= 0:
            raise ValueError("gauge_weight must be finite and positive")
        import scipy.sparse as sp
        from ngsolve.la import SparseMatrixdouble
        Gf = G[:, kfree].copy().tocsr()
        Gf = sp.diags(free.astype(float)) @ Gf
        Gf = Gf[:, np.asarray(Gf.getnnz(axis=0)).ravel() > 0]
        As = sp.csr_matrix(a.mat.CSR()).copy()
        Af = As[free][:, free]
        Qf = Gf[free]
        gauge_null_defect = float(sp.linalg.norm(Af @ Qf) / (sp.linalg.norm(Af) * sp.linalg.norm(Qf)))
        if not math.isfinite(gauge_null_defect) or gauge_null_defect > 1e-10:
            raise RuntimeError(f"Invalid gradient kernel [{label}]: {gauge_null_defect}")
        GG = (Gf @ Gf.T).tocsr()
        gauge_scale = gauge_weight * float(abs(Af.diagonal()).max() / GG.diagonal().max())
        gauged = (As + gauge_scale * GG + sp.diags((~free).astype(float))).tocoo()
        solve_mat = SparseMatrixdouble.CreateFromCOO(gauged.row.tolist(), gauged.col.tolist(), gauged.data.tolist(), fes.ndof, fes.ndof)

    if kernel_fraction is not None and kernel_fraction > kernel_fraction_bound:
        raise RuntimeError(f"Incompatible load [{label}]: kernel fraction {kernel_fraction:.3e} exceeds {kernel_fraction_bound:.3e}")
    solver = SparseSolvSolver(solve_mat, method="ICCG", freedofs=freedofs,
                              tol=tol, maxiter=int(maxiter), shift=shift,
                              save_best_result=True, printrates=False)
    solver.diagonal_scaling = bool(diagonal_scaling)
    solver.auto_shift = bool(auto_shift)

    # sparsesolv's internal stop test is not purely relative for small
    # right-hand sides (measured: with ||f|| << 1 it reports converged=True at
    # relative residuals up to 1e-2).  Normalising the free part of the rhs to
    # unit length makes its test a true relative test; the solution is scaled
    # back afterwards (the system is linear).
    bnorm = float(np.linalg.norm(b))
    rs = rhs.CreateVector()
    rs[:] = 0.0
    rs.FV().NumPy()[free] = b
    u = rhs.CreateVector()
    if bnorm == 0.0:
        u[:] = 0.0
        converged, solver_flag, iterations, solver_res = True, True, 0, 0.0
    else:
        rs.data *= 1.0 / bnorm
        u.data = solver * rs
        u.data *= bnorm
        rs.data *= bnorm
        res = solver.last_result
        # judged on the reported relative residual, not on the flag alone
        # (see module docstring)
        converged = float(res.final_residual) <= STALL_FACTOR * tol
        solver_flag = bool(res.converged)
        iterations, solver_res = int(res.iterations), float(res.final_residual)
    uv = u.FV().NumPy()
    uv[~free] = 0.0

    true_res = explicit_residual(a.mat, u, rs, free)
    info = {
        "label": label,
        "solver": "sparsesolv ICCG",
        "gauge_null_defect": gauge_null_defect,
        "gauge_scale": gauge_scale,
        "ndof_free": int(free.sum()),
        "shift": float(shift),
        "shift_used": float(solver.shift),
        "tol": float(tol),
        "maxiter": int(maxiter),
        "rhs_norm": bnorm,
        "rhs_kernel_fraction": kernel_fraction,
        "original_rhs_residual": float(explicit_residual(a.mat, u, rhs, free)),
        "gauge_weight": float(gauge_weight) if kernel is not None else None,
        "converged": converged,
        "solver_flag": solver_flag,
        "iterations": iterations,
        "solver_residual": solver_res,
        "explicit_residual": float(true_res),
        "residual_bound": float(residual_bound),
    }
    if not converged:
        raise RuntimeError(
            f"ICCG did not converge [{label}]: {iterations} iterations, "
            f"relative residual {solver_res:.3e} (tol {tol:.1e}, shift {shift})")
    if not math.isfinite(true_res) or true_res > residual_bound:
        raise RuntimeError(
            f"ICCG explicit residual check failed [{label}]: "
            f"||f - A u||/||f|| = {true_res:.3e} > bound {residual_bound:.1e}")
    if kernel_fraction is not None and kernel_fraction > kernel_fraction_bound:
        raise RuntimeError(
            f"rhs of singular system [{label}] has a kernel component of "
            f"{kernel_fraction:.3e} (> {kernel_fraction_bound:.1e}) of its norm")

    gv = gf.vec.FV().NumPy()
    gv[free] = uv[free]
    return gf, info


def solve_cg(mat, pre, rhs, sol, freedofs, tol=1e-12, maxiter=10000,
             residual_bound=None, label=""):
    """NGSolve preconditioned CG (``CGSolver``) with loud failure.

    ``sol`` is overwritten (zero initial guess, as in ``ngsolve.solvers.CG``).
    Returns the ``info`` dict; raises RuntimeError on non-convergence.
    """
    from ngsolve.krylovspace import CGSolver

    if residual_bound is None:
        residual_bound = max(RESIDUAL_BOUND_FACTOR * tol, CG_RESIDUAL_FLOOR)
    solver = CGSolver(mat=mat, pre=pre, tol=tol, maxiter=int(maxiter))
    solver.Solve(rhs=rhs, sol=sol, initialize=True)
    hist = list(solver.residuals)
    if not hist or hist[0] == 0.0:
        converged = True
        rel = 0.0
    else:
        rel = hist[-1] / hist[0]
        converged = rel <= tol
    free = _free_mask(freedofs, len(sol))
    true_res = explicit_residual(mat, sol, rhs, free)
    info = {
        "label": label,
        "solver": "ngsolve CGSolver",
        "ndof_free": int(free.sum()),
        "tol": float(tol),
        "maxiter": int(maxiter),
        "converged": bool(converged),
        "iterations": int(solver.iterations),
        "solver_residual": float(rel),
        "explicit_residual": float(true_res),
        "residual_bound": float(residual_bound),
    }
    if not converged:
        raise RuntimeError(
            f"CG did not converge [{label}]: {solver.iterations} iterations, "
            f"preconditioned residual ratio {rel:.3e} > tol {tol:.1e}")
    if not math.isfinite(true_res) or true_res > residual_bound:
        raise RuntimeError(
            f"CG explicit residual check failed [{label}]: "
            f"||f - A u||/||f|| = {true_res:.3e} > bound {residual_bound:.1e}")
    return info


def print_solve(info):
    print(f"  [{info['label']}] {info['solver']}: ndof={info['ndof_free']} "
          f"iterations={info['iterations']} "
          f"solver residual={info['solver_residual']:.3e} "
          f"residual={info['explicit_residual']:.3e}"
          + (f" rhs kernel fraction={info['rhs_kernel_fraction']:.3e}"
             if info.get("rhs_kernel_fraction") is not None else ""))


def rel_err(value, theory):
    return abs(value - theory) / abs(theory)


def write_json(path, data):
    """Write ``data`` as indented JSON (numpy scalars converted)."""
    import json

    def conv(x):
        if isinstance(x, dict):
            return {str(k): conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [conv(v) for v in x]
        if isinstance(x, np.generic):
            return x.item()
        return x

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(conv(data), fh, indent=2)
    print(f"wrote {path}")


def sample_fields(mesh, E0, B1, E2, radius, length):
    """Sample independent physical fields, not potential coordinates."""
    rho=np.linspace(.05,.95,25)
    points=[mesh(float(t*radius),0.0,length/2) for t in rho]
    return {"rho":rho.tolist(),
            "E0":[list(map(float,E0(pt))) for pt in points],
            "B1":[list(map(float,B1(pt))) for pt in points],
            "E2":[list(map(float,E2(pt))) for pt in points]}
