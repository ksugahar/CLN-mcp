"""Anderson-accelerated fixed-point iteration x = G(x) that fails loudly when it does not converge."""
import numpy as np


class FixedPointNotConverged(RuntimeError):
    pass


def anderson(Gfun, x0, m=6, maxit=400, tol=1e-8):
    """Return (x, iterations) with ||G(x) - x|| <= tol ||G(x)||; raise FixedPointNotConverged after maxit."""
    X_, F_ = [], []
    x = np.array(x0, dtype=float, copy=True)
    res = np.inf
    for k in range(maxit):
        gx = Gfun(x); f = gx - x
        res = np.linalg.norm(f)/max(np.linalg.norm(gx), 1e-300)
        if not np.isfinite(res):
            raise FixedPointNotConverged(f"non-finite residual at iteration {k + 1}")
        if res <= tol:
            return gx, k + 1
        X_.append(gx); F_.append(f)
        if len(F_) > m + 1:
            X_.pop(0); F_.pop(0)
        if len(F_) == 1:
            x = gx
        else:
            dF = np.column_stack([F_[i+1] - F_[i] for i in range(len(F_)-1)])
            dX = np.column_stack([X_[i+1] - X_[i] for i in range(len(X_)-1)])
            gam = np.linalg.lstsq(dF, f, rcond=None)[0]
            x = gx - dX @ gam
    raise FixedPointNotConverged(f"no convergence in {maxit} iterations (relative residual {res:.3e} > {tol:.1e})")
