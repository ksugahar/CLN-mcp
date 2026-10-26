"""Material-metric projection of initial fields onto retained CLN coordinates."""
from dataclasses import dataclass
import numpy as np


@dataclass
class InitialProjection:
    """Projection diagnostics; residual storage is not a time-domain error bound."""
    x0: np.ndarray
    diagnostics: dict

    def to_dict(self):
        return {"initial_state": self.x0.tolist(), **self.diagnostics}


def project_metric(gram, moments, full_norm, residual_norm, *, circuit_metric,
                   tolerance=1e-8):
    """Solve the measured Gram system; integrate the residual without cancellation."""
    gram = np.asarray(gram, dtype=float)
    moments = np.asarray(moments, dtype=float)
    if (gram.ndim != 2 or gram.shape[0] != gram.shape[1] or not gram.size or
            moments.shape != (gram.shape[0],) or
            not np.all(np.isfinite(gram)) or not np.all(np.isfinite(moments)) or
            not np.isfinite(full_norm) or full_norm < 0):
        raise ValueError("initial field metric and moments must be finite and real")
    roots = np.sqrt(np.diag(gram))
    if not np.all(np.isfinite(roots)) or np.any(roots <= 0):
        raise ValueError("initial field Gram matrix must be positive definite")
    scaled = gram / np.outer(roots, roots)
    if not np.allclose(scaled, scaled.T, rtol=0, atol=1e-12):
        raise ValueError("initial field Gram matrix must be symmetric")
    try:
        chol = np.linalg.cholesky(scaled)
        x0 = np.linalg.solve(chol.T, np.linalg.solve(chol, moments/roots))/roots
    except np.linalg.LinAlgError as exc:
        raise ValueError("initial field basis is singular") from exc
    remainder = float(residual_norm(x0))
    retained = float(x0 @ gram @ x0)
    circuit = float(x0 @ circuit_metric @ x0)
    scale = max(full_norm, retained, circuit, np.finfo(float).tiny)
    pythagoras = abs(full_norm-retained-remainder)/scale
    metric_error = abs(retained-circuit)/scale
    values = np.array([remainder, retained, circuit, pythagoras, metric_error])
    if (not np.all(np.isfinite(x0)) or not np.all(np.isfinite(values)) or
            np.any(values[:3] < 0) or max(pythagoras, metric_error) > tolerance):
        raise ValueError("initial field projection failed its storage certificate")
    return InitialProjection(x0, {
        "full_quadratic_storage": full_norm/2,
        "retained_quadratic_storage": retained/2,
        "residual_quadratic_storage": remainder/2,
        "circuit_quadratic_storage": circuit/2,
        "relative_field_error": float(np.sqrt(remainder/full_norm)) if full_norm else 0.,
        "pythagorean_relative_residual": pythagoras,
        "field_circuit_relative_residual": metric_error,
        "scope": "material-metric best approximation at t=0; not a future output-error bound",
    })
