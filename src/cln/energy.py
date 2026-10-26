"""Material-normalized RC CLN with field/circuit quadratic-storage certificates.

K u + M u_dot = f F, y = f.T u. K and M are real symmetric positive definite.
For heat, the quadratic storage has units J K, not J; heat content is a separate
linear functional. See docs/physical_diffusion_cln.md for the coordinate map.
"""
from numbers import Integral

import numpy as np


class EnergyCertificationError(ValueError):
    """Measured field/circuit identities failed the requested tolerance."""

    def __init__(self, residuals):
        self.residuals = residuals
        super().__init__(f"CLN energy certificate failed (loss of orthogonality/normalization): {residuals}")


def _vector(value, name, size=None):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    value = np.asarray(value, dtype=float)
    if value.ndim != 1 or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite real vector")
    if size is not None and value.size != size:
        raise ValueError(f"{name} must have length {size}")
    return value


def _spd(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    a = np.asarray(value, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not a.size:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite")
    scale = np.max(np.abs(a))
    if np.max(np.abs(a - a.T)) > 1e-12*scale:
        raise ValueError(f"{name} must be symmetric; advection/nonreciprocal systems are unsupported")
    a = a/2 + a.T/2
    try:
        np.linalg.cholesky(a)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite; remove fixed DOFs and treat null modes explicitly") from exc
    return a


def _scaled_matrix_error(actual, expected):
    # Take square roots BEFORE forming the outer product: squaring a valid
    # diagonal such as 1e200 would overflow and could falsely report zero error.
    roots = np.sqrt(np.diag(expected))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        scale = np.outer(roots, roots)
        error = float(np.max(np.abs(actual - expected)/scale))
    return error if np.isfinite(error) else float("inf")


class EnergyCLN:
    """A capacitor-terminated physical RC ladder, with measured projection forms.

Node coordinates v reconstruct u=E v. Capacitor drops are w=B v, where B has
diagonal +1 and upper diagonal -1. The final capacitor connects to reference.
"""

    def __init__(self, R, C, projected_stiffness, projected_capacity,
                 projected_load, odd_capacity, *, basis=None, odd_basis=None,
                 storage_weights=None, audit_tolerance=1e-8):
        self.R = _vector(R, "R")
        self.C = _vector(C, "C", self.R.size)
        n = self.R.size
        if not n or np.any(self.R <= 0) or np.any(self.C <= 0):
            raise ValueError("R and C must be nonempty positive element arrays")
        if not np.isfinite(audit_tolerance) or not 0 < audit_tolerance <= 1e-4:
            raise ValueError("audit_tolerance must be positive and at most 1e-4")
        self.B = np.eye(n) - np.eye(n, k=1)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            self.G = np.diag(1/self.R)
            self.M = self.B.T @ np.diag(self.C) @ self.B
        if (not np.all(np.isfinite(self.G)) or not np.all(np.isfinite(self.M)) or
                np.any(np.diag(self.G) <= 0) or np.any(np.diag(self.M) <= 0)):
            raise ValueError("circuit matrices exceed the supported floating-point range")
        self.port = np.eye(n)[0]
        self.projected_stiffness = np.asarray(projected_stiffness, dtype=float)
        self.projected_capacity = np.asarray(projected_capacity, dtype=float)
        self.projected_load = np.asarray(projected_load, dtype=float)
        odd_capacity = np.asarray(odd_capacity, dtype=float)
        for name, actual, shape in (
            ("projected_stiffness", self.projected_stiffness, (n, n)),
            ("projected_capacity", self.projected_capacity, (n, n)),
            ("projected_load", self.projected_load, (n,)),
            ("odd_capacity", odd_capacity, (n, n)),
        ):
            if actual.shape != shape or not np.all(np.isfinite(actual)):
                raise ValueError(f"{name} has invalid shape or nonfinite values")
        errors = {
            "dissipation_orthogonality": _scaled_matrix_error(self.projected_stiffness, self.G),
            "capacity_orthogonality": _scaled_matrix_error(odd_capacity, np.diag(self.C)),
            "field_circuit_storage": _scaled_matrix_error(self.projected_capacity, self.M),
            "collocated_port": float(np.max(np.abs(self.projected_load - self.port))),
        }
        if max(errors.values()) > audit_tolerance:
            raise EnergyCertificationError(errors)
        self.certificate = {"passed": True, "tolerance": audit_tolerance, "residuals": errors,
                            "scope": "identities on the retained subspace, not full-model approximation accuracy"}
        self.basis = None if basis is None else np.asarray(basis, dtype=float)
        self.odd_basis = None if odd_basis is None else np.asarray(odd_basis, dtype=float)
        self.storage_weights = (None if storage_weights is None else
                                _vector(storage_weights, "storage_weights", n))
        self.field_modes = None
        self.flux_modes = None
        self.metadata = {}
        self._initial_projector = None

    def impedance(self, s):
        """Collocated impedance with all n resistors AND all n capacitors retained."""
        values = np.asarray(s, dtype=complex)
        if not np.all(np.isfinite(values)):
            raise ValueError("s must be finite")
        result = np.array([self.port @ np.linalg.solve(self.G + z*self.M, self.port)
                           for z in values.flat]).reshape(values.shape)
        return result[()] if result.ndim == 0 else result

    def to_lti(self, initial_field=None):
        """Export the full RC model; LTI states retain node-potential coordinates."""
        from .export import rc_lti
        model = rc_lti(self.R, self.C, termination="capacitance")
        if initial_field is not None:
            projection = self.project_initial(initial_field)
            model = model.with_initial_state(projection.x0)
            model.metadata["initial_projection"] = projection.to_dict()
        return model

    def project_initial(self, field):
        """Capacity-orthogonal projection; input uses the original free FE DOFs.

        Native FE models require a real GridFunction in their retained H1 space,
        satisfying the homogeneous essential boundary conditions. Temperatures
        are excess above the reference. Linear heat content need not be preserved.
        """
        if self._initial_projector is None:
            raise ValueError("initial-field metric was not retained; rebuild the energy model")
        return self._initial_projector(field)

    def reconstruct(self, nodes):
        """Return full coordinates (matrix backend) or a native field (FE backend)."""
        nodes = _vector(nodes, "nodes", self.R.size)
        if self.basis is not None:
            return self.basis @ nodes
        if self.field_modes is not None:
            from ngsolve import GridFunction
            field = GridFunction(self.field_modes[0].space)
            for v, mode in zip(nodes, self.field_modes):
                field.vec.data += float(v)*mode.vec
            return field
        raise ValueError("field modes were not retained")

    def balance(self, nodes, drive=0.0):
        """Evaluate storage, dissipation, dynamics and the collocated supply balance.

For heat: quadratic_storage is J K; linear_storage_content is J when physical
integral weights were supplied. Temperatures are excess above the reference.
"""
        v = _vector(nodes, "nodes", self.R.size)
        if not np.isfinite(drive) or np.iscomplexobj(drive):
            raise ValueError("drive must be finite and real")
        dv = np.linalg.solve(self.M, self.port*drive - self.G @ v)
        w = self.B @ v
        W = float(0.5*np.dot(self.C, w*w))
        D = float(np.dot(v/self.R, v))
        supply = float(drive*v[0])
        rate = float(v @ self.M @ dv)
        out = {"node_rates": dv.tolist(), "capacitor_drops": w.tolist(),
               "quadratic_storage": W, "dissipation": D, "supply": supply,
               "storage_rate": rate, "balance_residual": rate + D - supply,
               "field_quadratic_storage": float(0.5*v @ self.projected_capacity @ v),
               "field_dissipation": float(v @ self.projected_stiffness @ v),
               "field_supply": float(drive*self.projected_load @ v),
               "port_output": float(v[0])}
        # The rate-tested PDE gives a second identity. For magnetic diffusion,
        # this (rather than the capacity metric) can be the physical energy law.
        rate_D = float(dv @ self.M @ dv)
        rate_supply = float(drive*dv[0])
        stiffness_rate = float(v @ self.G @ dv)
        out["rate_tested_balance"] = {
            "stiffness_storage": D/2, "rate_dissipation": rate_D,
            "supply": rate_supply, "storage_rate": stiffness_rate,
            "balance_residual": stiffness_rate + rate_D - rate_supply}
        if self.storage_weights is not None:
            out["linear_storage_content"] = float(self.storage_weights @ v)
            out["linear_storage_rate"] = float(self.storage_weights @ dv)
        if self.basis is not None:
            out["reconstructed_field_coordinates"] = self.reconstruct(v).tolist()
        return out

    def reconstruct_flux(self, nodes):
        """Reconstruct native FE flux from resistor currents and flux modes."""
        v = _vector(nodes, "nodes", self.R.size)
        if self.flux_modes is None:
            raise ValueError("spatial flux modes require the native FE backend")
        terms = [float(current)*mode for current, mode in zip(v/self.R, self.flux_modes)]
        return sum(terms[1:], terms[0])

    def netlist(self):
        """SPICE RC one-port; drive node p with a flux/current source, reference 0."""
        lines = ["* Physical diffusion CLN: shunt R, series C, final C to reference",
                 "* Modal coordinates; not a conventional series-R/shunt-C thermal network"]
        names = ["p"] + [f"n{i}" for i in range(1, self.R.size)]
        for i, (r, c) in enumerate(zip(self.R, self.C)):
            tail = names[i+1] if i+1 < len(names) else "0"
            lines += [f"R{i} {names[i]} 0 {r:.17g}",
                      f"C{i} {names[i]} {tail} {c:.17g}"]
        return "\n".join(lines) + "\n"

    def to_dict(self):
        out = {"R": self.R.tolist(), "C": self.C.tolist(), "G": self.G.tolist(),
               "M": self.M.tolist(), "B": self.B.tolist(), "port": self.port.tolist(),
               "termination": "capacitor_to_reference", "certificate": self.certificate,
               "metadata": self.metadata, "netlist": self.netlist(),
               "equations": {"dynamics": "M dv/dt + G v = port F; y = port^T v",
                             "reconstruction": "u = E v = H (B v)",
                             "storage": "W = 1/2 v^T M v = 1/2 sum C_i (Bv)_i^2",
                             "dissipation": "D = v^T G v = sum v_i^2/R_i",
                             "balance": "dW/dt + D = F*y",
                             "rate_tested_balance": "d(1/2 v^T G v)/dt + vdot^T M vdot = F*ydot"}}
        if self.basis is not None:
            out["even_basis"] = self.basis.tolist()
            out["odd_basis"] = self.odd_basis.tolist()
        if self.storage_weights is not None:
            out["linear_storage_weights"] = self.storage_weights.tolist()
        return out


def _certified_prefix(R, C, kg, mg, port, mh, *, basis=None, odd_basis=None,
                      storage_weights=None, audit_tolerance=1e-8):
    """Keep the largest certified prefix, reporting any rejected extra stages."""
    rejected = {}
    for n in range(len(R), 0, -1):
        try:
            model = EnergyCLN(R[:n], C[:n], kg[:n, :n], mg[:n, :n], port[:n], mh[:n, :n],
                              basis=None if basis is None else basis[:, :n],
                              odd_basis=None if odd_basis is None else odd_basis[:, :n],
                              storage_weights=None if storage_weights is None else storage_weights[:n],
                              audit_tolerance=audit_tolerance)
        except EnergyCertificationError as exc:
            rejected[str(n)] = exc.residuals
            if n == 1:
                raise
        else:
            model.certificate["rejected_larger_prefixes"] = rejected
            return model


def physical_cln(stiffness, capacity, load, nstages=6, *, storage_integral=None,
                 audit_tolerance=1e-8):
    """Derive a SISO physical CLN from an assembled coercive diffusion system.

The load is used AS SUPPLIED. To interpret F as total injected heat, assemble
it from a continuum source of unit integral; summing free-DOF loads is not a
valid source normalization. storage_integral[j] = integral b*phi_j dOmega.
"""
    K, M = _spd(stiffness, "stiffness"), _spd(capacity, "capacity")
    if K.shape != M.shape:
        raise ValueError("stiffness and capacity must have the same shape")
    f = _vector(load, "load", K.shape[0])
    if not np.any(f):
        raise ValueError("load must be nonzero")
    if isinstance(nstages, bool) or not isinstance(nstages, Integral) or nstages < 1:
        raise ValueError("nstages must be a positive integer")
    weights = (None if storage_integral is None else
               _vector(storage_integral, "storage_integral", f.size))
    R, C, even, odd = [], [], [], []
    q, previous = f.copy(), np.zeros_like(f)
    qscale = np.linalg.norm(f)
    if not np.isfinite(qscale) or qscale == 0:
        raise ValueError("load magnitude is outside the supported floating-point range")
    reason = "requested_stages"
    for _ in range(min(nstages, f.size)):
        if np.linalg.norm(q) <= 1e-13*qscale:
            reason = "invariant_subspace_breakdown"
            break
        t = np.linalg.solve(K, q)
        r = float(t @ K @ t)
        if not np.isfinite(r) or r <= 0:
            raise ValueError("nonpositive/nonfinite CLN resistance: invalid or ill-conditioned system")
        e = t/r
        h = previous + e
        c = float(h @ M @ h)
        if not np.isfinite(r) or not np.isfinite(c) or r <= 0 or c <= 0:
            raise ValueError("nonpositive/nonfinite CLN element: invalid or ill-conditioned system")
        R.append(r); C.append(c); even.append(e); odd.append(h)
        update = M @ h/c
        q = q - update
        qscale = max(qscale, np.linalg.norm(update))
        previous = h
    if nstages > f.size and len(R) == f.size:
        reason = "full_dimension"
    E, H = np.column_stack(even), np.column_stack(odd)
    model = _certified_prefix(R, C, E.T @ K @ E, E.T @ M @ E, E.T @ f, H.T @ M @ H,
                             basis=E, odd_basis=H,
                             storage_weights=None if weights is None else E.T @ weights,
                             audit_tolerance=audit_tolerance)
    if model.R.size < len(R):
        reason = "orthogonality_limit"
    model.metadata = {"backend": "assembled_matrices", "requested_stages": int(nstages),
                      "effective_stages": int(model.R.size), "full_dimension": f.size,
                      "stop_reason": reason, "source_normalization": "as_supplied",
                      "stiffness_condition_estimate": float(np.linalg.cond(K)),
                      "capacity_condition_estimate": float(np.linalg.cond(M))}
    # Own the original metric: caller mutation must not change the projection.
    M = M.copy()
    from .initial import project_metric

    def project(field):
        u = _vector(field, "initial_field", f.size)
        E = model.basis
        result = project_metric(E.T @ M @ E, E.T @ M @ u, float(u @ M @ u),
                                lambda v: (u-E @ v) @ M @ (u-E @ v),
                                circuit_metric=model.M, tolerance=audit_tolerance)
        if weights is not None:
            result.diagnostics["full_linear_storage_content"] = float(weights @ u)
            result.diagnostics["retained_linear_storage_content"] = float(weights @ (E @ result.x0))
        return result

    model._initial_projector = project
    return model
