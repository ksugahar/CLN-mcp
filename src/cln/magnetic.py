"""Field-state map for the storage-retaining RL admittance termination."""
import numpy as np
from ngsolve import CF, Integrate, GridFunction
from .energy import _vector, _scaled_matrix_error
from .export import rl_lti
from .initial import project_metric


class MagneticFieldCLN:
    """H-orthogonal initial projection onto an inductance-terminated CLN.

    Input is total port voltage [V], output current [A], states branch currents
    [A]. The field is H [A/m], not B, T alone, or an ungauged vector potential.
    The caller supplies physically admissible fields on the same conductor mesh.
    """

    def __init__(self, solver, resistance, inductance, *, audit_tolerance=1e-7):
        if not np.isfinite(audit_tolerance) or not 0 < audit_tolerance <= 1e-4:
            raise ValueError("audit_tolerance must be positive and at most 1e-4")
        self.solver = solver
        self.field_modes = list(solver.field_modes)
        self.current_modes = list(solver.current_modes)
        self._lti = rl_lti(resistance, inductance, s0=solver.s0,
                           length=solver.length, quantity='admittance')
        self.metric = np.diag([inductance[i]*solver.length for i in sorted(inductance)])
        self.gram = np.array([[self._mass(a, b) for b in self.field_modes]
                              for a in self.field_modes])
        self.dissipation = np.array([[Integrate(a*b/solver.sigma*solver._dx, solver.mesh)
                                       for b in self.current_modes] for a in self.current_modes])
        # The natural port is the axial current averaged over the segment.
        # Boundary interpolation can give a common non-unit normalization.
        self.port = np.array([Integrate(j[2]*solver._dx, solver.mesh)/solver.length
                              for j in self.current_modes])
        self.port_scale = float(np.mean(self.port))
        if not np.isfinite(self.port_scale) or self.port_scale <= 0:
            raise ValueError('invalid measured axial port normalization')
        port_error = float(np.max(np.abs(self.port/self.port_scale-1)))
        if port_error > audit_tolerance:
            raise ValueError('magnetic port modes lack a common normalization; refine the mesh')
        # An ideal transformer gives physical V/I without changing field states.
        self._lti.B *= self.port_scale
        self._lti.C *= self.port_scale
        factor = format(self.port_scale**2, '.17g')
        self._lti.expression = f'({factor}*{self._lti.expression})'
        self._lti.impedance_expression = f'({self._lti.impedance_expression}/{factor})'
        self._lti.metadata['physical_port_scale'] = self.port_scale
        expected_loss = -self.metric @ self._lti.A
        mass_error = _scaled_matrix_error(self.gram, self.metric)
        # Physical Joule loss includes the -s0*mu term when derived from a_s0.
        if np.any(np.diag(expected_loss) <= 0):
            raise ValueError('retained magnetic modes must have positive Joule dissipation')
        loss_scale = np.outer(np.sqrt(np.diag(expected_loss)), np.sqrt(np.diag(expected_loss)))
        with np.errstate(invalid='ignore', divide='ignore'):
            loss_error = float(np.max(np.abs(self.dissipation-expected_loss)/loss_scale))
        if (not np.isfinite(loss_error) or mass_error > audit_tolerance or
                loss_error > audit_tolerance):
            raise ValueError("magnetic field/circuit certificate failed; reduce stages or refine the mesh")
        try:
            np.linalg.cholesky(self.dissipation)
        except np.linalg.LinAlgError as exc:
            raise ValueError("retained magnetic modes must have positive Joule dissipation") from exc
        self.tolerance = audit_tolerance
        self.certificate = {'capacity_relative_error': mass_error,
                            'joule_loss_relative_error': loss_error,
                            'port_relative_error': port_error,
                            'measured_port_scale': self.port_scale,
                            'scope': 'retained H-mode energy, Joule loss and physical port; not full-field accuracy'}
        self._lti.metadata['state_coordinates'] = 'shunt inductor currents; H=sum(H_odd[i]*x[i]); mu-metric field projection retained'
        self._lti.metadata['field_certificate'] = self.certificate

    def _mass(self, a, b):
        s = self.solver
        return Integrate(s.mu*a*b*s._dx, s.mesh)

    def reconstruct(self, state):
        state = _vector(state, 'state', len(self.field_modes))
        return sum((float(x)*h for x, h in zip(state, self.field_modes)), CF((0, 0, 0)))

    def project_initial(self, field):
        """Project physical H; report discarded magnetic energy in Joules.

        CoefficientFunctions must belong to this mesh. No automatic gauge,
        boundary lifting, or conversion from B/magnetic potential is performed.
        """
        if isinstance(field, GridFunction) and field.space.mesh != self.solver.mesh:
            raise ValueError('initial H must use this model mesh')
        field = CF(field)
        if field.dim != 3 or field.is_complex:
            raise ValueError('initial H must be a real three-component field')
        result = project_metric(self.gram, [self._mass(h, field) for h in self.field_modes],
                                self._mass(field, field),
                                lambda x: self._mass(field-self.reconstruct(x), field-self.reconstruct(x)),
                                circuit_metric=self.metric, tolerance=self.tolerance)
        result.diagnostics['storage_units'] = 'J (magnetic energy)'
        return result

    def to_lti(self, initial_field=None):
        model = self._lti.with_initial_state(np.zeros(len(self.field_modes)))
        if initial_field is not None:
            p = self.project_initial(initial_field)
            model = model.with_initial_state(p.x0)
            model.metadata['initial_projection'] = p.to_dict()
        return model
