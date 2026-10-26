"""Shifted elements against the analytic circular-wire Bessel impedance."""
import math
import pytest
from cln import TOmegaCLN, cylinder_mesh


def test_positive_shift_elements_against_bessel_taylor():
    # Independent oracle: Taylor-expand k*I0(k*a)/(2*pi*a*sigma*I1(k*a))
    # at s0 (k=sqrt(s*mu*sigma)), then alternately extract its constant term
    # and the leading pole of the reciprocal remainder (80-digit arithmetic).
    # Higher ladder indices amplify FE errors: test only indices 0..4 here.
    exact = {0: .01501404993, 1: 1.118475497e-8, 2: .02851732268,
             3: 1.112470798e-8, 4: .02878546397}
    radius = length = .01
    mesh = cylinder_mesh(radius, length, maxh=radius/2, maxh_conductor=radius/10)
    model = TOmegaCLN(mesh, 1e6, 4e-7*math.pi, radius, length,
                     s0=2*math.pi*1e5, order=3, bc="Neumann", solver="direct")
    resistance, inductance, _ = model.run(nmax=4)
    elements = {**resistance, **inductance}
    for index, value in exact.items():
        assert elements[index] == pytest.approx(value, rel=5e-3, abs=0), index
