"""Consistent volume/boundary metrics for curved 3D electromagnetic elements."""
from numbers import Integral
from ngsolve import ET, IntegrationRule, dx, ds


def electromagnetic_measures(order, integration_order=None):
    degree = 2*order+6 if integration_order is None else integration_order
    if isinstance(degree, bool) or not isinstance(degree, Integral) or degree < 1:
        raise ValueError("integration_order must be a positive integer")
    volume = {et: IntegrationRule(et, order=int(degree)) for et in (ET.TET, ET.PRISM, ET.PYRAMID, ET.HEX)}
    surface = {et: IntegrationRule(et, order=int(degree)) for et in (ET.TRIG, ET.QUAD)}
    return dx(intrules=volume), ds(intrules=surface), int(degree)
