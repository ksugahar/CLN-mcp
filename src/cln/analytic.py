"""Analytic references for an infinitely long round wire (internal impedance per unit length)."""
import math
import numpy as np
from scipy.special import jve


def wire_internal_impedance(s, radius, sigma, mu):
    """Z(s) = k/(2 pi a sigma) J0(ka)/J1(ka),  k^2 = -s mu sigma  [Ohm/m]."""
    k = np.sqrt(-np.asarray(s, dtype=complex)*mu*sigma)
    x = k*radius
    small = np.abs(x) < 1e-3
    # The series removes the removable singularity at DC; scaled Bessel
    # functions cancel their common exponential factor at large arguments.
    safe_x = np.where(small, 1.0 + 0j, x)
    ratio = safe_x*jve(0, safe_x)/(2*jve(1, safe_x))
    x2 = np.where(small, x, 0.0)**2
    ratio = np.where(small, 1 - x2/8 - x2**2/192, ratio)
    result = ratio/(np.pi*radius**2*sigma)
    return result[()] if result.ndim == 0 else result


def wire_cln_s0_zero(n, radius, sigma, mu):
    """Exact Cauer elements of the wire for s0 = 0:
    R_{2k} = (2k+1)/(pi a^2 sigma),   L_{2k+1} = mu/(8 pi (k+1))."""
    if n % 2 == 0:
        return (n + 1)/(math.pi*radius**2*sigma)
    return mu/(4*math.pi*(n + 1))
