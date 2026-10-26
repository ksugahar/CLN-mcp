"""Frequency-domain reference impedance by a direct NGSolve solve (no model reduction).

For every Laplace variable s (real, or complex s = j omega) solve the same
conductor-only T problem that the CLN reduces,

    (1/sigma curl T, curl N) + s (mu T, N) = <drive, N>,

and return the impedance per unit length. With the same mesh, order and
boundary condition this is the exact reference the ladder converges to, so it
replaces external 2-D codes for geometries without an analytic solution.

Impedance (non-conjugated bilinear power, valid for complex s):
    Neumann   (E_t = 1 V/m on the lateral surface):  Z = length / a_s(T, T)
    Dirichlet (I = 1 A through n x T):               Z = a_s(T, T) / length
"""
import math
import numpy as np
from ngsolve import (HCurl, BilinearForm, LinearForm, GridFunction, Integrate, CF, Cross,
                     curl, BND, specialcf, TaskManager)
from ._integration import electromagnetic_measures


def reference_impedance(mesh, sigma, mu, radius, length, s, bc="Neumann",
                        lateral="conductorBND", order=3, inverse="sparsecholesky", integration_order=None):
    """Z(s) [Ohm/m] of the conductor at one Laplace variable s (s != 0)."""
    if bc not in ("Neumann", "Dirichlet"):
        raise ValueError("bc must be 'Neumann' or 'Dirichlet'")
    s = complex(s)
    if s == 0:
        raise ValueError("s = 0 is the static limit; use the CLN element R0 instead")
    cplx = s.imag != 0.0
    sv = s if cplx else s.real
    volume, surface, _ = electromagnetic_measures(order, integration_order)
    kw = dict(dirichlet=lateral) if bc == "Dirichlet" else {}
    fes = HCurl(mesh, order=order, type1=True, complex=cplx, **kw)
    T, N = fes.TnT()
    a = BilinearForm(fes, symmetric=True)
    a += (1/sigma*curl(T)*curl(N) + sv*mu*T*N)*volume
    f = LinearForm(fes)
    n = specialcf.normal(3)
    gf = GridFunction(fes)
    if bc == "Dirichlet":
        gf.Set(1/(2*math.pi*radius)*Cross(CF((0, 0, 1)), n), BND,
               definedon=mesh.Boundaries(lateral), dual=True)
    else:
        f += -(Cross(CF((0, 0, 1)), N.Trace())*n)*surface(lateral)
    with TaskManager():
        a.Assemble()
        f.Assemble()
    r = f.vec.CreateVector()
    r.data = f.vec - a.mat*gf.vec
    gf.vec.data += a.mat.Inverse(fes.FreeDofs(), inverse=inverse)*r
    p = (Integrate(1/sigma*curl(gf)*curl(gf)*volume, mesh)
         + sv*Integrate(mu*gf*gf*volume, mesh))
    z = length/p if bc == "Neumann" else p/length
    return complex(z) if cplx else float(np.real(z))


def reference_sweep(mesh, sigma, mu, radius, length, freqs, bc="Neumann",
                    jomega=True, **kw):
    """Z at s = j 2 pi f (jomega=True) or s = 2 pi f (jomega=False) for each f in freqs."""
    out = []
    for fq in freqs:
        s = 2j*math.pi*fq if jomega else 2*math.pi*fq
        out.append(reference_impedance(mesh, sigma, mu, radius, length, s, bc=bc, **kw))
    return np.array(out)
