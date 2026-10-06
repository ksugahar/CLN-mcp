"""Energy-derived current-driven Type1 CLN for TEAM 28 Model A.

Axisymmetric fixed-height phasors, finite air boundary, peak coil current.
Elements come from magnetic field energy and electric Joule-loss norms.
The reference is independently solved on the same mesh, not measured force.
"""
import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import netgen
import ngsolve
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from netgen.occ import Glue, MoveTo, OCCGeometry, X
from ngsolve import (BilinearForm, ET, GridFunction, H1, Integrate, IntegrationRule, LinearForm,
                     Mesh, SetNumThreads, TaskManager, dx, grad, x)
from surface_hybrid.ladder_eval import z_type1

ROOT = Path(__file__).resolve().parents[1]
MU = 4e-7 * math.pi
SIGMA = 3.4e7
FREQUENCY = 50.0
CURRENT_PEAK = 20.0
MASS = .107
GRAVITY = 9.80665
REFERENCE_HEIGHT_MM = 11.3
REFERENCE_URL = 'https://www.compumag.org/jsite/images/stories/TEAM/problem28.pdf'
QUADRATURE_ORDER = 12
RULES = {ET.TRIG: IntegrationRule(ET.TRIG, QUADRATURE_ORDER)}


def assemble(height, maxh=.006, radius=.60, order=2):
    """w=A_theta/r is finite at the axis; A_theta=r*w is regular there."""
    circle = MoveTo(0, 0).Circle(radius).Face()
    left = MoveTo(-radius, -radius).Rectangle(radius, 2 * radius).Face()
    air = circle - left
    air.bc('air')
    air.edges.name = 'outer'
    air.edges.Min(X).name = 'axis'
    coil1 = MoveTo(.041-.028/2, -.052).Rectangle(.028, .052).Face()
    coil1.bc('coil1')
    coil2 = MoveTo(.0875-.015/2, -.052).Rectangle(.015, .052).Face()
    coil2.bc('coil2')
    disk = MoveTo(0, height).Rectangle(.065, .003).Face()
    disk.bc('disk')
    disk.maxh = .0015
    mesh = Mesh(OCCGeometry(Glue([air, coil1, coil2, disk]), dim=2).GenerateMesh(maxh=maxh))
    fes = H1(mesh, order=order, dirichlet='outer', complex=True)
    u, v = fes.TnT()
    sigma = mesh.MaterialCF({'disk': SIGMA}, default=0)
    # Unit series current, opposite winding sense. Actual excitation is 20 A peak.
    source = mesh.MaterialCF({'coil1': 960/(.028*.052), 'coil2': -576/(.015*.052)}, default=0)
    k = BilinearForm(fes)
    k += 2*math.pi/MU*x * (x*x*grad(u)[1]*grad(v)[1] + (2*u+x*grad(u)[0])*(2*v+x*grad(v)[0]))*dx(intrules=RULES)
    m = BilinearForm(fes)
    m += 2*math.pi*sigma*x**3 * u*v*dx(intrules=RULES)
    f = LinearForm(fes)
    f += 2*math.pi*source*x*x*v*dx(intrules=RULES)
    with TaskManager():
        k.Assemble(); m.Assemble(); f.Assemble()
    free = np.fromiter(fes.FreeDofs(), bool, fes.ndof)

    def matrix(form):
        a = sp.csr_matrix(form.mat.CSR()).copy()[free][:, free]
        assert sla.norm(a.imag) == 0
        a = a.real.tocsc()
        assert sla.norm(a-a.T)/sla.norm(a) < 1e-12
        return a

    return fes, free, sigma, matrix(k), matrix(m), f.vec.FV().NumPy()[free].real.copy()


def field_function(fes, free, values):
    """Fresh storage: no saved mode refers to a subsequently mutated field."""
    gf = GridFunction(fes)
    gf.vec.FV().NumPy()[free] = values
    return gf


def force(values, fes, free, sigma):
    phi = field_function(fes, free, values)
    br = -x*grad(phi)[1]
    current = -2j*math.pi*FREQUENCY*sigma*x*phi
    # e_theta cross e_r = -e_z; time average for PEAK phasors requires 1/2.
    density = -.5*(current.real*br.real + current.imag*br.imag)
    return float(Integrate(density*2*math.pi*x*dx(fes.mesh.Materials('disk'), intrules=RULES), fes.mesh))


def physical_cln(k, m, f, count):
    """Coupled magnetic/electric field recursion, without Arnoldi substitution."""
    factor = sla.splu(k)
    a = factor.solve(f)
    e = np.zeros_like(a)
    magnetic, electric, elements = [], [], []
    for _ in range(count):
        inductance = float(a@(k@a))
        assert np.isfinite(inductance) and inductance > 0
        e = e + a/inductance
        loss = float(e@(m@e))
        assert np.isfinite(loss) and loss > 0
        resistance = 1/loss
        magnetic.append(a.copy()); electric.append(e.copy())
        elements.extend([inductance, resistance])
        a = a - resistance*factor.solve(m@e)
    q = np.column_stack(magnetic); eq = np.column_stack(electric)
    kr = q.T@(k@q); mr = q.T@(m@q); er = eq.T@(m@eq)
    k_defect = float(np.max(abs(kr/np.sqrt(np.outer(np.diag(kr), np.diag(kr)))-np.eye(count))))
    m_defect = float(np.max(abs(er/np.sqrt(np.outer(np.diag(er), np.diag(er)))-np.eye(count))))
    assert max(k_defect, m_defect) < 1e-8
    return q, eq, elements, kr, mr, q.T@f, k_defect, m_defect


def solve_height(height_mm, count=6, maxh=.006, radius=.60, order=2):
    fes, free, sigma, k, m, f = assemble(height_mm/1000, maxh, radius, order)
    s = 2j*math.pi*FREQUENCY
    full = sla.spsolve(k+s*m, CURRENT_PEAK*f)
    residual = float(np.linalg.norm((k+s*m)@full-CURRENT_PEAK*f)/np.linalg.norm(CURRENT_PEAK*f))
    assert residual < 1e-9
    q, eq, elements, kr, mr, fr, kd, md = physical_cln(k, m, f, count)
    yr = np.linalg.solve(kr+s*mr, CURRENT_PEAK*fr)
    reduced = q@yr
    zfull = s*(f@full)/CURRENT_PEAK
    zreduced = s*(f@reduced)/CURRENT_PEAK
    circuit_defects = []
    for hz in [10., 25., 50., 100., 200.]:
        sj = 2j*math.pi*hz
        zfield = sj*(fr@np.linalg.solve(kr+sj*mr, fr))
        circuit_defects.append(float(abs(z_type1(elements, 0., sj)/zfield-1)))
    assert max(circuit_defects) < 1e-8
    volume_defects = []
    for j in range(count):
        a = field_function(fes, free, q[:, j]); e = field_function(fes, free, eq[:, j])
        l = float(Integrate((x*x*grad(a)[1].real**2+(2*a.real+x*grad(a)[0].real)**2)*2*math.pi/MU*x*dx(intrules=RULES), fes.mesh))
        loss = float(Integrate(sigma*e.real*e.real*2*math.pi*x**3*dx(intrules=RULES), fes.mesh))
        volume_defects.extend([abs(l/elements[2*j]-1), abs(loss*elements[2*j+1]-1)])
    assert max(volume_defects) < 1e-9
    joule = float(.5*abs(s)**2*np.vdot(full, m@full).real)
    power = .5*CURRENT_PEAK**2*zfull.real
    power_defect = float(abs(power/joule-1))
    assert power > 0 and power_defect < 1e-8
    full_force = force(full, fes, free, sigma)
    reduced_force = force(reduced, fes, free, sigma)
    assert abs(full_force-reduced_force) < .001
    return dict(height_mm=height_mm, ne=fes.mesh.ne, free_dofs=int(free.sum()),
                L_henry=elements[::2], R_ohm=elements[1::2], elements_LR=elements,
                full_upward_force_N=full_force, cln_upward_force_N=reduced_force,
                force_absolute_error_N=abs(full_force-reduced_force), full_relative_residual=residual,
                full_impedance_real=float(zfull.real), full_impedance_imag=float(zfull.imag),
                cln_impedance_real=float(zreduced.real), cln_impedance_imag=float(zreduced.imag),
                impedance_relative_error=float(abs(zreduced/zfull-1)),
                circuit_field_relative_defect=max(circuit_defects), magnetic_energy_orthogonality=kd,
                electric_loss_orthogonality=md, element_volume_integral_defect=max(volume_defects),
                full_joule_W=joule, full_input_loss_W=power, full_power_relative_defect=power_defect)


def equilibrium(rows, name):
    roots = []
    for a, b in zip(rows[:-1], rows[1:]):
        fa = a[name]-MASS*GRAVITY; fb = b[name]-MASS*GRAVITY
        if fa > 0 and fb < 0:  # downward force slope: stable static crossing
            roots.append(a['height_mm']+(b['height_mm']-a['height_mm'])*fa/(fa-fb))
    assert len(roots) == 1, roots
    return roots[0]


def run(output, count=6):
    paths = [Path(__file__), ROOT/'validation/surface_hybrid/ladder_eval.py']
    identity = lambda: {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_text(encoding='utf-8-sig').encode()).hexdigest() for p in paths}
    sources = identity(); SetNumThreads(4)
    out = dict(scope='TEAM 28 Model A, axisymmetric fixed-height 50 Hz; same-mesh reduction accuracy',
               method='Type1 physical magnetic/electric energy recursion, actual series-coil current, s0=0',
               stages=count, element_count=2*count, expansion_point=0., frequency_hz=FREQUENCY,
               peak_current_A=CURRENT_PEAK, mesh_maxh_m=.006, disk_maxh_m=.0015,
               air_radius_m=.60, FE_order=2, field_variable='w=A_theta/r, finite at axis; outer Dirichlet only', quadrature_order=QUADRATURE_ORDER, disk_mass_kg=MASS, gravity_m_s2=GRAVITY,
               measured_stationary_height_mm=REFERENCE_HEIGHT_MM, measured_reference_url=REFERENCE_URL,
               reference_scope='Only stationary height is an external measured reference. All force references are newly computed same-mesh FE.',
               force_convention='Upward, -0.5 Re[Jtheta conjugate(Br)], peak phasors, volume 2*pi*r dr dz',
               runtime=dict(python=platform.python_version(), ngsolve=ngsolve.__version__, netgen=netgen.__version__),
               source_sha256_lf=sources, complete=False, heights=[])
    for height in 3.8+np.arange(25):
        start=time.perf_counter(); row=solve_height(float(height), count)
        row['seconds']=time.perf_counter()-start; out['heights'].append(row)
        output.write_text(json.dumps(out, indent=2), encoding='utf-8')
        print(height, row['full_upward_force_N'], row['cln_upward_force_N'], row['force_absolute_error_N'], flush=True)
    full=equilibrium(out['heights'], 'full_upward_force_N'); cln=equilibrium(out['heights'], 'cln_upward_force_N')
    assert abs(full-REFERENCE_HEIGHT_MM) < .6 and abs(cln-REFERENCE_HEIGHT_MM) < .6
    assert identity() == sources, 'Source changed during sweep'
    # Independent settings probes quantify discrete-model sensitivity near equilibrium.
    probes = []
    for label, maxh, radius in [('finer_mesh', .004, .60), ('larger_air', .006, .90)]:
        rows = [solve_height(h, count, maxh=maxh, radius=radius) for h in [10.8, 11.8]]
        probes.append(dict(label=label, mesh_maxh_m=maxh, air_radius_m=radius,
                           heights=rows, full_equilibrium_height_mm=equilibrium(rows, 'full_upward_force_N')))
        print('settings probe', label, probes[-1]['full_equilibrium_height_mm'], flush=True)
    assert identity() == sources, 'Source changed during settings probes'
    out.update(settings_probes=probes, full_equilibrium_height_mm=full, cln_equilibrium_height_mm=cln,
               full_equilibrium_reference_error_mm=abs(full-REFERENCE_HEIGHT_MM),
               cln_equilibrium_reference_error_mm=abs(cln-REFERENCE_HEIGHT_MM), complete=True)
    output.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print('PASS same-mesh force <1mN, equilibrium <0.6mm, energy/circuit/power checks', full, cln)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output', required=True)
    p.add_argument('--stages', type=int, default=6); args=p.parse_args()
    if not 1 <= args.stages <= 8: p.error('stages must be 1..8; accuracy gates remain mandatory')
    run(Path(args.output), args.stages)
