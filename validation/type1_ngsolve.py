"""Type1 wire: two independent weighted finite-element formulations.

Dimensionless meridian rectangle (r,axial) in [0,1]^2, homogeneous wire.
A uses axial potential; T uses H_theta=r*v to avoid division by r.
Compare four field-energy elements with independently differentiated Bessel Z.
No three-dimensional A-Phi/T-Omega/A-T claim; no private solver required.
"""
import argparse
import json
import math
from pathlib import Path
import mpmath as mp
import numpy as np
import ngsolve as ng
from netgen.geom2d import SplineGeometry


def oracle(z):
    with mp.workdps(70):
        z = mp.mpf(str(z))
        def g(s):
            return 2*mp.pi*mp.sqrt(s)*mp.besseli(1,mp.sqrt(s))/mp.besseli(0,mp.sqrt(s))
        c=mp.taylor(g,z,3)
        g0,g1=c[:2]
        g2=-g1*g1/c[2]
        g3=c[3]*g2*g2/(g1*g1)-g1
        return np.array([float(1/v) for v in (g0,g1,g2,g3)])


def solve_case(z,maxh,order):
    geo=SplineGeometry()
    geo.AddRectangle((0,0),(1,1),bcs=['bottom','outer','top','axis'])
    mesh=ng.Mesh(geo.GenerateMesh(maxh=maxh))
    dvol=ng.dx(bonus_intorder=2*order+4)
    residuals=[]
    def integral(f): return ng.Integrate(2*math.pi*ng.x*f*dvol,mesh)
    def field_energy(e,h): return integral(e*e+z*h*h)
    def solve(fes,a,f,lift=None):
        a.Assemble(); f.Assemble()
        gf=ng.GridFunction(fes)
        if lift is not None:
            gf.Set(lift,definedon=mesh.Boundaries('outer'))
        rhs=f.vec.CreateVector(); rhs.data=f.vec-a.mat*gf.vec
        gf.vec.data += a.mat.Inverse(fes.FreeDofs(),inverse='sparsecholesky')*rhs
        defect=rhs.CreateVector(); defect.data=f.vec-a.mat*gf.vec
        free=np.array(list(fes.FreeDofs()),bool)
        scale=max(np.linalg.norm(rhs.FV().NumPy()[free]),1e-30)
        rel=float(np.linalg.norm(defect.FV().NumPy()[free])/scale)
        if not np.isfinite(rel) or rel>1e-9:
            raise RuntimeError(f'true free-DOF residual {rel}')
        residuals.append(rel)
        return gf
    out={}
    for name in ('A','T'):
        fes=ng.H1(mesh,order=order,dirichlet='outer' if name=='A' else '')
        u,v=fes.TnT()
        a=ng.BilinearForm(fes,symmetric=True)
        f=ng.LinearForm(fes)
        if name=='A':
            a += ng.x*(ng.grad(u)*ng.grad(v)+z*u*v)*dvol
            ef=solve(fes,a,f,1)
            e,h=ef,ng.grad(ef)[0]/z
        else:
            def divergence(g): return 2*g+ng.x*ng.grad(g)[0]
            a += (ng.x*divergence(u)*divergence(v)+ng.x**3*ng.grad(u)[1]*ng.grad(v)[1]+z*ng.x**3*u*v)*dvol
            f += ng.x**2*v*ng.ds('outer')
            tf=solve(fes,a,f)
            e,h=divergence(tf),ng.x*tf
        adm=field_energy(e,h)
        lm=field_energy(e/adm,h/adm)/z
        r0=1/integral(e*e)
        fc=ng.LinearForm(fes)
        if name=='A':
            fc += ng.x*r0*e*v*dvol
            uc=solve(fes,a,fc)
            e1,h1=-z*uc,-ng.grad(uc)[0]
        else:
            fc += ng.x*r0*e*divergence(v)*dvol
            tc=solve(fes,a,fc)
            e1,h1=divergence(tc)-r0*e,ng.x*tc
        l1=field_energy(e1,h1)/z
        e2=e+e1/(z*l1)
        r2=1/integral(e2*e2)
        elements=np.array([lm,r0,l1,r2])
        if not np.all(np.isfinite(elements)) or not np.all(elements>0):
            raise RuntimeError('nonpositive or nonfinite elements')
        with mp.workdps(40):
            zm=mp.mpf(str(z))
            def ee(rr,ss): return mp.besseli(0,mp.sqrt(ss)*rr)/mp.besseli(0,mp.sqrt(ss))
            def hh(rr,ss): return mp.besseli(1,mp.sqrt(ss)*rr)/(mp.sqrt(ss)*mp.besseli(0,mp.sqrt(ss)))
            def yy(ss): return 2*mp.pi*hh(1,ss)
            exact_r0=1/(yy(zm)+zm*mp.diff(yy,zm))
            samples=np.linspace(0.02,0.98,25)
            exact_fields=np.array([[float(exact_r0*zm*mp.diff(lambda ss:ee(rr,ss),zm)),
                float(exact_r0*(hh(rr,zm)+zm*mp.diff(lambda ss:hh(rr,ss),zm)))] for rr in samples])
        measured=np.array([[e1(mesh(float(rr),0.5)),h1(mesh(float(rr),0.5))] for rr in samples])
        profile_error=(np.linalg.norm(measured-exact_fields,axis=0)/np.linalg.norm(exact_fields,axis=0)).tolist()
        out[name]={'elements':elements.tolist(),'relative_errors':(elements/oracle(z)-1).tolist(),
            'E1_H1_profile_relative_RMS':profile_error}
    out.update(maxh=maxh,order=order,ne=mesh.ne,max_true_residual=max(residuals))
    return out


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True)
    args=parser.parse_args()
    ng.SetNumThreads(2)
    z=8*math.pi**2
    with ng.TaskManager():
        cases=[solve_case(z,h,3) for h in (0.3,0.15,0.075,0.05)]
    for name in ('A','T'):
        errors=[max(abs(e) for e in c[name]['relative_errors']) for c in cases]
        if not errors[-1]<2e-5 or not errors[-1]<errors[0]/8:
            raise RuntimeError(f'{name} convergence gate failed: {errors}')
        profile=cases[-1][name]['E1_H1_profile_relative_RMS']
        if not np.all(np.isfinite(profile)) or max(profile)>2e-3:
            raise RuntimeError(f'{name} field profile failed')
    disagreement=max(abs(a/b-1) for a,b in zip(cases[-1]['A']['elements'],cases[-1]['T']['elements']))
    if disagreement>4e-5: raise RuntimeError(f'A/T mismatch {disagreement}')
    result={'ngsolve_version':ng.__version__,'scope':'Type1 continuum wire meridian FE, four elements, real z=8pi^2','z':z,'oracle':oracle(z).tolist(),'cases':cases,'fine_A_T_relative_difference':disagreement,'passed':True}
    Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    for c in cases:
        print(f"h={c['maxh']}: A error={max(map(abs,c['A']['relative_errors'])):.3e}, T error={max(map(abs,c['T']['relative_errors'])):.3e}, residual={c['max_true_residual']:.3e}")
    print(f'PASS A/T fine-mesh relative difference {disagreement:.3e}')

if __name__=='__main__': main()


