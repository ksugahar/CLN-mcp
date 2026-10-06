"""Physical terminal-port Type2 CLN on a notched 3D conductor.

The magnetic boundary n x A=0 encloses the conductor: no exterior or return
path is modelled. Positive-shift resistor coefficients include s0 times
magnetic energy. High-frequency comparisons may be under-resolved FE data.
"""
import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

SIGMA = 1e6
MU = 4e-7 * math.pi
HEIGHT = .015
TAU = MU * SIGMA * .01**2
FREQUENCIES = [1e3, 1e4, 1e5, 1e6]


def csr(form):
    return sp.csr_matrix(form.mat.CSR()).copy()


def packed(mat):
    a = sp.coo_matrix(mat)
    return dict(shape=list(a.shape), row=a.row.tolist(), col=a.col.tolist(), data=a.data.tolist())


def complex_pair(value):
    return [float(np.real(value)), float(np.imag(value))]


def source_identity():
    source = Path(__file__)
    return {"validation/cln3d/"+p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [source,source.with_name("general_shape_t.py")]}


class Model:
    def __init__(self, maxh, order, bonus=6):
        import ngsolve as ng
        from netgen.occ import Box, OCCGeometry, Z
        self.ng = ng
        shape = Box((-.01, -.01, 0), (.01, .01, HEIGHT)) - Box(
            (.002, .002, .0075), (.012, .012, .017))
        shape.faces.name = "surface"
        shape.faces.Min(Z).name = "in"
        shape.faces.Max(Z).name = "out"
        shape.mat("conductor")
        self.mesh = ng.Mesh(OCCGeometry(shape).GenerateMesh(maxh=maxh))
        self.va = ng.HCurl(self.mesh, order=order, nograds=False, dirichlet="surface|in|out")
        self.vp = ng.H1(self.mesh, order=order+1, dirichlet="in|out")
        self.fa = np.array(list(self.va.FreeDofs()), bool)
        self.fp = np.array(list(self.vp.FreeDofs()), bool)
        self.dx = ng.dx(bonus_intorder=bonus)
        a, w = self.va.TnT()
        p, q = self.vp.TnT()
        sf = ng.BilinearForm(self.vp)
        sf += SIGMA * ng.grad(p) * ng.grad(q) * self.dx
        sf.Assemble()
        self.S = csr(sf)[self.fp][:, self.fp].tocsc()
        self.slu = sla.splu(self.S)
        self.phi0 = ng.GridFunction(self.vp)
        self.phi0.Set(1, definedon=self.mesh.Boundaries("in"))
        lift = self.phi0.vec.CreateVector()
        lift.data = sf.mat * self.phi0.vec
        self.phi0.vec.FV().NumPy()[self.fp] = self.slu.solve(-lift.FV().NumPy()[self.fp])
        self.e0 = -ng.grad(self.phi0)
        self.G0 = float(ng.Integrate(SIGMA*self.e0*self.e0*self.dx, self.mesh))
        self.Kform = ng.BilinearForm(self.va)
        self.Kform += ng.curl(a)*ng.curl(w)/MU*self.dx
        self.Kform.Assemble()
        self.K = csr(self.Kform)[self.fa][:, self.fa].tocsc()
        mf = ng.BilinearForm(self.va)
        mf += SIGMA*a*w*self.dx
        mf.Assemble()
        self.M = csr(mf)[self.fa][:, self.fa].tocsc()
        lf = ng.LinearForm(self.va)
        lf += SIGMA*self.e0*w*self.dx
        lf.Assemble()
        self.f = lf.vec.FV().NumPy()[self.fa].copy()
        mixed = self.va*self.vp
        (a, p), (w, q) = mixed.TnT()
        cf = ng.BilinearForm(mixed)
        cf += SIGMA*a*ng.grad(q)*self.dx
        cf.Assemble()
        self.C = csr(cf)[self.va.ndof:, :self.va.ndof][self.fp][:, self.fa].tocsc()
        gm, h1 = self.va.CreateGradient()
        gh = np.array(list(h1.FreeDofs()), bool)
        self.G = sp.csr_matrix(gm.CSR()).copy()[self.fa][:, gh].tocsc()
        gg = self.G@self.G.T
        self.gauge_scale = float(abs(self.K.diagonal()).max()/gg.diagonal().max())
        self.Kg = self.K+self.gauge_scale*gg
        self.klu = sla.splu(self.Kg.tocsc())
        self.kernel_defect = float(sla.norm(self.K@self.G)/(sla.norm(self.K)*sla.norm(self.G)))
        assert self.kernel_defect < 1e-10
        self.na, self.np = len(self.f), self.S.shape[0]
        self.residuals = []
        self.kernel_fractions = []
        self.response_lus = {}
        self.metadata = dict(maxh=maxh, order=order, bonus_intorder=bonus, ne=self.mesh.ne,
                             free_A=self.na, free_phi=self.np, sigma=SIGMA, mu=MU,
                             volume=float(ng.Integrate(1, self.mesh)),
                             boundary_area={name: float(ng.Integrate(1, self.mesh,
                                 definedon=self.mesh.Boundaries(name))) for name in ["in", "out", "surface"]})
        self.geometry_arrays = dict(points=[list(point.p) for point in self.mesh.ngmesh.Points()],
                                    tetrahedra=[[v.nr for v in el.vertices] for el in self.mesh.ngmesh.Elements3D()])
        self.metadata["mesh_sha256"] = hashlib.sha256(json.dumps(self.geometry_arrays,sort_keys=True).encode()).hexdigest()
        assert abs(self.metadata["volume"]-5.52e-6) < 1e-12
        self.metadata["phi_departure_l2_relative"] = float(np.sqrt(ng.Integrate(
            (self.phi0-(1-ng.z/HEIGHT))**2*self.dx, self.mesh)/ng.Integrate(self.phi0**2*self.dx, self.mesh)))
        self.metadata["dc_transverse_current_fraction"] = float(ng.Integrate(
            (self.e0[0]**2+self.e0[1]**2)*self.dx, self.mesh)/ng.Integrate(self.e0*self.e0*self.dx, self.mesh))
        assert self.metadata["phi_departure_l2_relative"] > 1e-3
        assert self.metadata["dc_transverse_current_fraction"] > 1e-3

    def gf_a(self, a):
        gf = self.ng.GridFunction(self.va)
        gf.vec.FV().NumPy()[self.fa] = np.real(a)
        return gf

    def gf_p(self, p):
        gf = self.ng.GridFunction(self.vp)
        gf.vec.FV().NumPy()[self.fp] = np.real(p)
        return gf

    def split(self, e):
        return e[0], e[1:1+self.na], e[1+self.na:]

    def field(self, e):
        alpha, a, p = self.split(e)
        return alpha*self.e0+self.gf_a(a)-self.ng.grad(self.gf_p(p))

    def load(self, e):
        alpha, a, p = self.split(e)
        return alpha*self.f+self.M@a-self.C.T@p

    def r(self, e, v):
        alpha, a, p = self.split(e)
        beta, b, q = self.split(v)
        return alpha*beta*self.G0+alpha*(self.f@b)+beta*(self.f@a)+a@(self.M@b)-a@(self.C.T@q)-p@(self.C@b)+p@(self.S@q)

    def port(self, e):
        alpha, a, _ = self.split(e)
        return alpha*self.G0+self.f@a

    def magnetic(self, e):
        rhs = self.load(e)
        # Projection is a recorded load change; reject incompatible loads.
        gram = (self.G.T@self.G).tocsc()
        projected = rhs-self.G@sla.spsolve(gram, self.G.T@rhs)
        fraction = float(np.linalg.norm(projected-rhs)/np.linalg.norm(rhs))
        self.kernel_fractions.append(fraction)
        assert fraction <= 1e-8
        a = self.klu.solve(projected)
        residual = float(np.linalg.norm(self.K@a-rhs)/np.linalg.norm(rhs))
        self.residuals.append(residual)
        assert residual <= 1e-8
        return a

    def response(self, e, s, gauge_weight=1):
        alpha, v, p = self.split(e)
        rhs_a = self.load(e)
        rhs_p = self.C@v-self.S@p
        if s == 0:
            chi = self.slu.solve(rhs_p)
            out = e.copy()
            out[1+self.na:] += chi
            return out, self.magnetic(out)
        key = (complex(s), gauge_weight)
        if key not in self.response_lus:
            kg = self.K+gauge_weight*self.gauge_scale*(self.G@self.G.T)
            block = sp.bmat([[kg+s*self.M, self.C.T], [self.C, self.S/s]], format="csc")
            scale = 1/np.sqrt(abs(block.diagonal()))
            d = sp.diags(scale)
            self.response_lus[key] = (sla.splu((d@block@d).tocsc()), scale)
        lu, scale = self.response_lus[key]
        solution = scale*lu.solve(scale*np.r_[rhs_a, rhs_p/s])
        a, chi = solution[:self.na], solution[self.na:]
        out = e.astype(solution.dtype).copy()
        out[1:1+self.na] -= s*a
        out[1+self.na:] += chi
        r_a = (self.K+s*self.M)@a+self.C.T@chi-rhs_a
        r_p = s*(self.C@a)+self.S@chi-rhs_p
        res = max(np.linalg.norm(r_a)/max(np.linalg.norm(rhs_a), 1e-300),
                  np.linalg.norm(r_p)/max(np.linalg.norm(s*(self.C@a)), np.linalg.norm(rhs_p), 1e-300))
        self.residuals.append(float(res))
        assert res <= 1e-8, res
        return out, a

    def energy_integrals(self, e, a):
        ef = self.field(e)
        bf = self.ng.curl(self.gf_a(a))
        return [float(self.ng.Integrate(SIGMA*ef*ef*self.dx, self.mesh)),
                float(self.ng.Integrate(bf*bf/MU*self.dx, self.mesh))]

    def terminal_reactions(self,e):
        reactions = {}
        for part in ["real","imag"]:
            ef = self.field(getattr(e,part))
            lf = self.ng.LinearForm(self.vp)
            lf += SIGMA*ef*self.ng.grad(self.vp.TestFunction())*self.dx
            lf.Assemble()
            for name in ["in","out"]:
                lift = self.ng.GridFunction(self.vp)
                lift.Set(1,definedon=self.mesh.Boundaries(name))
                val = float(lift.vec.FV().NumPy()@lf.vec.FV().NumPy())
                reactions[name] = reactions.get(name,0)+val*(1 if part=="real" else 1j)
        current = self.port(e)
        defect = max(abs(reactions["in"]+current),abs(reactions["out"]-current))/abs(current)
        assert defect < 1e-8
        return dict(in_entering=complex_pair(-reactions["in"]),out_leaving=complex_pair(reactions["out"]),relative_defect=float(defect))

    def export(self):
        return dict(K=packed(self.K), M=packed(self.M), S=packed(self.S), C=packed(self.C),
                    G=packed(self.G), f=self.f.tolist(), G0=self.G0, gauge_scale=self.gauge_scale,
                    phi0=self.phi0.vec.FV().NumPy().tolist(), free_A=self.fa.tolist(),
                    free_phi=self.fp.tolist(), metadata=self.metadata,
                    geometry=self.geometry_arrays,
                    electric_representation="E=alpha*e0+N*v-grad(q)*psi; vector=[alpha,v,psi]",
                    units="SI; terminal lift 1 V; no length-squared normalization")


def ladder(R, L, s, s0):
    q = s-s0
    z = q*L[-1]
    for k in range(len(R)-1, -1, -1):
        z = R[k]+z
        if k:
            z = q*L[k-1]*z/(q*L[k-1]+z)
    return z


def inverse_series(y):
    z = [1/y[0]]
    for k in range(1, len(y)):
        z.append(-sum(y[j]*z[k-j] for j in range(1, k+1))/y[0])
    return np.array(z)


def moments(model, s0, rp, lp, bp, count):
    # Coefficients in t=(s-s0)*TAU; derivatives by sparse full mixed solves.
    base = np.zeros(1+model.na+model.np)
    base[0] = 1
    e, a = model.response(base, s0)
    full_y = [model.port(e)]
    aa = a.copy()
    for _ in range(1, count):
        forcing = np.zeros_like(base)
        forcing[1:1+model.na] = -aa/TAU
        ec, aa = model.response(forcing, s0)
        full_y.append(model.port(ec))
    h = rp+s0*lp
    x = np.linalg.solve(h, bp)
    reduced_y = [bp@x]
    for _ in range(1, count):
        x = -np.linalg.solve(h, lp@x)/TAU
        reduced_y.append(bp@x)
    full_z, reduced_z = inverse_series(full_y), inverse_series(reduced_y)
    # A count=2m+1 export checks matched coefficients 0,...,2m-1 and the first unmatched one.
    matched = count-1
    error = float(np.max(abs(full_z[:matched]-reduced_z[:matched]))/np.max(abs(full_z[:matched])))
    assert error < 1e-8, (error, full_z, reduced_z)
    return dict(variable="t=(s-s0)*TAU", TAU=TAU, contact_order=matched,
                full_Z_coefficients=full_z.tolist(), galerkin_Z_coefficients=reduced_z.tolist(),
                matched_coefficient_error=error)


def recursion(model, s0, nmodes):
    initial = np.zeros(1+model.na+model.np)
    initial[0] = 1
    e, a = model.response(initial, s0)
    x, ax = np.zeros_like(e), np.zeros_like(a)
    es, aas, xs, axs, R, L, energy = [], [], [], [], [], [], []
    for k in range(nmodes):
        loss = model.r(e, e)
        mag = a@(model.K@a)
        rh = 1/(loss+s0*mag)
        R.append(float(rh))
        es.append(e.copy()); aas.append(a.copy())
        x += rh*e; ax += rh*a
        lk = ax@(model.K@ax)
        L.append(float(lk)); xs.append(x.copy()); axs.append(ax.copy())
        volume = model.energy_integrals(e, a)
        xvolume = model.energy_integrals(x, ax)[1]
        assert max(abs(volume[0]/loss-1), abs(volume[1]/mag-1), abs(xvolume/lk-1)) < 1e-9
        energy.append(dict(joule=float(loss), magnetic=float(mag), R_Joule=float(1/loss),
                           field_integrals=volume, cumulative_magnetic_integral=xvolume))
        if k+1 < nmodes:
            u = np.zeros_like(e); u[1:1+model.na] = ax
            correction, ac = model.response(u, s0)
            e -= correction/lk; a -= ac/lk
    rp = np.array([[model.r(u,v) for v in es] for u in es])
    lp = np.array([[u@(model.K@v) for v in aas] for u in aas])
    bp = np.array([model.port(u) for u in es])
    hat = rp+s0*lp
    mx = np.array([[u@(model.K@v) for v in axs] for u in axs])
    def orth(mat):
        norm = np.sqrt(np.outer(np.diag(mat), np.diag(mat)))
        return float(np.max(abs((mat-np.diag(np.diag(mat)))/norm)))
    electric_orth, magnetic_orth = orth(hat), orth(mx)
    assert electric_orth <= 1e-8 and magnetic_orth <= 1e-8
    assert min(R+L) > 0
    comparisons = []
    for hz in FREQUENCIES:
        s = 2j*math.pi*hz
        ef, af = model.response(initial, s)
        zfull = 1/model.port(ef)
        zlad = ladder(R,L,s,s0)
        zgal = 1/(bp@np.linalg.solve(rp+s*lp,bp))
        lg = abs(zlad/zgal-1)
        assert lg < 1e-10
        loss = float(np.real(model.r(ef.conj(),ef)))
        mag = float(np.real(af.conj()@(model.K@af)))
        vr = model.energy_integrals(ef.real,af.real)
        vi = model.energy_integrals(ef.imag,af.imag)
        volume_defect = max(abs((vr[0]+vi[0])/loss-1),abs((vr[1]+vi[1])/mag-1))
        assert volume_defect < 1e-9
        power = abs((loss+s*mag)/np.conj(model.port(ef))-1)
        assert power < 1e-8, power
        delta = math.sqrt(2/(2*math.pi*hz*MU*SIGMA))
        comparisons.append(dict(hz=hz, full_Z=complex_pair(zfull), ladder_Z=complex_pair(zlad),
                                reduction_error=float(abs(zlad/zfull-1)), ladder_galerkin_error=float(lg),
                                power_balance_error=float(power), skin_depth_m=delta,
                                full_field_energy_defect=float(volume_defect),terminal_reactions=model.terminal_reactions(ef),
                                mesh_resolution_screen=delta >= 2*model.metadata["maxh"]/model.metadata["order"],
                                full_loss=loss, full_magnetic_energy_form=mag))
    phi_energy = [float(model.ng.Integrate(model.ng.grad(model.gf_p(model.split(u)[2]))**2*model.dx,model.mesh)) for u in es]
    gauge_errors=[]
    for weight in [.1,10.]:
        alt,_=model.response(initial,2j*math.pi*1e4,gauge_weight=weight)
        standard,_=model.response(initial,2j*math.pi*1e4)
        gauge_errors.append(float(abs(model.port(alt)/model.port(standard)-1)))
    assert max(gauge_errors)<1e-8
    return dict(s0=s0, modes=nmodes, Rhat=R, L=L, energy=energy, R_phys=rp.tolist(),
                L_phys=lp.tolist(), port=bp.tolist(), electric_orth=electric_orth,
                scalar_gradient_norm_squared=phi_energy,gauge_port_invariance_max=max(gauge_errors),
                magnetic_orth=magnetic_orth, frequency=comparisons,
                moments=moments(model,s0,rp,lp,bp,2*nmodes+1),
                electric_modes=[u.tolist() for u in es], magnetic_potentials=[u.tolist() for u in aas],
                cumulative_current_modes=[u.tolist() for u in xs],
                cumulative_magnetic_potentials=[u.tolist() for u in axs])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--export")
    p.add_argument("--maxh", type=float, nargs="+", default=[.004,.003,.002])
    p.add_argument("--order", type=int, default=1)
    p.add_argument("--modes", type=int, default=2)
    p.add_argument("--with-t", action="store_true", help="Run side-trace current-port A-T/T-Omega cross-checks")
    args = p.parse_args()
    if args.order not in [1,2] or args.modes<2 or any(not np.isfinite(h) or h<=0 for h in args.maxh):
        p.error("Require order 1/2, at least two modes, and positive finite maxh")
    import ngsolve, netgen
    ngsolve.SetNumThreads(4)
    identities = source_identity()
    result = dict(scope="Physical Type2 terminal CLN; notched conductor with n x A=0; no exterior",
                  source_sha256=identities, complete=False, cases=[],
                  runtime=dict(python=platform.python_version(), ngsolve=ngsolve.__version__, netgen=netgen.__version__),
                  solver="Explicit sparse LU for all subproblems; diagonal equilibration for mixed systems",
                  limitations="Intermesh differences are not exact errors. Both 100 kHz and 1 MHz fail the delta >= 2h/p screen on every study mesh and support only same-mesh reduction comparisons; screen passage is not an FE accuracy bound.")
    for h in args.maxh:
        start = time.perf_counter()
        with ngsolve.TaskManager():
            model = Model(h,args.order)
            runs = [recursion(model,s0,args.modes) for s0 in [0., 2*math.pi*1e4]]
            t_data, t_exports = {}, {}
            if args.with_t:
                from general_shape_t import CurrentModel, run_current
                for formulation in ["A-T","T-Omega"]:
                    print("cross-check", formulation, h, args.order, flush=True)
                    current = CurrentModel(model,formulation)
                    t_data[formulation] = [run_current(current,s0,args.modes) for s0 in [0.,2*math.pi*1e4]]
                    if args.export and h == args.maxh[0]:
                        t_exports[formulation] = current.export()
        result["cases"].append(dict(mesh=model.metadata, runs=runs, seconds=time.perf_counter()-start,
                                    kernel_defect=model.kernel_defect, kernel_fraction_max=max(model.kernel_fractions),
                                    original_residual_max=max(model.residuals)))
        result["cases"][-1]["cross_formulations"] = t_data
        if args.export and h == args.maxh[0]:
            export = model.export()
            export.update(source_sha256=identities, runs=runs, complete=True)
            export["cross_formulations"] = t_exports
            export["cross_formulation_runs"] = t_data
            Path(args.export).write_text(json.dumps(export,indent=2)+"\n",encoding="utf-8")
        # Full mode arrays belong to the small checker export, not every refined case.
        array_keys={"electric_modes","magnetic_potentials","cumulative_current_modes",
                    "cumulative_magnetic_potentials","current_mode_coefficients","magnetic_coefficients"}
        result["cases"][-1]["runs"]=[{key:value for key,value in run.items() if key not in array_keys} for run in runs]
        result["cases"][-1]["cross_formulations"]={name:[{key:value for key,value in run.items() if key not in array_keys}
            for run in group] for name,group in t_data.items()}
        Path(args.output).write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        print(h,args.order,model.mesh.ne,result["cases"][-1]["seconds"],flush=True)
    assert source_identity() == identities, "Source changed during computation"
    result["complete"] = True
    Path(args.output).write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
