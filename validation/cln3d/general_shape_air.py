"""Physical terminal Type2 CLN in a perfectly conducting enclosure.

Conductor-only conductivity and global magnetic energy; ideal insulated
bottom voltage port and bonded top return. No displacement current or shell loss.
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
    return sp.csr_matrix(form.mat.CSR(),shape=(form.mat.height,form.mat.width)).copy()


def packed(mat):
    a = sp.coo_matrix(mat)
    return dict(shape=list(a.shape), row=a.row.tolist(), col=a.col.tolist(), data=a.data.tolist())


def complex_pair(value):
    return [float(np.real(value)), float(np.imag(value))]


def source_identity():
    source = Path(__file__)
    return {"validation/cln3d/"+p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [source,source.with_name("general_shape_air_t.py"),source.with_name("coax_oracle.py"),source.with_name("general_shape_air_controls.py"),source.with_name("general_shape_air_study.py"),source.with_name("impedance_taylor.py")]}


def refined_solve(lu,matrix,rhs):
    """Three residual corrections of the SAME sparse LU, no mode reorthogonalization.

    NumPy longdouble is platform-dependent and equals float64 on Windows."""
    dtype=np.clongdouble if np.iscomplexobj(rhs) or np.iscomplexobj(matrix.data) else np.longdouble
    out=lu.solve(rhs)
    precise=matrix.astype(dtype)
    for _ in range(3):
        defect=np.asarray(rhs,dtype=dtype)-precise@np.asarray(out,dtype=dtype)
        out+=lu.solve(np.asarray(defect,dtype=out.dtype))
    return out


class Model:
    def __init__(self, maxh, order, bonus=6, airh=.008, enclosure=.020, geometry="notched", geometry_order=None, interior_control=False, refine_region=None, refine_steps=0):
        import ngsolve as ng
        from netgen.occ import Box, Cylinder, OCCGeometry, Glue, Z
        self.ng = ng
        if geometry == "coax":
            conductor = Cylinder((0,0,0), Z, r=.005, h=HEIGHT)
            box = Cylinder((0,0,0), Z, r=enclosure, h=HEIGHT)
        else:
            conductor = Box((-.01,-.01,0),(.01,.01,HEIGHT))-Box((.002,.002,.0075),(.012,.012,.017))
            box = Box((-enclosure,-enclosure,0),(enclosure,enclosure,HEIGHT))
        conductor.faces.name="surface"
        conductor.faces.Min(Z).name="in"; conductor.faces.Max(Z).name="out"
        conductor.mat("conductor");conductor.solids.maxh=maxh;conductor.faces.maxh=maxh;conductor.edges.maxh=maxh
        box.faces.name="outer";box.faces.maxh=airh;box.edges.maxh=airh
        air=box-conductor;air.mat("air");air.solids.maxh=airh
        shape=Glue([conductor,air])
        self.mesh=ng.Mesh(OCCGeometry(shape).GenerateMesh(maxh=min(maxh,airh),curvaturesafety=3 if geometry=="coax" else 2))
        for _ in range(refine_steps):
            for element in self.mesh.Elements(ng.VOL):
                self.mesh.SetRefinementFlag(element,refine_region=="both" or element.mat==refine_region)
            self.mesh.Refine(onlyonce=True)
        geometry_order=geometry_order or 2
        self.mesh.Curve(geometry_order)
        self.va=ng.HCurl(self.mesh,order=order,nograds=False,dirichlet="outer|in|out"+("|surface" if interior_control else ""))
        self.vp=ng.H1(self.mesh,order=order+1,dirichlet="in|out")
        self.fa = np.array(list(self.va.FreeDofs() & self.va.GetDofs(self.mesh.Materials("conductor")) if interior_control else self.va.FreeDofs()), bool)
        self.fp = np.array(list(self.vp.FreeDofs() & self.vp.GetDofs(self.mesh.Materials("conductor"))), bool)
        intrules={ng.ET.TET:ng.IntegrationRule(ng.ET.TET,order=2*order+bonus+8)}
        self.dx = ng.dx(intrules=intrules)
        self.dxD = ng.dx("conductor",intrules=intrules)
        self.dxAir = ng.dx("air",intrules=intrules)
        a, w = self.va.TnT()
        p, q = self.vp.TnT()
        sf = ng.BilinearForm(self.vp)
        sf += SIGMA * ng.grad(p) * ng.grad(q) * self.dxD
        sf.Assemble()
        self.S = csr(sf)[self.fp][:, self.fp].tocsc()
        self.slu = sla.splu(self.S)
        self.phi0 = ng.GridFunction(self.vp)
        self.phi0.Set(1, definedon=self.mesh.Boundaries("in"))
        lift = self.phi0.vec.CreateVector()
        lift.data = sf.mat * self.phi0.vec
        self.phi0.vec.FV().NumPy()[self.fp] = refined_solve(self.slu,self.S,-lift.FV().NumPy()[self.fp])
        self.e0 = -ng.grad(self.phi0)
        self.G0 = float(ng.Integrate(SIGMA*self.e0*self.e0*self.dxD, self.mesh))
        self.Kform = ng.BilinearForm(self.va)
        self.Kform += ng.curl(a)*ng.curl(w)/MU*self.dx
        self.Kform.Assemble()
        kd=ng.BilinearForm(self.va);kd+=ng.curl(a)*ng.curl(w)/MU*self.dxD;kd.Assemble()
        ka=ng.BilinearForm(self.va);ka+=ng.curl(a)*ng.curl(w)/MU*self.dxAir;ka.Assemble()
        self.K_D=csr(kd)[self.fa][:,self.fa].tocsc()
        self.K_air=csr(ka)[self.fa][:,self.fa].tocsc()
        self.K = csr(self.Kform)[self.fa][:, self.fa].tocsc()
        mf = ng.BilinearForm(self.va)
        mf += SIGMA*a*w*self.dxD
        mf.Assemble()
        self.M = csr(mf)[self.fa][:, self.fa].tocsc()
        lf = ng.LinearForm(self.va)
        lf += SIGMA*self.e0*w*self.dxD
        lf.Assemble()
        self.f = lf.vec.FV().NumPy()[self.fa].copy()
        mixed = self.va*self.vp
        (a, p), (w, q) = mixed.TnT()
        cf = ng.BilinearForm(mixed)
        cf += SIGMA*a*ng.grad(q)*self.dxD
        cf.Assemble()
        self.C = csr(cf)[self.va.ndof:, :self.va.ndof][self.fp][:, self.fa].tocsc()
        gm, h1 = self.va.CreateGradient()
        gh = np.array(list(h1.FreeDofs() & h1.GetDofs(self.mesh.Materials("conductor")) if interior_control else h1.FreeDofs()), bool)
        self.G = sp.csr_matrix(gm.CSR(),shape=(gm.height,gm.width)).copy()[self.fa][:, gh].tocsc()
        assert h1.ndof==self.vp.ndof
        self.P=sp.eye(h1.ndof,format="csr")[self.fp][:,gh].tocsc()
        def defect(a,b):return float(sla.norm(a-b)/max(sla.norm(a),sla.norm(b),1e-300))
        self.identities=dict(K_split=defect(self.K,self.K_D+self.K_air),
            MG_CTP=defect(self.M@self.G,self.C.T@self.P),CG_SP=defect(self.C@self.G,self.S@self.P),
            fG=float(np.linalg.norm(self.G.T@self.f)/max(sla.norm(self.G)*np.linalg.norm(self.f),1e-300)))
        assert max(self.identities.values())<1e-10,self.identities
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
                             free_A=self.na, free_phi=self.np, sigma=SIGMA, sigma_air=0., mu=MU,
                             volume=float(ng.Integrate(1, self.mesh)),
                             effective_global_maxh=min(maxh,airh),geometry=geometry,curvaturesafety=3 if geometry=="coax" else 2,interior_control=interior_control,airh=airh,enclosure=enclosure,geometry_order=geometry_order,refine_region=refine_region,refine_steps=refine_steps,
                             conductor_volume=float(ng.Integrate(ng.CF(1)*self.dxD,self.mesh)),
                             air_volume=float(ng.Integrate(ng.CF(1)*self.dxAir,self.mesh)),
                             boundary_names=list(self.mesh.GetBoundaries()),
                             boundary_area={name: float(ng.Integrate(1, self.mesh,
                                 definedon=self.mesh.Boundaries(name))) for name in ["in", "out", "surface", "outer"]})
        self.geometry_arrays = dict(points=[list(point.p) for point in self.mesh.ngmesh.Points()],
                                    tetrahedra=[[v.nr for v in el.vertices] for el in self.mesh.ngmesh.Elements3D()])
        self.metadata["mesh_sha256"] = hashlib.sha256(json.dumps(self.geometry_arrays,sort_keys=True).encode()).hexdigest()
        regions=[el.index for el in self.mesh.ngmesh.Elements3D()]
        self.geometry_arrays["region_indices"]=regions
        self.geometry_arrays["materials"]=list(self.mesh.GetMaterials())
        points=np.array(self.geometry_arrays["points"])
        sizes={"conductor":[],"air":[]};jacobians=[];centers=[]
        rule=ng.IntegrationRule(ng.ET.TET,order=4)
        for element,indices,region in zip(self.mesh.Elements(ng.VOL),self.geometry_arrays["tetrahedra"],regions):
            xyz=points[np.array(indices)-1]
            sizes[self.mesh.GetMaterials()[region-1]].append(float(np.max(np.linalg.norm(xyz[:,None]-xyz[None,:],axis=2))))
            tr=self.mesh.GetTrafo(element)
            determinants=[float(np.linalg.det(np.array(tr(ip).jacobi))) for ip in rule]
            assert min(determinants)>0,(element.nr,min(determinants))
            jacobians.extend(determinants)
            centers.append(list(tr(ng.IntegrationRule(ng.ET.TET,order=1)[0]).point))
        self.geometry_arrays["curved_cell_centers"]=centers
        self.metadata["region_element_counts"]={k:len(v) for k,v in sizes.items()}
        self.metadata["region_max_chord_m"]={k:max(v) for k,v in sizes.items()}
        self.metadata["minimum_sampled_jacobian_determinant"]=min(jacobians)
        self.metadata["mesh_sha256"]=hashlib.sha256(json.dumps(self.geometry_arrays,sort_keys=True).encode()).hexdigest()
        if geometry=="coax":
            exact=math.pi*.005**2*HEIGHT
            self.metadata["conductor_volume_error"]=abs(self.metadata["conductor_volume"]/exact-1)
            self.metadata["contact_area_error"]=abs(self.metadata["boundary_area"]["in"]/(math.pi*.005**2)-1)
            side=float(ng.Integrate(1,self.mesh,definedon=self.mesh.Boundaries("surface")))
            self.metadata["conductor_perimeter_error"]=abs(side/(2*math.pi*.005*HEIGHT)-1)
        self.metadata["phi_departure_l2_relative"]=float(np.sqrt(ng.Integrate(
            (self.phi0-(1-ng.z/HEIGHT))**2*self.dxD,self.mesh)/ng.Integrate(self.phi0**2*self.dxD,self.mesh)))
        self.metadata["dc_transverse_current_fraction"]=float(ng.Integrate(
            (self.e0[0]**2+self.e0[1]**2)*self.dxD,self.mesh)/ng.Integrate(self.e0*self.e0*self.dxD,self.mesh))

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
        a = refined_solve(self.klu,self.Kg,projected)
        residual = float(np.linalg.norm(self.K@a-rhs)/np.linalg.norm(rhs))
        self.residuals.append(residual)
        assert residual <= 1e-8
        return a

    def response(self, e, s, gauge_weight=1):
        alpha, v, p = self.split(e)
        rhs_a = self.load(e)
        rhs_p = self.C@v-self.S@p
        if s == 0:
            chi = refined_solve(self.slu,self.S,rhs_p)
            out = e.copy()
            out[1+self.na:] += chi
            return out, self.magnetic(out)
        key = (complex(s), gauge_weight)
        if key not in self.response_lus:
            kg = self.K+gauge_weight*self.gauge_scale*(self.G@self.G.T)
            block = sp.bmat([[kg+s*self.M, self.C.T], [self.C, self.S/s]], format="csc")
            scale = 1/np.sqrt(abs(block.diagonal()))
            d = sp.diags(scale)
            self.response_lus[key] = (sla.splu((d@block@d).tocsc()), scale,(d@block@d).tocsc())
        lu, scale, block = self.response_lus[key]
        solution = scale*refined_solve(lu,block,scale*np.r_[rhs_a, rhs_p/s])
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
        return [float(self.ng.Integrate(SIGMA*ef*ef*self.dxD, self.mesh)),
                float(self.ng.Integrate(bf*bf/MU*self.dx, self.mesh))]

    def terminal_reactions(self,e):
        reactions = {}
        for part in ["real","imag"]:
            ef = self.field(getattr(e,part))
            lf = self.ng.LinearForm(self.vp)
            lf += SIGMA*ef*self.ng.grad(self.vp.TestFunction())*self.dxD
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

    def circulation(self,a,current):
        if self.metadata["interior_control"]:return {}
        # Two independent loop radii/sizes, midpoint sampling, no field fit.
        field=self.ng.curl(self.gf_a(a))/MU
        rows=[]
        for fraction in [.72,.90]:
            if self.metadata["geometry"]=="coax":
                radius=.005+fraction*(self.metadata["enclosure"]-.005)
                ts=(np.arange(256)+.5)*2*math.pi/256
                points=[(radius*math.cos(t),radius*math.sin(t),HEIGHT/2) for t in ts]
                tangents=[(-radius*math.sin(t),radius*math.cos(t),0) for t in ts]
                step=2*math.pi/256
            else:
                extent=.01+fraction*(self.metadata["enclosure"]-.01)
                points=[];tangents=[];step=1/128
                corners=[(-extent,-extent),(extent,-extent),(extent,extent),(-extent,extent),(-extent,-extent)]
                for p,q in zip(corners,corners[1:]):
                    for t in (np.arange(128)+.5)/128:
                        points.append((p[0]+t*(q[0]-p[0]),p[1]+t*(q[1]-p[1]),HEIGHT/2))
                        tangents.append((q[0]-p[0],q[1]-p[1],0))
            value=sum(np.dot(np.array(field(self.mesh(*p))),v)*step for p,v in zip(points,tangents))
            rows.append(dict(loop_fraction=fraction,H_circulation=float(value),relative_current_defect=float(abs(value/current-1))))
        return dict(sampling="256-point circle or 128 midpoint samples per rectangle side at mid-height; FE sampling diagnostic, not an exact trace",loops=rows)

    def export(self):
        return dict(K=packed(self.K), M=packed(self.M), S=packed(self.S), C=packed(self.C),
                    G=packed(self.G), P=packed(self.P), K_D=packed(self.K_D), K_air=packed(self.K_air), identities=self.identities, f=self.f.tolist(), G0=self.G0, gauge_scale=self.gauge_scale,
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
    # Direct unit-current constrained coefficients avoid inverse-series cancellation.
    from impedance_taylor import mixed, reduced
    full_z=mixed(model,s0,count,TAU)
    reduced_z=reduced(rp,lp,bp,s0,count,TAU)
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
        bd=model.ng.curl(model.gf_a(a));bx=model.ng.curl(model.gf_a(ax))
        split=[float(model.ng.Integrate(bd*bd/MU*dx,model.mesh)) for dx in [model.dxD,model.dxAir]]
        splitx=[float(model.ng.Integrate(bx*bx/MU*dx,model.mesh)) for dx in [model.dxD,model.dxAir]]
        assert abs(sum(split)/mag-1)<1e-9 and abs(sum(splitx)/lk-1)<1e-9
        energy.append(dict(magnetic_region_integrals=split,cumulative_region_integrals=splitx,joule=float(loss), magnetic=float(mag), R_Joule=float(1/loss),
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
    assert electric_orth <= 1e-8 and magnetic_orth <= 1e-8,(s0,electric_orth,magnetic_orth)
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
        regions=[float(np.real(af.conj()@(k@af))) for k in [model.K_D,model.K_air]]
        integrated=[]
        for dx in [model.dxD,model.dxAir]:
            integrated.append(sum(float(model.ng.Integrate(model.ng.curl(model.gf_a(part))**2/MU*dx,model.mesh)) for part in [af.real,af.imag]))
        assert max(abs(np.array(integrated)-regions)/np.maximum(abs(np.array(regions)),1e-14*mag))<1e-9
        comparisons.append(dict(magnetic_regions=regions,magnetic_region_integrals=integrated,hz=hz, full_Z=complex_pair(zfull), ladder_Z=complex_pair(zlad),
                                reduction_error=float(abs(zlad/zfull-1)), ladder_galerkin_error=float(lg),
                                power_balance_error=float(power), skin_depth_m=delta,
                                full_field_energy_defect=float(volume_defect),terminal_reactions=model.terminal_reactions(ef),
                                mesh_resolution_screen=delta >= 2*model.metadata["region_max_chord_m"]["conductor"]/model.metadata["order"],
                                full_loss=loss, full_magnetic_energy_form=mag))
    phi_energy = [float(model.ng.Integrate(model.ng.grad(model.gf_p(model.split(u)[2]))**2*model.dxD,model.mesh)) for u in es]
    gauge_errors=[];gauge_fields=[]
    for weight in [.1,10.]:
        alt,aa=model.response(initial,2j*math.pi*1e4,gauge_weight=weight)
        standard,bb=model.response(initial,2j*math.pi*1e4)
        gauge_errors.append(float(abs(model.port(alt)/model.port(standard)-1)))
        de=alt-standard;da=aa-bb
        jdef=sum(float(model.ng.Integrate(SIGMA*model.field(v)**2*model.dxD,model.mesh)) for v in [de.real,de.imag])
        jnorm=float(np.real(model.r(standard.conj(),standard)))
        bdef=float(np.real(da.conj()@(model.K@da)));bnorm=float(np.real(bb.conj()@(model.K@bb)))
        gauge_fields.append(dict(weight=weight,J_relative=math.sqrt(abs(jdef/jnorm)),B_relative=math.sqrt(abs(bdef/bnorm))))
        assert max(gauge_fields[-1]['J_relative'],gauge_fields[-1]['B_relative'])<1e-8
    assert max(gauge_errors)<1e-8
    return dict(s0=s0, modes=nmodes, Rhat=R, L=L, energy=energy, R_phys=rp.tolist(),
                L_phys=lp.tolist(), port=bp.tolist(), electric_orth=electric_orth,
                gauge_field_invariance=gauge_fields,scalar_gradient_norm_squared=phi_energy,gauge_port_invariance_max=max(gauge_errors),
                magnetic_orth=magnetic_orth, frequency=comparisons,
                moments=moments(model,s0,rp,lp,bp,2*nmodes+1),
                electric_modes=[u.tolist() for u in es], magnetic_potentials=[u.tolist() for u in aas],
                cumulative_current_modes=[u.tolist() for u in xs],
                cumulative_magnetic_potentials=[u.tolist() for u in axs])



def compute_case(maxh,airh,order,enclosure,geometry,modes=4,with_at=True,bonus=6,export_path=None,refine_region=None,refine_steps=0):
    import ngsolve as ng
    start=time.perf_counter()
    with ng.TaskManager():
        model=Model(maxh,order,bonus=bonus,airh=airh,enclosure=enclosure,geometry=geometry,refine_region=refine_region,refine_steps=refine_steps)
        runs=[recursion(model,s,modes) for s in [0.,2*math.pi*1e4]]
        at_runs=[];at_export={}
        if with_at:
            from general_shape_air_t import CurrentModel,run_current
            current=CurrentModel(model,"A-T")
            at_runs=[run_current(current,s,modes) for s in [0.,2*math.pi*1e4]]
            at_export=current.export()
    case=dict(mesh=model.metadata,identities=model.identities,runs=runs,A_T_runs=at_runs,
        kernel_defect=model.kernel_defect,kernel_fraction_max=max(model.kernel_fractions),
        original_residual_max=max(model.residuals),seconds=time.perf_counter()-start)
    initial=np.zeros(1+model.na+model.np);initial[0]=1
    edc,adc=model.response(initial,0.)
    case["DC_ampere"]=model.circulation(adc,model.port(edc))
    if geometry=="coax":
        from coax_oracle import impedance,elements
        for run in runs:
            rr,ll=elements(run["s0"],modes)
            run["oracle_Rhat"]=rr;run["oracle_L"]=ll
            for row in run["frequency"]:
                oracle=complex(impedance(2j*math.pi*row["hz"],b=enclosure))
                row["oracle_Z"]=complex_pair(oracle)
                row["full_oracle_error"]=float(abs(complex(*row["full_Z"])/oracle-1))
        dc=runs[0]["energy"][0]["cumulative_region_integrals"]
        targets=[MU*HEIGHT/(8*math.pi),MU*HEIGHT*math.log(enclosure/.005)/(2*math.pi)]
        case["oracle_dc_region_L"]=targets
        case["oracle_dc_region_relative_errors"]=[abs(a/b-1) for a,b in zip(dc,targets)]
    if export_path:
        out=model.export();out.update(runs=runs,cross_formulations={"A-T":at_export},
            cross_formulation_runs={"A-T":at_runs},complete=True,source_sha256=source_identity())
        Path(export_path).write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8",newline="\n")
    arrays={"electric_modes","magnetic_potentials","cumulative_current_modes",
        "cumulative_magnetic_potentials","current_mode_coefficients","magnetic_coefficients"}
    case["runs"]=[{k:v for k,v in run.items() if k not in arrays} for run in runs]
    case["A_T_runs"]=[{k:v for k,v in run.items() if k not in arrays} for run in at_runs]
    return case


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True);p.add_argument("--export")
    p.add_argument("--geometry",choices=["coax","notched"],default="coax")
    p.add_argument("--conductor-h",type=float,default=.006);p.add_argument("--air-h",type=float,default=.008)
    p.add_argument("--order",type=int,default=1);p.add_argument("--enclosure",type=float,default=.015)
    p.add_argument("--modes",type=int,default=4);p.add_argument("--no-at",action="store_true")
    args=p.parse_args()
    import ngsolve,netgen
    ngsolve.SetNumThreads(4);identities=source_identity()
    case=compute_case(args.conductor_h,args.air_h,args.order,args.enclosure,args.geometry,args.modes,not args.no_at,export_path=args.export)
    assert source_identity()==identities
    out=dict(complete=True,source_sha256=identities,cases=[case],
        runtime=dict(python=platform.python_version(),ngsolve=ngsolve.__version__,netgen=netgen.__version__),
        scope="Conductor in a perfectly conducting enclosure, ideal source-break terminal port; sigma_air=0 exactly; total magnetic energy")
    Path(args.output).write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8",newline="\n")
    print("PASS native",args.geometry,args.order,case["mesh"]["ne"],case["seconds"])


if __name__=="__main__":main()
