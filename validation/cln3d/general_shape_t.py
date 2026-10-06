"""D2 current-port cross-checks: J=I*j_dc+curl(T), side-only T trace.

The common DC current lift comes from A-phi. T-Omega also uses its magnetic
port lift, explicitly shared; correction magnetic fields are independent.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from general_shape import SIGMA, MU, TAU, FREQUENCIES, csr, packed, ladder, inverse_series, complex_pair


def real_lu_solve(lu,rhs):
    return lu.solve(rhs.real)+1j*lu.solve(rhs.imag) if np.iscomplexobj(rhs) else lu.solve(rhs)


class CurrentModel:
    def __init__(self, base, formulation):
        ng = base.ng
        self.base, self.formulation = base, formulation
        self.vt = ng.HCurl(base.mesh, order=base.metadata["order"], nograds=False, dirichlet="surface")
        self.ft = np.array(list(self.vt.FreeDofs()), bool)
        t, w = self.vt.TnT()
        rf = ng.BilinearForm(self.vt)
        rf += ng.curl(t)*ng.curl(w)/SIGMA*base.dx
        rf.Assemble()
        rt = csr(rf)[self.ft][:,self.ft].tocsc()
        self.R = sp.block_diag([sp.csc_matrix([[1/base.G0]]),rt],format="csc")
        gm, h1 = self.vt.CreateGradient()
        fh = np.array(list(h1.FreeDofs()),bool)
        gt = sp.csr_matrix(gm.CSR()).copy()[self.ft][:,fh]
        # Keep the port scalar out of the T gradient gauge.
        self.G = sp.vstack([sp.csr_matrix((1,gt.shape[1])),gt],format="csc")
        gg = self.G@self.G.T
        scale = abs(rt.diagonal()).max()/gg.diagonal().max()
        self.Rg = self.R+scale*gg
        self.rlu = sla.splu(self.Rg.tocsc())
        self.kernel_defect = float(sla.norm(self.R@self.G)/(sla.norm(self.R)*sla.norm(self.G)))
        assert self.kernel_defect < 1e-10
        self.b = np.zeros(self.R.shape[0]); self.b[0] = 1
        port_test = ng.LinearForm(self.vt)
        port_test += base.e0*ng.curl(self.vt.TestFunction())*base.dx
        port_test.Assemble()
        self.zero_port_load_norm = float(np.linalg.norm(port_test.vec.FV().NumPy()[self.ft]))
        assert self.zero_port_load_norm < 1e-8
        self.jdc = SIGMA*base.e0/base.G0
        self.adc = base.magnetic(np.r_[1/base.G0,np.zeros(base.na+base.np)])
        self.hdc = ng.curl(base.gf_a(self.adc))/MU
        dc = self.rlu.solve(self.b)
        assert abs(dc[0]/base.G0-1)<1e-12 and np.linalg.norm(dc[1:])<1e-12
        self.residuals = []
        self.lus = {}
        if formulation == "A-T":
            mixed = base.va*self.vt
            (a,t),(n,w) = mixed.TnT()
            df = ng.BilinearForm(mixed)
            df += a*ng.curl(w)*base.dx
            df.Assemble()
            dt = csr(df)[base.va.ndof:,:base.va.ndof][self.ft][:,base.fa].T
            self.D = sp.hstack([sp.csc_matrix((base.f/base.G0)[:,None]),dt],format="csc")
        else:
            self.vo = ng.H1(base.mesh,order=base.metadata["order"]+1)
            self.fo = np.array(list(self.vo.FreeDofs()),bool)
            self.fo[0] = False  # Neumann scalar gauge: remove one nodal constant degree of freedom.
            o,q = self.vo.TnT()
            of = ng.BilinearForm(self.vo)
            of += MU*ng.grad(o)*ng.grad(q)*base.dx
            of.Assemble()
            self.So = csr(of)[self.fo][:,self.fo].tocsc()
            self.olu = sla.splu(self.So)
            uf = ng.BilinearForm(self.vt)
            uf += MU*t*w*base.dx
            uf.Assemble()
            ut = csr(uf)[self.ft][:,self.ft]
            lf = ng.LinearForm(self.vt)
            lf += MU*self.hdc*w*base.dx
            lf.Assemble()
            cross = lf.vec.FV().NumPy()[self.ft].copy()
            energy = float(self.adc@(base.K@self.adc))
            self.U = sp.bmat([[sp.csc_matrix([[energy]]),sp.csr_matrix(cross[None,:])],
                              [sp.csc_matrix(cross[:,None]),ut]],format="csc")
            mixed = self.vt*self.vo
            (t,o),(w,q) = mixed.TnT()
            wf = ng.BilinearForm(mixed)
            wf += MU*t*ng.grad(q)*base.dx
            wf.Assemble()
            wt = csr(wf)[self.vt.ndof:,:self.vt.ndof][self.fo][:,self.ft]
            lo = ng.LinearForm(self.vo)
            lo += MU*self.hdc*ng.grad(self.vo.TestFunction())*base.dx
            lo.Assemble()
            w0 = lo.vec.FV().NumPy()[self.fo].copy()
            self.W = sp.hstack([sp.csc_matrix(w0[:,None]),wt],format="csc")

    def magnetic(self,z):
        if self.formulation == "A-T":
            a = real_lu_solve(self.base.klu,self.D@z)
            res = np.linalg.norm(self.base.K@a-self.D@z)/np.linalg.norm(self.D@z)
            self.residuals.append(float(res)); assert res < 1e-8
            return a
        return -real_lu_solve(self.olu,self.W@z)

    def ell(self,z,v):
        if self.formulation == "A-T":
            return (self.D@z)@real_lu_solve(self.base.klu,self.D@v)
        return z@(self.U@v)-(self.W@z)@real_lu_solve(self.olu,self.W@v)

    def ell_load(self,z):
        if self.formulation == "A-T":
            return self.D.T@self.magnetic(z)
        return self.U@z+self.W.T@self.magnetic(z)

    def response(self,rhs,s):
        if s == 0:
            z = self.rlu.solve(rhs)
            return z
        key = complex(s)
        if key not in self.lus:
            if self.formulation == "A-T":
                block = sp.bmat([[self.Rg,self.D.T],[self.D,-self.base.Kg/s]],format="csc")
            else:
                block = sp.bmat([[self.Rg+s*self.U,self.W.T],[self.W,self.So/s]],format="csc")
            scale = 1/np.sqrt(abs(block.diagonal()))
            d = sp.diags(scale)
            self.lus[key] = (sla.splu((d@block@d).tocsc()),scale)
        lu,scale = self.lus[key]
        extra = len(scale)-len(rhs)
        solution = scale*lu.solve(scale*np.r_[rhs,np.zeros(extra)])
        z = solution[:len(rhs)]
        residual = self.R@z+s*self.ell_load(z)-rhs
        defect = float(np.linalg.norm(residual)/max(np.linalg.norm(rhs),1e-300))
        self.residuals.append(defect); assert defect < 1e-8, defect
        return z

    def fields(self,z):
        ng,base = self.base.ng,self.base
        t = ng.GridFunction(self.vt)
        t.vec.FV().NumPy()[self.ft] = z[1:]
        j = z[0]*self.jdc+ng.curl(t)
        if self.formulation == "A-T":
            b = ng.curl(base.gf_a(self.magnetic(z)))
        else:
            omega = ng.GridFunction(self.vo)
            omega.vec.FV().NumPy()[self.fo] = self.magnetic(z)
            b = MU*(z[0]*self.hdc+t+ng.grad(omega))
        return j,b

    def export(self):
        out = dict(formulation=self.formulation,R=packed(self.R),Rg=packed(self.Rg),G=packed(self.G),
                   b=self.b.tolist(),free_T=self.ft.tolist(),port_current_lift="j_dc=sigma*e0/G0",
                   trace="n x T=0 on insulating side including rims; natural on terminal faces",
                   zero_port_load_norm=self.zero_port_load_norm)
        if self.formulation == "A-T":
            out["D"] = packed(self.D)
        else:
            out.update(U=packed(self.U),W=packed(self.W),So=packed(self.So),free_Omega=self.fo.tolist(),
                       magnetic_port_lift="H_dc=curl(A[j_dc])/mu, shared A-phi lift; independent T/Omega corrections")
        return out


def run_current(model,s0,nmodes):
    base = model.base
    z = model.response(model.b,s0)
    x = np.zeros_like(z)
    zs,xs,Rs,Ls,energies = [],[],[],[],[]
    for k in range(nmodes):
        r = z@(model.R@z); ell=model.ell(z,z)
        rh = 1/(r+s0*ell)
        x += rh*z
        lk = model.ell(x,x)
        Rs.append(float(rh)); Ls.append(float(lk)); zs.append(z.copy()); xs.append(x.copy())
        j,b = model.fields(z)
        loss_volume = float(base.ng.Integrate(j*j/SIGMA*base.dx,base.mesh))
        mag_volume = float(base.ng.Integrate(b*b/MU*base.dx,base.mesh))
        assert max(abs(loss_volume/r-1),abs(mag_volume/ell-1)) < 1e-9
        energies.append(dict(joule=float(r),magnetic=float(ell),field_integrals=[loss_volume,mag_volume]))
        if k+1 < nmodes:
            z -= model.response(model.ell_load(x),s0)/lk
    rp = np.array([[u@(model.R@v) for v in zs] for u in zs])
    lp = np.array([[model.ell(u,v) for v in zs] for u in zs])
    bp = np.array([model.b@u for u in zs])
    def orth(g):
        return float(np.max(abs((g-np.diag(np.diag(g)))/np.sqrt(np.outer(np.diag(g),np.diag(g))))))
    eo,mo = orth(rp+s0*lp),orth(np.array([[model.ell(u,v) for v in xs] for u in xs]))
    assert max(eo,mo) < 1e-8 and min(Rs+Ls)>0
    hz_data=[]
    for hz in FREQUENCIES:
        s=2j*np.pi*hz
        full_current=model.response(model.b,s)
        full_z=1/(model.b@full_current)
        loss=float(np.real(full_current.conj()@(model.R@full_current)))
        mag=float(np.real(model.ell(full_current.conj(),full_current)))
        power=float(abs((loss+s*mag)/np.conj(model.b@full_current)-1))
        assert power<1e-8
        reduced_z=ladder(Rs,Ls,s,s0)
        gal_z=1/(bp@np.linalg.solve(rp+s*lp,bp))
        lg=float(abs(reduced_z/gal_z-1));assert lg < 1e-10
        hz_data.append(dict(hz=hz,full_Z=complex_pair(full_z),ladder_Z=complex_pair(reduced_z),
                            reduction_error=float(abs(reduced_z/full_z-1)),ladder_galerkin_error=lg,
                            power_balance_error=power,full_loss=loss,full_magnetic_energy_form=mag))
    count=2*nmodes+1
    v=model.response(model.b,s0)
    yf=[model.b@v]
    for _ in range(1,count):
        v=-model.response(model.ell_load(v),s0)/TAU
        yf.append(model.b@v)
    v=np.linalg.solve(rp+s0*lp,bp); yg=[bp@v]
    for _ in range(1,count):
        v=-np.linalg.solve(rp+s0*lp,lp@v)/TAU;yg.append(bp@v)
    zf,zg=inverse_series(yf),inverse_series(yg)
    error=float(np.max(abs(zf[:-1]-zg[:-1]))/np.max(abs(zf[:-1])))
    assert error < 1e-8
    return dict(s0=s0,modes=nmodes,Rhat=Rs,L=Ls,R_phys=rp.tolist(),L_phys=lp.tolist(),port=bp.tolist(),
                energy=energies,electric_orth=eo,magnetic_orth=mo,frequency=hz_data,
                moments=dict(variable="t=(s-s0)*TAU",TAU=TAU,contact_order=2*nmodes,
                             full_Z_coefficients=zf.tolist(),galerkin_Z_coefficients=zg.tolist(),matched_coefficient_error=error),
                current_mode_coefficients=[u.tolist() for u in zs],cumulative_current_modes=[u.tolist() for u in xs],
                magnetic_coefficients=[model.magnetic(u).tolist() for u in zs],
                original_residual_max=max(model.residuals),kernel_defect=model.kernel_defect)
