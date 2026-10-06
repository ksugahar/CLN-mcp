"""Conductor-supported A-T cross-check with global enclosure magnetic solve.

The common DC current lift and global A magnetic operator are shared.
T-Omega is excluded because air circulation needs a cut/source field.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from general_shape_air import SIGMA, MU, TAU, FREQUENCIES, csr, packed, ladder, inverse_series, complex_pair, refined_solve


def real_lu_solve(lu,rhs,matrix):
    return refined_solve(lu,matrix,rhs.real)+1j*refined_solve(lu,matrix,rhs.imag) if np.iscomplexobj(rhs) else refined_solve(lu,matrix,rhs)


class CurrentModel:
    def __init__(self, base, formulation):
        assert formulation=="A-T", "T-Omega excluded: air circulation requires a cut/source field"
        ng = base.ng
        self.base, self.formulation = base, formulation
        self.vt = ng.HCurl(base.mesh, order=base.metadata["order"], nograds=False, definedon=base.mesh.Materials("conductor"), dirichlet="surface")
        self.ft = np.array(list(self.vt.FreeDofs()), bool)
        t, w = self.vt.TnT()
        rf = ng.BilinearForm(self.vt)
        rf += ng.curl(t)*ng.curl(w)/SIGMA*base.dxD
        rf.Assemble()
        rt = csr(rf)[self.ft][:,self.ft].tocsc()
        self.R = sp.block_diag([sp.csc_matrix([[1/base.G0]]),rt],format="csc")
        gm, h1 = self.vt.CreateGradient()
        fh = np.array(list(h1.FreeDofs()),bool)
        gt = sp.csr_matrix(gm.CSR(),shape=(gm.height,gm.width)).copy()[self.ft][:,fh]
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
        port_test += base.e0*ng.curl(self.vt.TestFunction())*base.dxD
        port_test.Assemble()
        self.zero_port_load_norm = float(np.linalg.norm(port_test.vec.FV().NumPy()[self.ft]))
        assert self.zero_port_load_norm < 1e-8,self.zero_port_load_norm
        self.jdc = SIGMA*base.e0/base.G0
        self.adc = base.magnetic(np.r_[1/base.G0,np.zeros(base.na+base.np)])
        self.hdc = ng.curl(base.gf_a(self.adc))/MU
        dc = self.rlu.solve(self.b)
        assert abs(dc[0]/base.G0-1)<1e-12 and np.linalg.norm(dc[1:])<1e-12
        self.residuals = []
        self.lus = {}
        mixed = base.va*self.vt
        (a,t),(n,w) = mixed.TnT()
        df = ng.BilinearForm(mixed)
        df += a*ng.curl(w)*base.dxD
        df.Assemble()
        dt = csr(df)[base.va.ndof:,:base.va.ndof][self.ft][:,base.fa].T
        self.D = sp.hstack([sp.csc_matrix((base.f/base.G0)[:,None]),dt],format="csc")

    def magnetic(self,z):
        a=real_lu_solve(self.base.klu,self.D@z,self.base.Kg)
        res=np.linalg.norm(self.base.K@a-self.D@z)/np.linalg.norm(self.D@z)
        self.residuals.append(float(res));assert res<1e-8
        return a

    def ell(self,z,v):
        return (self.D@z)@real_lu_solve(self.base.klu,self.D@v,self.base.Kg)

    def ell_load(self,z):return self.D.T@self.magnetic(z)

    def response(self,rhs,s):
        if s == 0:
            z = refined_solve(self.rlu,self.Rg,rhs)
            return z
        key = complex(s)
        if key not in self.lus:
            block=sp.bmat([[self.Rg,self.D.T],[self.D,-self.base.Kg/s]],format="csc")
            scale = 1/np.sqrt(abs(block.diagonal()))
            d = sp.diags(scale)
            self.lus[key] = (sla.splu((d@block@d).tocsc()),scale,(d@block@d).tocsc())
        lu,scale,block = self.lus[key]
        extra = len(scale)-len(rhs)
        solution = scale*refined_solve(lu,block,scale*np.r_[rhs,np.zeros(extra)])
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
        b=ng.curl(base.gf_a(self.magnetic(z)))
        return j,b

    def export(self):
        out = dict(formulation=self.formulation,R=packed(self.R),Rg=packed(self.Rg),G=packed(self.G),
                   b=self.b.tolist(),free_T=self.ft.tolist(),port_current_lift="j_dc=sigma*e0/G0",
                   trace="n x T=0 on insulating side including rims; natural on terminal faces",
                   zero_port_load_norm=self.zero_port_load_norm)
        out["D"]=packed(self.D)
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
        loss_volume = float(base.ng.Integrate(j*j/SIGMA*base.dxD,base.mesh))
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
    from impedance_taylor import current, reduced
    zf=current(model,s0,count,TAU)
    zg=reduced(rp,lp,bp,s0,count,TAU)
    error=float(np.max(abs(zf[:-1]-zg[:-1]))/np.max(abs(zf[:-1])))
    assert error < 1e-8
    return dict(s0=s0,modes=nmodes,Rhat=Rs,L=Ls,R_phys=rp.tolist(),L_phys=lp.tolist(),port=bp.tolist(),
                energy=energies,electric_orth=eo,magnetic_orth=mo,frequency=hz_data,
                moments=dict(variable="t=(s-s0)*TAU",TAU=TAU,contact_order=2*nmodes,
                             full_Z_coefficients=zf.tolist(),galerkin_Z_coefficients=zg.tolist(),matched_coefficient_error=error),
                current_mode_coefficients=[u.tolist() for u in zs],cumulative_current_modes=[u.tolist() for u in xs],
                magnetic_coefficients=[model.magnetic(u).tolist() for u in zs],
                original_residual_max=max(model.residuals),kernel_defect=model.kernel_defect)
