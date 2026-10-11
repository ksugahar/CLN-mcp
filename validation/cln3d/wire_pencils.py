"""Independent full FE pencils for the matched internal round-wire study.

Scalar and magnetic subsolves use NGSolve sparsecholesky. Schur complements
are applied as operators, never regularized with a physical mass. This is a
linear-response/Taylor oracle, not another implementation of field CLN.
"""
import numpy as np
import ngsolve as ng
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg

from ._iccg import gradient_kernel


def csr(form):
    return sp.csr_matrix(form.mat.CSR()).copy()


def factor(matrix):
    """Diagonally scaled SPD Cholesky; retain matrix ownership in the closure."""
    scale=1/np.sqrt(matrix.diagonal())
    d=sp.diags(scale)
    coo=(d@matrix@d).tocoo()
    mat=ng.la.SparseMatrixdouble.CreateFromCOO(coo.row.tolist(),coo.col.tolist(),
                                              coo.data.tolist(),*matrix.shape)
    inv=mat.Inverse(inverse='sparsecholesky')
    def solve(rhs):
        vec=mat.CreateRowVector();vec.FV().NumPy()[:]=scale*rhs
        out=mat.CreateColVector();out.data=inv*vec
        value=scale*out.FV().NumPy().copy()
        defect=np.linalg.norm(matrix@value-rhs)/max(np.linalg.norm(rhs),1e-300)
        if not np.isfinite(defect) or defect>1e-8:
            raise RuntimeError(f'sparsecholesky subsolve failed: {defect}')
        return value
    return solve


def gauge(space, matrix, free):
    g,gfree=gradient_kernel(space)
    g=g[free][:,gfree]
    g=g[:,np.asarray(g.getnnz(axis=0)).ravel()>0]
    defect=float(sp.linalg.norm(matrix@g)/max(sp.linalg.norm(matrix)*sp.linalg.norm(g),1e-300))
    if defect>1e-10:
        raise RuntimeError(f'Invalid exact gradient kernel: {defect}')
    gg=g@g.T
    return gg*(max(abs(matrix.diagonal()))/max(gg.diagonal())),defect


class Pencils:
    def __init__(self, mesh, order, bonus=8):
        self.mesh=mesh
        self.sigma=1e6;self.mu=4e-7*np.pi
        self.tau=self.mu*self.sigma*.01**2
        measure=ng.dx(bonus_intorder=bonus)
        va=ng.HCurl(mesh,order=order,nograds=False,dirichlet='in|out|conductorBND')
        vc=ng.HCurl(mesh,order=order,nograds=True,dirichlet='in|out|conductorBND')
        vt=ng.HCurl(mesh,order=order,nograds=True)
        vp=ng.H1(mesh,order=order+1,dirichlet='in|out')
        vo=ng.H1(mesh,order=order+1)
        def mask(space):return np.array(list(space.FreeDofs()),bool)
        fa,fc,ft,fp,fo=map(mask,(va,vc,vt,vp,vo))
        fo[0]=False
        def matrix(space,expression):
            form=ng.BilinearForm(space);form+=expression;form.Assemble();return csr(form)
        def own(space,expression,free):return matrix(space,expression)[free][:,free].tocsr()
        def cross(left,right,expression,fl,fr):
            mixed=left*right
            (u,v),(w,q)=mixed.TnT()
            return matrix(mixed,expression(u,q))[left.ndof:,:left.ndof][fr][:,fl].tocsr()
        a,w=va.TnT();ac,wc=vc.TnT();t,v=vt.TnT();p,q=vp.TnT();o,qo=vo.TnT()
        self.K=own(va,ng.curl(a)*ng.curl(w)/self.mu*measure,fa)
        self.M=own(va,self.sigma*a*w*measure,fa)
        kc=own(vc,ng.curl(ac)*ng.curl(wc)/self.mu*measure,fc)
        self.R=own(vt,ng.curl(t)*ng.curl(v)/self.sigma*measure,ft)
        self.U=own(vt,self.mu*t*v*measure,ft)
        s=own(vp,self.sigma*ng.grad(p)*ng.grad(q)*measure,fp)
        so=own(vo,self.mu*ng.grad(o)*ng.grad(qo)*measure,fo)
        cp=cross(va,vp,lambda a,q:self.sigma*a*ng.grad(q)*measure,fa,fp)
        co=cross(vt,vo,lambda t,q:self.mu*t*ng.grad(q)*measure,ft,fo)
        # D maps a curl(T) current to its vector-potential load.
        self.D=cross(vt,vc,lambda t,w:w*ng.curl(t)*measure,ft,fc)
        self.ga,da=gauge(va,self.K,fa);self.gt,dt=gauge(vt,self.R,ft)
        gc,dc=gauge(vc,kc,fc)
        ka_factor=factor(kc+gc)
        self.magnetic_subsolves=dict(count=0,max_original_residual=0.)
        def magnetic_solve(rhs):
            out=ka_factor(rhs)
            defect=float(np.linalg.norm(kc@out-rhs)/max(np.linalg.norm(rhs),1e-300))
            self.magnetic_subsolves['count']+=1
            self.magnetic_subsolves['max_original_residual']=max(
                self.magnetic_subsolves['max_original_residual'],defect)
            if not np.isfinite(defect) or defect>1e-8:
                raise RuntimeError(f'Original magnetic subsolve residual failed: {defect}')
            return out
        self.ka_solve=magnetic_solve
        phi_solve=factor(s);omega_solve=factor(so)
        self.mp=lambda x:self.M@x-cp.T@phi_solve(cp@x)
        self.lt=lambda x:self.U@x-co.T@omega_solve(co@x)
        self.lat=lambda x:self.D.T@self.ka_solve(self.D@x)
        def load(space,expr,free):
            form=ng.LinearForm(space);form+=expr;form.Assemble()
            return form.vec.FV().NumPy()[free].copy()
        self.f=load(va,self.sigma*w[2]*measure,fa)
        self.b=load(vt,ng.curl(v)[2]*measure,ft)
        self.G0=float(ng.Integrate(ng.CF(self.sigma)*measure,mesh))
        self.r0=1/(np.pi*.01**2*self.sigma*.01)
        self.diagnostics=[]
        self.metadata=dict(free_A=int(fa.sum()),free_compact_A=int(fc.sum()),free_T=int(ft.sum()),
                           free_phi=int(fp.sum()),free_Omega=int(fo.sum()),
                           kernel_defects=[da,dt,dc],backend='NGSolve sparsecholesky subsolves; matrix-free CG Schur solve',
                           mass_perturbation=0,scalar_order=order+1)
        self.metadata['magnetic_subsolves']=self.magnetic_subsolves

    def solver(self,family,s0):
        if family=='A-Phi':
            stiffness=self.K; mass=self.mp; gauge_matrix=self.ga
            pre=factor(self.K+s0*self.M+self.ga)
        else:
            stiffness=self.R; mass=self.lt if family=='T-Omega' else self.lat;gauge_matrix=self.gt
            pre=factor(self.R+s0*self.U+self.gt)
        size=stiffness.shape[0]
        op=LinearOperator((size,size),matvec=lambda x:stiffness@x+s0*mass(x)+gauge_matrix@x,dtype=float)
        pc=LinearOperator((size,size),matvec=pre,dtype=float)
        def solve(rhs):
            norm=np.linalg.norm(rhs)
            x,flag=cg(op,rhs/norm,M=pc,rtol=2e-13,atol=0,maxiter=3000)
            x*=norm
            defect=float(np.linalg.norm(stiffness@x+s0*mass(x)-rhs)/norm)
            self.diagnostics.append(dict(family=family,s0=float(s0),cg_flag=int(flag),true_residual=defect))
            if flag!=0 or not np.isfinite(defect) or defect>1e-8:
                raise RuntimeError(f'{family} original-system CG residual failed: {flag}, {defect}')
            return x
        return solve,mass

    def admittance_moments(self,family,z0,count=7):
        s0=z0/self.tau
        solve,mass=self.solver(family,s0)
        if family=='A-Phi':
            if z0==0:
                # Magnetic moments of the scalar-projected electric pencil.
                x=solve(self.f)
                y=[self.G0]
                for k in range(1,count):
                    y.append(-self.f@x/self.tau)
                    if k+1<count:x=-solve(mass(x)/self.tau)
            else:
                x=solve(self.f);y=[self.G0-s0*(self.f@x)]
                for _ in range(1,count):
                    previous=x;x=-solve(mass(x)/self.tau)
                    y.append(-s0*(self.f@x)-(self.f@previous)/self.tau)
        else:
            x=solve(self.b);y=[self.b@x]
            for _ in range(1,count):
                x=-solve(mass(x)/self.tau);y.append(self.b@x)
        return [float(v*self.r0) for v in y]
