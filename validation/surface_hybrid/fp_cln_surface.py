"""FP-CLN with surface modes: nonlinear iron rod in a transverse time-varying field (Claude, 2026-10-03).

2-D A_z.  Rod radius a, sigma; saturating BH law H(B) = nu0 B - (nu0 - nu_i) Bs tanh(B/Bs) (explicit;
differential reluctivity between nu_i and nu0, so the FP split converges for nu_FP > nu0/2, i.e. mu_r^FP ~ 2,
for this stated constitutive law. FP iterations are Anderson-accelerated.
Air outside: linear, static -> condensed exactly onto the rod surface (Schur complement S of an air
annulus to r = b with the exact Fourier DtN on r = b), or replaced by K Steklov surface modes.
Fixed-point split (FP-CLN): nu(B) -> constant nu_FP on the left; the residual
    g(A) = ((nu(|grad A|^2) - nu_FP) grad A, grad v)_rod
goes to the right as a controlled source.  Backward Euler in time, FP iteration at every step.
Linear operator (constant):  K = nu_FP K_rod + S_surface,  M = sigma (.,.)_rod,  source f u(t) with
u(t) = H0 sin(w t), flux linkage lambda(t) = f' A + c0 u.

Reference: the same FP scheme on all rod dofs (no reduction).
FP-CLN:    the independent CLN potential basis of the linear pencil (Kameari Type1 recursion at s0, corrections with the port
           shorted) -> Galerkin reduction; the ladder circuit is this reduced model in its own basis.
usage: python fp_cln_surface.py <order> <h_over_a> <out.json>
"""
import sys, json, math, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import eigh
from netgen.geom2d import SplineGeometry
from ngsolve import (Mesh, H1, BilinearForm, LinearForm, GridFunction, x as X, y as Y, atan2, dx, ds,
                     cos, sin, exp, sqrt, grad, InnerProduct, SetNumThreads)
tanh = lambda w: 1 - 2/(exp(2*w) + 1)            # NGSolve has no tanh CF
SetNumThreads(4)
ORDER, HA, OUT = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
A_, SIG, MU0 = 0.01, 2e6, 4e-7*math.pi
NU0, B_, KB = 1/MU0, 0.05, 40
MUR_I, BS = 200.0, 1.5                           # initial relative permeability, saturation scale [T]
NUI = NU0/MUR_I
MUR_FP = float(sys.argv[4]) if len(sys.argv) > 4 else 1.8
NUFP = NU0/MUR_FP                                # FP reluctivity: contraction needs nu_FP > nu0/2
F_HZ, H0AMP = 50.0, 6.0e5
W = 2*math.pi*F_HZ
NSTEP_PER, NPER = 200, 2
DT = 1/(F_HZ*NSTEP_PER)
S0 = W                                           # CLN expansion point
TOL = 1e-8

# ---------------- mesh and operators (rod + air annulus, air condensed onto Gamma) ----------------
g2 = SplineGeometry()
g2.AddCircle((0, 0), A_, bc="gamma", leftdomain=1, rightdomain=2, maxh=A_*HA)
g2.AddCircle((0, 0), B_, bc="outer", leftdomain=2, rightdomain=0)
g2.SetMaterial(1, "rod"); g2.SetMaterial(2, "air"); g2.SetDomainMaxH(1, A_*HA)
mesh = Mesh(g2.GenerateMesh(maxh=B_/6)).Curve(ORDER)
V = H1(mesh, order=ORDER)
u, v = V.TnT()
def mat(form):
    bf = BilinearForm(V); bf += form; bf.Assemble()
    i, j, w = bf.mat.COO()
    return sp.csr_matrix((np.array(w), (np.array(i), np.array(j))), shape=(V.ndof, V.ndof))
Krod = mat(grad(u)*grad(v)*dx("rod", bonus_intorder=4))
Kair = mat(NU0*grad(u)*grad(v)*dx("air", bonus_intorder=4))
Mfull = mat(SIG*u*v*dx("rod", bonus_intorder=4))
Bg_full = mat(u*v*ds("gamma", bonus_intorder=6))
th = atan2(Y, X)
def lin(cf, region):
    lf = LinearForm(V); lf += cf*v*ds(region, bonus_intorder=8); lf.Assemble(); return np.array(lf.vec)
Uo = np.array([lin(c, "outer") for m in range(1, KB + 1) for c in (cos(m*th), sin(m*th))]).T
wo = np.array([NU0*(m/B_)/(math.pi*B_) for m in range(1, KB + 1) for _ in (0, 1)])
f_out = 2*lin(sin(th), "outer")
def region_dofs(name):
    mark = np.zeros(V.ndof, bool)
    for el in V.Elements():
        if el.mat == name:
            mark[list(el.dofs)] = True
    return mark
in_rod, in_air = region_dofs("rod"), region_dofs("air")
Gm = in_rod & in_air
I, G, R = np.where(in_air & ~Gm)[0], np.where(Gm)[0], np.where(in_rod)[0]
KII = Kair[I][:, I].tocsc(); KIG = Kair[I][:, G].toarray(); KGG = Kair[G][:, G].toarray(); UI = Uo[I]
luI = spla.splu(KII)
ZcI = np.column_stack([luI.solve(UI[:, j]) for j in range(UI.shape[1])]); capI = np.diag(1/wo) + UI.T @ ZcI
def KIIinv(r):
    y = np.column_stack([luI.solve(r[:, j]) for j in range(r.shape[1])]) if r.ndim == 2 else luI.solve(r)
    return y - ZcI @ np.linalg.solve(capI, UI.T @ y)
S = KGG - KIG.T @ KIIinv(KIG); S = (S + S.T)/2
yI = KIIinv(f_out[I]); fG = -KIG.T @ yI; c0 = float(f_out[I] @ yI)
Bgg = Bg_full[G][:, G].toarray()
lam, Vs = eigh(S, Bgg); o = np.argsort(lam); lam, Vs = lam[o], Vs[:, o]
nzm = lam > 1e-9*lam.max(); lam, Vs = lam[nzm], Vs[:, nzm]
pos = {d: k for k, d in enumerate(R)}; Gr = np.array([pos[d] for d in G])
Kr = (NUFP*Krod[R][:, R]).tocsr(); Mr = Mfull[R][:, R].tocsr()
fr = np.zeros(len(R)); fr[Gr] = fG
nR = len(R)


def surface_K(kind):
    """dense nR x nR surface operator (stored only on Gamma) for: 'exact', 'none', or int K Steklov modes."""
    Sx = np.zeros((len(G), len(G)))
    if kind == "exact":
        Sx = S
    elif kind != "none":
        BV = Bgg @ Vs[:, :kind]; Sx = BV @ np.diag(lam[:kind]) @ BV.T
    return Sx


def Kfull_matrix(Sx):
    Ka = Kr.tolil(); Ka[np.ix_(Gr, Gr)] = Ka[np.ix_(Gr, Gr)] + Sx; return Ka.tocsr()


# ---------------- nonlinear residual g(A) on the rod ----------------
gfA = GridFunction(V)
gv = grad(gfA)
b2 = InnerProduct(gv, gv)
bb = sqrt(b2 + 1e-30)
nu_B = NU0 - (NU0 - NUI)*BS*tanh(bb/BS)/bb
lf_g = LinearForm(V); lf_g += (nu_B - NUFP)*gv*grad(v)*dx("rod", bonus_intorder=4)
def g_of(Arod):
    vec = gfA.vec.FV().NumPy(); vec[:] = 0; vec[R] = Arod
    lf_g.Assemble(); return np.array(lf_g.vec)[R]
def bmax(Arod):
    vec = gfA.vec.FV().NumPy(); vec[:] = 0; vec[R] = Arod
    return max(math.sqrt(max(b2(mesh(px, py)), 0)) for px, py in ((0, 0), (A_*0.99, 0), (0, A_*0.99), (A_*0.7, A_*0.7)))


from fixed_point import anderson as _anderson   # raises FixedPointNotConverged after maxit


def anderson(Gfun, x0, m=6, maxit=400, tol=TOL):
    return _anderson(Gfun, x0, m=m, maxit=maxit, tol=tol)


# ---------------- time stepping ----------------
tt = DT*np.arange(1, NSTEP_PER*NPER + 1)
uu = H0AMP*np.sin(W*tt)


def run_full(Sx):
    Ka = Kfull_matrix(Sx); lu = spla.splu((Ka + Mr/DT).tocsc())
    A = np.zeros(nR); lam_t, its = [], []; joule=0.
    for step,un in enumerate(uu):
        previous=A.copy()
        rhs0 = fr*un + Mr @ A/DT
        A, k = anderson(lambda Ak: lu.solve(rhs0 - g_of(Ak)), A)
        its.append(k); lam_t.append(fr @ A + c0*un)
        if step>=NSTEP_PER:
            change=A-previous; joule+=float(change@(Mr@change))/DT
    return np.array(lam_t), its, A, joule



BASIS_AUDIT=[]

def energy_basis(columns, Ka):
    """Keep only independent field directions in the fixed linear energy norm."""
    energy=(Ka+S0*Mr).toarray()
    c=np.column_stack(columns)
    norms=np.sqrt(np.einsum('ij,ij->j',c,energy@c))
    if np.any(~np.isfinite(norms)) or np.any(norms<=0):
        raise RuntimeError('Nonpositive or nonfinite mode energy')
    chol=np.linalg.cholesky((energy+energy.T)/2)
    left,sv,_=np.linalg.svd(chol.T@(c/norms),full_matrices=False)
    rank=int(np.sum(sv>sv[0]*1e-11))
    BASIS_AUDIT.append({'columns':c.shape[1],'independent_rank':rank,
                        'relative_singular_values':(sv/sv[0]).tolist()})
    return np.linalg.solve(chol.T,left[:,:rank])


def cln_basis(Sx, n):
    """Kameari Type1 vectors (port shorted) of the linear pencil at s0, rank-revealed in the fixed energy metric for Galerkin."""
    Ka = Kfull_matrix(Sx); lu = spla.splu((Ka + S0*Mr).tocsc()); a = lambda r: lu.solve(r)
    x = a(fr); vecs = [x]; Ls = {-1: x @ (Ka @ x) + S0*(x @ (Mr @ x))}; xs = {-1: x}; Rs = {}
    for k in range(0, n):
        if k % 2 == 0:
            xn = xs[k-1]/(S0*Ls[k-1]) + (xs[k-2] if k >= 2 else 0); Rs[k] = 1/(S0**2*(xn @ (Mr @ xn)))
        else:
            y = a(-S0*Rs[k-1]*(Mr @ xs[k-1])); y = y - (fr @ y)/(fr @ x)*x
            xn = y + (xs[k-2] if k > 1 else 0); Ls[k] = xn @ (Ka @ xn) + S0*(xn @ (Mr @ xn))
        xs[k] = xn; vecs.append(xn)
    Q = energy_basis(vecs, Ka)
    return Q, Ka


def run_cln(Sx, n):
    Q, Ka = cln_basis(Sx, n)
    Kq, Mq, fq = Q.T @ (Ka @ Q), Q.T @ (Mr @ Q), Q.T @ fr
    Aq = np.linalg.inv(Kq + Mq/DT)
    y = np.zeros(Q.shape[1]); lam_t, its = [], []; joule=0.
    for step,un in enumerate(uu):
        previous=y.copy()
        rhs0 = fq*un + Mq @ y/DT
        y, k = anderson(lambda yk: Aq @ (rhs0 - Q.T @ g_of(Q @ yk)), y)
        its.append(k); lam_t.append(fr @ (Q @ y) + c0*un)
        if step>=NSTEP_PER:
            change=y-previous; joule+=float(change@(Mq@change))/DT
    return np.array(lam_t), its, Q @ y, joule


def run_cln_multiport(Sx, n, extra_modes, nk):
    """Port-1 CLN basis (n+1 candidates, filtered by energy rank) + a block Krylov basis for extra PORTS = Steklov surface modes
    (additional angular field directions); nk vectors per extra port."""
    Q1, Ka = cln_basis(Sx, n)
    lu = spla.splu((Ka + S0*Mr).tocsc())
    cols = [Q1]
    for j in extra_modes:
        fe = np.zeros(nR); fe[Gr] = Bgg @ Vs[:, j]
        w = lu.solve(fe); blk = [w]
        for _ in range(nk - 1):
            w = lu.solve(Mr @ w); blk.append(w)
        cols.append(np.column_stack(blk))
    Q = energy_basis(cols, Ka)
    Kq, Mq, fq = Q.T @ (Ka @ Q), Q.T @ (Mr @ Q), Q.T @ fr
    Aq = np.linalg.inv(Kq + Mq/DT)
    y = np.zeros(Q.shape[1]); lam_t, its = [], []; joule=0.
    for step,un in enumerate(uu):
        previous=y.copy()
        rhs0 = fq*un + Mq @ y/DT
        y, k = anderson(lambda yk: Aq @ (rhs0 - Q.T @ g_of(Q @ yk)), y)
        its.append(k); lam_t.append(fr @ (Q @ y) + c0*un)
        if step>=NSTEP_PER:
            change=y-previous; joule+=float(change@(Mq@change))/DT
    return np.array(lam_t), its, Q.shape[1], joule


def metrics(lt, ref, joule, joule_ref):
    # Include both endpoints of cycle 2. Trapezoidal u dlambda cancels the
    # reversible static-air term exactly on a closed input cycle.
    sec=slice(NSTEP_PER-1,None)
    amp=float(np.abs(ref[sec]).max())
    rod_ref=ref-c0*uu; rod=lt-c0*uu
    rod_amp=float(np.abs(rod_ref[sec]).max())
    work=lambda values:float(np.sum(.5*(uu[sec][1:]+uu[sec][:-1])*np.diff(values[sec])))
    assert np.isfinite(joule) and joule>0 and joule_ref>0
    return dict(max_rel_err=float(np.abs(lt[sec]-ref[sec]).max()/amp),
                rod_linkage_relative_error=float(np.abs(rod[sec]-rod_ref[sec]).max()/rod_amp),
                input_cycle_work=work(lt),input_cycle_work_reference=work(ref),
                cycle_work_relative_error=abs(work(lt)-work(ref))/abs(work(ref)),
                joule_cycle=joule,joule_reference=joule_ref,
                joule_relative_error=abs(joule-joule_ref)/joule_ref)



res = dict(order=ORDER, h_over_a=HA, nR=nR, nG=len(G), sigma=SIG, nu_fp=NUFP, f=F_HZ, H0=H0AMP, dt=DT,
           bh=dict(mur_i=MUR_I, Bs=BS, mur_fp=MUR_FP), s0=S0, runs={})
t0 = time.time()
ref, its_ref, Aend, joule_ref = run_full(surface_K("exact"))
res["reference"] = dict(lambda_trace=ref.tolist(),joule_cycle=joule_ref,fp_iterations_mean=float(np.mean(its_ref)), fp_iterations_max=int(np.max(its_ref)),
                        B_probe_end=bmax(Aend), seconds=time.time()-t0)
print(f"reference (exact surface, all {nR} rod dofs): FP its mean {np.mean(its_ref):.1f} max {max(its_ref)}, |B| probe at end {bmax(Aend):.2f} T, {time.time()-t0:.0f}s", flush=True)
for kind in ("exact",):
    if kind != "exact":
        lt, its, _, joule = run_full(surface_K(kind))
        m = metrics(lt, ref, joule, joule_ref); m.update(fp_its=float(np.mean(its)), fp_its_max=int(np.max(its)))
        res["runs"][f"full, surface={kind}"] = m
        print(f"full model, surface={kind!s:5s}: lambda err {m['max_rel_err']:.2e}, Joule err {m['joule_relative_error']:.2e}", flush=True)
    for n in (4, 8, 12, 16):
        lt, its, _, joule = run_cln(surface_K(kind), n)
        m = metrics(lt, ref, joule, joule_ref); m.update(fp_its=float(np.mean(its)), fp_its_max=int(np.max(its)), basis=BASIS_AUDIT[-1]['independent_rank'],requested_basis=n+1,lambda_trace=lt.tolist())
        res["runs"][f"FP-CLN n={n+1}, surface={kind}"] = m
        print(f"FP-CLN candidates {n+1:2d}, independent {m['basis']:2d}, surface={kind!s:5s}: lambda err {m['max_rel_err']:.2e}, Joule err {m['joule_relative_error']:.2e}, FP its {np.mean(its):.1f}", flush=True)
for extra, nk in (([2, 3, 4, 5], 3), ([2, 3, 4, 5, 6, 7, 8, 9], 6)):
    lt, its, nb, joule = run_cln_multiport(surface_K("exact"), 12, extra, nk)
    m = metrics(lt, ref, joule, joule_ref); m.update(fp_its=float(np.mean(its)), fp_its_max=int(np.max(its)), basis=nb, extra_ports=extra, per_port=nk,lambda_trace=lt.tolist())
    res["runs"][f"multiport FP-CLN: port1 13 + Steklov ports {extra} x {nk}"] = m
    print(f"multiport FP-CLN (13 base candidates + {len(extra)} ports x {nk}; independent rank {nb}), surface=exact: lambda err {m['max_rel_err']:.2e}, Joule err {m['joule_relative_error']:.2e}, FP its {np.mean(its):.1f}", flush=True)
res['basis_audit']=BASIS_AUDIT
res['time']=tt.tolist(); res['input']=uu.tolist(); res['static_air_coefficient']=c0
res['reference']['relative_cycle_change']=float(np.max(abs(ref[NSTEP_PER:]-ref[:NSTEP_PER]))/np.max(abs(ref[NSTEP_PER:])))
res['joule_definition']='sum dt * (delta A / dt)^T M (delta A / dt), cycle 2, backward Euler field derivative'
res['work_definition']='trapezoidal H d(lambda) over both cycle endpoints; not assumed steady-state Joule loss'
res['scope']='Nonlinear 2D rod, same discrete full-model/BE reference; not a nonlinear Foster comparison'
res['review']='Codex self-review; Claude independent review pending'
with open(OUT,'w',encoding='utf-8') as fp:
    json.dump(res,fp,indent=1)
