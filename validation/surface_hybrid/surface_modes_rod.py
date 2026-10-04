"""Round rod in a uniform TRANSVERSE field: CLN with surface modes for the air (Claude, 2026-10-03).

2-D A_z formulation in the rod cross-section (radius a, sigma, mu).  The air outside is linear and
static, so its influence on the rod is the exterior Dirichlet-to-Neumann map of the Laplace equation,
which on a circle is diagonal in Fourier modes: Lambda (cos m th, sin m th) = (m/a) (cos m th, sin m th).
Keeping the first K Fourier modes = K surface modes (each a rank-2 term); K = 0 is the conductor-only model.

    K_A = nu (grad, grad)_rod + nu0 sum_{m<=K} (m/a)/(pi a) (c_m c_m' + s_m s_m')     magnetic (s-independent)
    M_A = sigma (., .)_rod                                                              eddy (times s)
    f   = 2 H0 int_Gamma sin(th) v ds         (uniform applied field H0 = 1 A/m along x, A0 = mu0 H0 y)
    Z(s) = s f' (K_A + s M_A)^-1 f            "impedance" per metre seen by the source, current H0 = 1
exact:  Z(s) = s 2 pi a^2 mu0 (1 - I2(x)/I0(x)),  x = a sqrt(s mu sigma)   (mu = mu0)

The CLN recursions are Kameari's Type1 (A-Phi notebook) and Type2 (Type2_A notebook, current drive)
written in matrix form on (K_A, M_A, f); E = -s0 p, B = curl(q), Pj = s0^2 p'Mp, Wm = q'Kq.
A variant meshes an air annulus a < r < b (sigma = 0) with the K-mode DtN on r = b instead, to show that
surface modes on the rod surface reproduce an explicitly meshed air region.
usage: python surface_modes_rod.py <order> <h_over_a> <out.json>
"""
import sys, json, math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import mpmath as mp
from netgen.geom2d import SplineGeometry
from ngsolve import Mesh, H1, BilinearForm, LinearForm, x as X, y as Y, atan2, dx, ds, cos, sin

ORDER, HA, OUT = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
A_, SIG, MU0 = 0.01, 1e6, 4e-7*math.pi
S0 = 2*math.pi*1e4                       # expansion point (skin depth ~ 5 mm = a/2 at f0)
NU0 = 1/MU0
FREQS = [1e2, 1e3, 3e3, 1e4, 3e4, 1e5, 1e6]


def mesh_disk(b=None):
    g = SplineGeometry()
    if b is None:
        g.AddCircle((0, 0), A_, bc="gamma", leftdomain=1, rightdomain=0)
        g.SetMaterial(1, "rod")
    else:
        g.AddCircle((0, 0), A_, bc="interface", leftdomain=1, rightdomain=2)
        g.AddCircle((0, 0), b, bc="gamma", leftdomain=2, rightdomain=0)
        g.SetMaterial(1, "rod"); g.SetMaterial(2, "air")
    return Mesh(g.GenerateMesh(maxh=A_*HA)).Curve(ORDER)


def csr(form):
    i, j, v = form.mat.COO()
    n = form.space.ndof
    return sp.csr_matrix((np.array(v), (np.array(i), np.array(j))), shape=(n, n))


def operators(K, b=None):
    mesh = mesh_disk(b)
    R = A_ if b is None else b
    V = H1(mesh, order=ORDER)
    u, v = V.TnT()
    sig = mesh.MaterialCF({"rod": SIG}, default=0)
    k = BilinearForm(V); k += NU0*u.Deriv()*v.Deriv()*dx(bonus_intorder=4); k.Assemble()   # grad.grad
    m = BilinearForm(V); m += sig*u*v*dx(bonus_intorder=4); m.Assemble()
    th = atan2(Y, X)
    def trace_vec(cf):
        lf = LinearForm(V); lf += cf*v*ds("gamma", bonus_intorder=8); lf.Assemble()
        return np.array(lf.vec)
    Kmat, Mmat = csr(k), csr(m)
    low = np.zeros((V.ndof, 0)); w = []
    for mm in range(1, K + 1):
        for cf in (cos(mm*th), sin(mm*th)):
            low = np.column_stack([low, trace_vec(cf)]); w.append(NU0*(mm/R)/(math.pi*R))
    # applied field A0 = mu0 H0 r sin(th), H0 = 1: on r = R, nu0 (dA0/dr + Lambda A0) = 2 sin(th) for any R
    f = 2*trace_vec(sin(th))
    return Kmat, Mmat, low, np.array(w), f, V.ndof, mesh.ne


class Pencil:
    """K = Kmat + low diag(w) low'  (dense low-rank part kept separate);  solves via sparse LU + Woodbury."""
    def __init__(self, Kmat, Mmat, low, w):
        self.K, self.M, self.U, self.w = Kmat, Mmat, low, w

    def Kx(self, x):
        return self.K @ x + (self.U @ (self.w*(self.U.T @ x)) if self.U.shape[1] else 0)

    def solver(self, s):
        A = (self.K + s*self.M).tocsc()
        if not self.U.shape[1]:
            # conductor-only: K singular only on constants? no: grad-grad on a disk with Neumann has the constant
            # kernel; s*M (sigma > 0 on the whole rod) removes it for s != 0.
            lu = spla.splu(A); return lu.solve
        lu = spla.splu(A)
        Z = np.column_stack([lu.solve(self.U[:, j].astype(A.dtype)) for j in range(self.U.shape[1])])
        cap = np.diag(1/self.w) + self.U.T @ Z
        def solve(rhs):
            y = lu.solve(rhs.astype(A.dtype))
            return y - Z @ np.linalg.solve(cap, self.U.T @ y)
        return solve


def z_direct(P, f, s):
    return s*(f @ P.solver(s)(f.astype(complex)))


def type1(P, f, s0, nmax, short_port=True):
    """Kameari Type1 in matrix form; returns [L-1, R0, L1, R2, ...]."""
    a = P.solver(s0)
    x = a(f)
    aq = lambda u: u @ P.Kx(u) + s0*(u @ (P.M @ u))
    Lm = aq(x)
    out = [Lm]; xs = {-1: x}; Ls = {-1: Lm}; Rs = {}
    for n in range(0, nmax + 1):
        if n % 2 == 0:
            xn = xs[n-1]/(s0*Ls[n-1]) + (xs[n-2] if n >= 2 else 0)
            Rn = 1/(s0**2*(xn @ (P.M @ xn))); xs[n], Rs[n] = xn, Rn; out.append(Rn)
        else:
            y = a(-s0*Rs[n-1]*(P.M @ xs[n-1]))
            if short_port:
                # Type1 corrections are defined with the port SHORTED (V = 0). With a current-type source
                # (applied field H0) that means zero flux linkage f'x = 0, enforced by the source amplitude.
                y = y - (f @ y)/(f @ x)*x
            xn = y + (xs[n-2] if n > 1 else 0)
            Ln = aq(xn); xs[n], Ls[n] = xn, Ln; out.append(Ln)
    return out


def type2(P, f, s0, nmax):
    """Kameari Type2_A (current drive) in matrix form; p: E = -s0 p, q: B = curl q.  [R0, L1, R2, ...]."""
    a = P.solver(s0)
    x = a(f)
    Z0 = s0*(f @ x)
    p = q = x/Z0
    po = qo = None
    out = []
    for n in range(0, nmax + 1, 2):
        R = 1/(s0**2*(p @ (P.M @ p)) + s0*(q @ P.Kx(q))); out.append(R)
        if n + 1 > nmax:
            break
        po = R*p if po is None else po + R*p
        qo = R*q if qo is None else qo + R*q
        L = qo @ P.Kx(qo); out.append(L)
        if n + 2 > nmax:
            break
        xin = qo/(s0*L)
        y = a(P.Kx(xin) - P.Kx(q) - s0*(P.M @ p))
        p, q = p + y, q + y - xin
    return out


from ladder_eval import z_type1 as zt1, z_type2 as zt2   # noqa: E402  (shared evaluator, both terminating parities)


# analytic reference and exact Cauer elements
mp.mp.dps = 150
def Zex(s):
    s = mp.mpc(s); xx = A_*mp.sqrt(s*MU0*SIG)
    return s*2*mp.pi*A_**2*MU0*(1 - mp.besseli(2, xx)/mp.besseli(0, xx))
def _inv(u):
    r = [1/u[0]]
    for n in range(1, len(u)): r.append(-mp.fsum(u[i]*r[n-i] for i in range(1, n+1))/u[0])
    return r
def _cauer(c, count):
    o, z = [], list(c)
    for k in range(count):
        o.append(z[0] if k % 2 == 0 else 1/z[0]); z = z[1:]
        if k < count-1: z = _inv(z)
    return o
NEL = 9
exact = {"Type2": [float(mp.re(v)) for v in _cauer(mp.taylor(Zex, mp.mpf(S0), NEL+2), NEL)],
         "Type1": [float(mp.re(1/v)) for v in _cauer(mp.taylor(lambda s: s/Zex(s), mp.mpf(S0), NEL+2), NEL)]}

res = dict(order=ORDER, h_over_a=HA, s0=S0, exact_elements=exact, runs={})
for label, K, b in (("rod only, K=0 (conductor-only)", 0, None), ("rod, K=1 surface mode", 1, None),
                    ("rod, K=4 surface modes", 4, None), ("rod + meshed air to b=3a, DtN K=8 at b", 8, 3*A_)):
    Km, Mm, U, w, f, ndof, ne = operators(K, b)
    P = Pencil(Km, Mm, U, w)
    zh = {fr: complex(z_direct(P, f, 2j*math.pi*fr)) for fr in FREQS}
    zb = {fr: complex(Zex(2j*math.pi*fr)) for fr in FREQS}
    # with air meshed to r = b the port also links the applied field's static energy in a < r < b:
    # Y_b = Y_a + 2 pi mu0 (b^2 - a^2) exactly, so compare Z - s*Lair with the rod reaction Z
    Lair = 0.0 if b is None else 2*math.pi*MU0*(b**2 - A_**2)
    e1, e2 = type1(P, f, S0, NEL-2), type2(P, f, S0, NEL-1)
    run = dict(K=K, b=b, ndof=ndof, ne=ne,
               Lair_subtracted=Lair,
               Zh_vs_exact={str(fr): abs((zh[fr]-2j*math.pi*fr*Lair)/zb[fr]-1) for fr in FREQS},
               ladder_type1_vs_exact={str(fr): abs((zt1(e1, S0, 2j*math.pi*fr)-2j*math.pi*fr*Lair)/zb[fr]-1) for fr in FREQS},
               ladder_type2_vs_exact={str(fr): abs((zt2(e2, S0, 2j*math.pi*fr)-2j*math.pi*fr*Lair)/zb[fr]-1) for fr in FREQS},
               type1=dict(elements=e1, deviation=[e1[i]/exact["Type1"][i]-1 for i in range(NEL)],
                          vs_Zh={str(fr): abs(zt1(e1, S0, 2j*math.pi*fr)/zh[fr]-1) for fr in FREQS},
                          positive=all(v > 0 for v in e1)),
               type2=dict(elements=e2, deviation=[e2[i]/exact["Type2"][i]-1 for i in range(NEL)],
                          vs_Zh={str(fr): abs(zt2(e2, S0, 2j*math.pi*fr)/zh[fr]-1) for fr in FREQS},
                          positive=all(v > 0 for v in e2)))
    res["runs"][label] = run
    fmt = lambda d: " ".join(f"{v:.1e}" for v in d.values())
    print(f"== {label}: ndof {ndof}")
    print(f"   Z_h vs exact (f={FREQS}): {fmt(run['Zh_vs_exact'])}")
    print(f"   ladder vs exact: type1 {fmt(run['ladder_type1_vs_exact'])} | type2 {fmt(run['ladder_type2_vs_exact'])}")
    for t in ("type1", "type2"):
        r = run[t]
        print(f"   {t}: elem dev " + " ".join(f"{d:+.1e}" for d in r['deviation']) + f" | vs Z_h {fmt(r['vs_Zh'])} | pos={r['positive']}")
json.dump(res, open(OUT, "w"), indent=1)
