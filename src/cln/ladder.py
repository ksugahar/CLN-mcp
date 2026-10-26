"""Cauer ladder impedance with expansion point s0."""


def ladder_impedance(R, L, s, s0=0.0, nmax=None):
    """Z(s) of the ladder R0 + 1/(1/((s-s0)L1) + 1/(R2 + ...)), truncated at stage nmax.

    R, L : dicts {even index: R_n}, {odd index: L_n}
    s    : complex or real Laplace variable (scalar or numpy array)
    """
    if nmax is None:
        nmax = max(list(R) + list(L))
    ds = s - s0
    z = R[nmax] if nmax % 2 == 0 else ds*L[nmax]
    for i in range(nmax - 1, -1, -1):
        if i % 2 == 0:
            z = R[i] + z
        else:
            zi = ds*L[i]
            z = zi*z/(zi + z)          # parallel; also valid at s = s0 (zi = 0)
    return z
