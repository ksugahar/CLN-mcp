import re
import numpy as np
import pytest
from cln.ladder import ladder_impedance
from cln.spice import ladder_netlist

R = {0: 1.5e-2, 2: 2.85e-2, 4: 2.89e-2}
L = {1: 1.12e-8, 3: 1.11e-8}
S0 = 2*np.pi*1e5


def _z_from_rl(net, s):
    """Tiny MNA: one-port impedance between p and 0 of an R/L netlist at complex s."""
    elems = [ln.split() for ln in net.splitlines() if ln and ln[0] in "RL"]
    nodes = sorted({x for e in elems for x in e[1:3]} - {"0"})
    idx = {nd: i for i, nd in enumerate(nodes)}
    Y = np.zeros((len(nodes), len(nodes)), complex)
    for name, a, b, v, *_ in elems:
        y = 1/float(v) if name[0] == "R" else 1/(s*float(v))
        for i, j, sg in ((a, a, 1), (b, b, 1), (a, b, -1), (b, a, -1)):
            if i != "0" and j != "0":
                Y[idx[i], idx[j]] += sg*y
    rhs = np.zeros(len(nodes), complex); rhs[idx["p"]] = 1.0
    return np.linalg.solve(Y, rhs)[idx["p"]]


@pytest.mark.parametrize("s0", [0.0, S0])
@pytest.mark.parametrize("nmax", [3, 4])
def test_rl_and_laplace_match_ladder(s0, nmax):
    for f in (1e3, 1e5, 1e6):
        s = 2j*np.pi*f
        z = ladder_impedance(R, L, s, s0, nmax)
        assert _z_from_rl(ladder_netlist(R, L, s0, nmax, style="rl"), s) == pytest.approx(z, rel=1e-6)
        expr = re.search(r"Laplace=(.*)$", ladder_netlist(R, L, s0, nmax, style="laplace")).group(1)
        assert eval(expr, {"s": s}) == pytest.approx(1/z, rel=1e-6)


@pytest.mark.parametrize("style", ["rl", "laplace"])
@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_export_rejects_unsupported_shift(style, bad):
    """Guards: no silently discarded shift. Assumes: nonnegative real CLN shift."""
    with pytest.raises(ValueError, match="s0"):
        ladder_netlist(R, L, s0=bad, style=style)


@pytest.mark.parametrize("field", ["name", "p", "n"])
@pytest.mark.parametrize("bad", ["bad\n.end", "bad name", "bad;directive", ""])
def test_export_rejects_netlist_identifier_injection(field, bad):
    """Guards: identifiers cannot emit extra tokens/lines. Assumes: simple nodes."""
    with pytest.raises(ValueError, match=field):
        ladder_netlist(R, L, **{field: bad})
