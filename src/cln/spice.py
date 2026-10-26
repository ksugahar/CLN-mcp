"""Export a Cauer ladder (with expansion point s0) to an LTspice / SPICE netlist.

Two equivalent forms of the one-port Z(s) between nodes `p` and `n`:

* ``style="rl"``      : plain elements (inductors get Rser=0: LTspice otherwise adds 1 mOhm). Series R_{2k}; shunt branch (s - s0) L_{2k+1}
                        realized as inductor L in series with the negative resistor -s0 L.
* ``style="laplace"`` : one G element (VCCS) with I = V / Z(s), Z(s) as a continued fraction.

Values are scaled by ``length`` (the CLN elements are per unit length).
"""
import math
from numbers import Integral
import re


def _fmt(x):
    return f"{x:.9g}"


def ladder_netlist(R, L, s0=0.0, nmax=None, length=1.0, style="rl", name="CLN", p="p", n="0"):
    """Export positive elements with finite s0>=0 and simple SPICE identifiers."""
    for label, value, pattern in (("name", name, r"[A-Za-z][A-Za-z0-9_]{0,47}"),
                                  ("p", p, r"[A-Za-z0-9_]{1,48}"),
                                  ("n", n, r"[A-Za-z0-9_]{1,48}")):
        if not isinstance(value, str) or not re.fullmatch(pattern, value):
            raise ValueError(f"{label} must be a simple SPICE identifier (1..48 characters)")
    if p == n:
        raise ValueError("p and n must be different nodes")
    if style not in ("rl", "laplace"):
        raise ValueError("style must be 'rl' or 'laplace'")
    if not math.isfinite(s0) or s0 < 0:
        raise ValueError("s0 must be finite and nonnegative")
    if not math.isfinite(length) or length <= 0:
        raise ValueError("length must be finite and positive")
    for label, values, parity in (("R", R, 0), ("L", L, 1)):
        for index, value in values.items():
            if isinstance(index, bool) or not isinstance(index, Integral) or index < 0 or index % 2 != parity:
                raise ValueError(f"{label} has invalid ladder index {index!r}")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{label}[{index}] must be finite and positive")
    if not R and not L:
        raise ValueError("A nonempty ladder is required")
    if nmax is None:
        nmax = max(list(R) + list(L))
    if isinstance(nmax, bool) or not isinstance(nmax, Integral) or nmax < 0:
        raise ValueError("nmax must be a nonnegative integer")
    for index in range(nmax + 1):
        if index not in (R if index % 2 == 0 else L):
            raise ValueError(f"Missing ladder element at index {index}")
    lines = [f"* {name}: Cauer ladder, s0 = {_fmt(s0)} 1/s, nmax = {nmax}, length = {_fmt(length)} m"]
    if style == "laplace":
        z = f"{_fmt(R[nmax]*length)}" if nmax % 2 == 0 else f"((s-{_fmt(s0)})*{_fmt(L[nmax]*length)})"
        for i in range(nmax - 1, -1, -1):
            if i % 2 == 0:
                z = f"({_fmt(R[i]*length)}+{z})"
            else:
                z = f"(1/(1/((s-{_fmt(s0)})*{_fmt(L[i]*length)})+1/{z}))"
        lines.append(f"G_{name} {p} {n} {p} {n} Laplace=1/{z}")
        return "\n".join(lines) + "\n"
    if style != "rl":
        raise ValueError("style must be 'rl' or 'laplace'")
    node = p
    for i in range(0, nmax + 1):
        if i % 2 == 0:                                   # series resistor to the next node
            nxt = n if i == nmax else f"{name}_{i}"
            lines.append(f"R{name}{i} {node} {nxt} {_fmt(R[i]*length)}")
            node = nxt
        else:                                            # shunt (s-s0)L from node to ground
            if s0 > 0:
                mid = f"{name}_{i}m"
                lines.append(f"L{name}{i} {node} {mid} {_fmt(L[i]*length)} Rser=0")
                lines.append(f"R{name}{i}neg {mid} {n} {_fmt(-s0*L[i]*length)}")
            else:
                lines.append(f"L{name}{i} {node} {n} {_fmt(L[i]*length)} Rser=0")
    return "\n".join(lines) + "\n"
