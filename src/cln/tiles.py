"""Tile representation of series-parallel (ladder) circuits.

A resistor R carrying current I at voltage E is drawn as a rectangle of width I and height E,
so its aspect ratio is E : I = R : 1.  Series connection stacks tiles vertically (same current);
parallel connection places them side by side (same voltage).  A whole network becomes one
rectangle whose height/width is its impedance, and the tile areas add up to the power E I.
Unit resistors are squares; tiling a p x q rectangle by squares is Euclid's algorithm, and the
square sizes are the continued-fraction digits of q/p (Sugahara, EMT2017-65).

A Cauer ladder evaluated at a real s > s0 is a resistive network, (s - s0) L acting as a
resistance, so it can be drawn in the same way.  The SVG output needs no plotting library.
"""
from dataclasses import dataclass
from fractions import Fraction
import html
import math
from numbers import Integral

MAX_TILES = 64

R_COLOR, L_COLOR, J_COLOR = "#F5B041", "#50A7C2", "#F5B041"


@dataclass
class Leaf:
    z: float                 # resistance (impedance at the real evaluation point)
    label: str = ""
    kind: str = "R"          # "R" (Joule) or "L" (inductive)
    joule_fraction: float = 1.0   # part of the tile drawn in the Joule colour (R tiles of a shifted ladder)

    def __post_init__(self):
        if not (isinstance(self.z, (int, float)) and math.isfinite(self.z) and self.z > 0):
            raise ValueError(f"tile {self.label or self.kind}: impedance must be finite and positive, got {self.z!r}")
        if not (0.0 <= self.joule_fraction <= 1.0):
            raise ValueError("joule_fraction must lie in [0, 1]")
        if self.kind not in ("R", "L"):
            raise ValueError("kind must be 'R' or 'L'")


@dataclass
class Series:
    parts: list


@dataclass
class Parallel:
    parts: list


def impedance(net):
    if isinstance(net, Leaf):
        return net.z
    if isinstance(net, Series):
        return sum(impedance(p) for p in net.parts)
    return 1/sum(1/impedance(p) for p in net.parts)


def layout(net, x=0.0, y=0.0, current=1.0):
    """Rectangles (x, y, w, h, leaf) for a network carrying `current`; y grows downward."""
    if isinstance(net, Leaf):
        return [(x, y, current, net.z*current, net)]
    out = []
    if isinstance(net, Series):
        for p in net.parts:
            out += layout(p, x, y, current)
            y += impedance(p)*current
        return out
    e = impedance(net)*current
    for p in net.parts:
        w = e/impedance(p)
        out += layout(p, x, y, w)
        x += w
    return out


def ladder_network(R, L, s, s0=0.0, nmax=None, ladder="type2", joule_fraction=None):
    """Network of a Cauer ladder at a real s.

    type2: Z = R0 + parallel((s-s0) L1, R2 + parallel((s-s0) L3, ...))   (R: dict even, L: dict odd)
    type1: Z = s * parallel(L[-1], R0/q + parallel(L1, R2/q + ...)), q = s - s0
           (elements as a list [L-1, R0, L1, R2, ...] passed in R, L ignored)
    joule_fraction: optional {index: Pj/(Pj + s0 Wm)} to split the R tiles of a shifted Type2 ladder.
    """
    if ladder not in ("type2", "type1"):
        raise ValueError("ladder must be 'type2' or 'type1'")
    if not (math.isfinite(s) and math.isfinite(s0)):
        raise ValueError("s and s0 must be finite")
    q = s - s0
    if q <= 0:
        raise ValueError("draw at a real s > s0, where every (s - s0) L is a positive resistance")
    if ladder == "type1" and s <= 0:
        raise ValueError("Type1 tiles need a positive real s")
    if ladder == "type2":
        nmax = max(list(R) + list(L)) if nmax is None else nmax
        missing = [i for i in range(nmax + 1) if i not in (R if i % 2 == 0 else L)]
        if missing:
            raise ValueError(f"Type2 needs R at even and L at odd indices 0..{nmax}; missing {missing}")
        if nmax + 1 > MAX_TILES:
            raise ValueError(f"at most {MAX_TILES} elements")
        def leaf(i):
            if i % 2 == 0:
                jf = 1.0 if joule_fraction is None else joule_fraction.get(i, 1.0)
                return Leaf(R[i], f"R{i}", "R", jf)
            return Leaf(q*L[i], f"L{i}", "L")
        net = leaf(nmax)
        for i in range(nmax - 1, -1, -1):
            net = Series([leaf(i), net]) if i % 2 == 0 else Parallel([leaf(i), net])
        return net
    if ladder == "type1":
        el = list(R)
        if not 1 <= len(el) <= MAX_TILES:
            raise ValueError(f"Type1 needs 1..{MAX_TILES} elements [L-1, R0, L1, ...]")
        lab = lambda k: ("L%d" % (k - 1)) if k % 2 == 0 else ("R%d" % (k - 1))
        def leaf(k):
            return Leaf(s*el[k], lab(k), "L") if k % 2 == 0 else Leaf(s*el[k]/q, lab(k), "R")
        net = leaf(len(el) - 1)
        for k in range(len(el) - 2, -1, -1):
            net = Parallel([leaf(k), net]) if k % 2 == 0 else Series([leaf(k), net])
        return net
    raise ValueError("ladder must be 'type2' or 'type1'")


def euclid_digits(p, q):
    """Continued-fraction digits of p/q (Euclid's quotients), without building anything."""
    digits = []
    while q:
        digits.append(p // q)
        p, q = q, p % q
    return digits


def euclid_network(p, q, max_tiles=MAX_TILES):
    """Unit-resistor series-parallel network of resistance p/q (positive integers).
    Its tiles are squares, and their sizes are the steps of Euclid's algorithm on (p, q).
    The number of tiles (sum of the quotients) is checked BEFORE any tile is built."""
    if not (isinstance(p, Integral) and isinstance(q, Integral) and not isinstance(p, bool)
            and not isinstance(q, bool) and p > 0 and q > 0):
        raise ValueError("p and q must be positive integers")
    ntiles = sum(euclid_digits(p, q))
    if ntiles > max_tiles:
        raise ValueError(f"{p}/{q} needs {ntiles} unit tiles; at most {max_tiles}")
    r = Fraction(p, q)
    def build(x, impedance_step):
        a = x.numerator // x.denominator
        rest = x - a
        units = [Leaf(1.0, "1", "R") for _ in range(a)]
        if rest == 0:
            return units[0] if a == 1 else (Series(units) if impedance_step else Parallel(units))
        tail = build(1/rest, not impedance_step)
        parts = units + [tail]
        return Series(parts) if impedance_step else Parallel(parts)
    # R = p/q: if R >= 1 start with series units, else start with parallel (admittance q/p).
    return build(r, True) if r >= 1 else build(1/r, False)


def svg(net, width_px=360, title="", labels=True, current=None, z_ref=None):
    """SVG string of the tile picture; the outer rectangle has height/width = impedance / z_ref.

    z_ref is the impedance unit of the drawing.  Default: 1 when `current` is given (Euclid's integer
    example), otherwise the network impedance itself, which makes the outer box a square.  Every
    tile keeps the aspect ratio resistance/z_ref."""
    z = impedance(net)
    cur = 1.0 if current is None else current
    zr = (1.0 if current is not None else z) if z_ref is None else z_ref
    rects = [(x, y/zr, w, h/zr, leaf) for x, y, w, h, leaf in layout(net, 0.0, 0.0, cur)]
    W, H = cur, z*cur/zr
    scale = width_px/W
    hp = H*scale
    pad, th = 8, (18 if title else 0)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_px + 2*pad:.0f}" height="{hp + 2*pad + th:.0f}" '
           f'viewBox="0 0 {width_px + 2*pad:.2f} {hp + 2*pad + th:.2f}" font-family="serif">']
    if title:
        out.append(f'<text x="{pad}" y="{pad + 12}" font-size="12">{html.escape(title)}</text>')
    oy = pad + th
    for x, y, w, h, leaf in rects:
        X, Y, Wp, Hp = pad + x*scale, oy + y*scale, w*scale, h*scale
        if leaf.kind == "R" and leaf.joule_fraction < 1.0:
            hj = Hp*leaf.joule_fraction
            out.append(f'<rect x="{X:.3f}" y="{Y:.3f}" width="{Wp:.3f}" height="{Hp - hj:.3f}" fill="{L_COLOR}"/>')
            out.append(f'<rect x="{X:.3f}" y="{Y + Hp - hj:.3f}" width="{Wp:.3f}" height="{hj:.3f}" fill="{J_COLOR}"/>')
        else:
            out.append(f'<rect x="{X:.3f}" y="{Y:.3f}" width="{Wp:.3f}" height="{Hp:.3f}" '
                       f'fill="{R_COLOR if leaf.kind == "R" else L_COLOR}"/>')
        out.append(f'<rect x="{X:.3f}" y="{Y:.3f}" width="{Wp:.3f}" height="{Hp:.3f}" fill="none" stroke="black" stroke-width="0.8"/>')
        if labels and leaf.label and min(Wp, Hp) > 14:
            fs = max(7, min(14, 0.35*min(Wp, Hp)))
            out.append(f'<text x="{X + Wp/2:.2f}" y="{Y + Hp/2 + fs/3:.2f}" font-size="{fs:.1f}" text-anchor="middle">{html.escape(leaf.label)}</text>')
    out.append(f'<rect x="{pad}" y="{oy}" width="{width_px}" height="{hp:.3f}" fill="none" stroke="black" stroke-width="1.5"/>')
    out.append("</svg>")
    return "\n".join(out)
