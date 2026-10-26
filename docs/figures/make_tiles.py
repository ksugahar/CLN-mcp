"""Regenerate the README tile figures (python docs/figures/make_tiles.py)."""
import math
from pathlib import Path
from cln import wire_cln_s0_zero
from cln.tiles import euclid_network, ladder_network, svg

here = Path(__file__).resolve().parent
(here/"tiles_euclid.svg").write_text(svg(euclid_network(16, 22), current=22, width_px=330,
                                         title="16/22 ohm: Euclid (22,16) = squares 16, 6, 6, 4, 2, 2"), encoding="utf-8")
a, sigma, mu = 0.01, 1e6, 4e-7*math.pi
R = {n: wire_cln_s0_zero(n, a, sigma, mu) for n in range(0, 9, 2)}
L = {n: wire_cln_s0_zero(n, a, sigma, mu) for n in range(1, 9, 2)}
(here/"tiles_wire.svg").write_text(svg(ladder_network(R, L, 2*math.pi*3e3), width_px=330,
                                       title="round-wire CLN, 9 elements, s = 2 pi 3 kHz"), encoding="utf-8")
