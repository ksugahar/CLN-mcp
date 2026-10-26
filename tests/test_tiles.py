import math
from fractions import Fraction
import pytest
from cln.tiles import Leaf, Series, Parallel, impedance, layout, ladder_network, euclid_network, svg
from cln import ladder_impedance, wire_cln_s0_zero


def test_series_parallel_tiles_conserve_power_and_aspect():
    net = Parallel([Leaf(1.0), Series([Leaf(2.0), Parallel([Leaf(1.0), Leaf(2.0)])])])
    z = impedance(net)
    assert z == pytest.approx(8/11)
    rects = layout(net)
    assert sum(w*h for *_, w, h, _ in [(r[0], r[1], r[2], r[3], r[4]) for r in rects]) == pytest.approx(z)  # area = E I, I = 1
    for x, y, w, h, leaf in rects:
        assert h/w == pytest.approx(leaf.z)                 # every tile has aspect R : 1


def test_euclid_16_over_22_gives_the_slide_squares():
    """EMT2017-65 slide 3-5: 16/22 = 1/(1 + 1/(2 + 1/(1 + 1/2))); squares 16, 6, 6, 4, 2, 2."""
    net = euclid_network(16, 22)
    assert impedance(net) == pytest.approx(16/22)
    rects = layout(net, current=22.0)                       # 22 wide, 16 tall
    sides = sorted((round(w, 9) for _, _, w, h, _ in rects), reverse=True)
    assert sides == [16, 6, 6, 4, 2, 2]
    assert all(abs(w - h) < 1e-9 for _, _, w, h, _ in rects)  # unit resistors are squares
    assert sum(w*h for _, _, w, h, _ in rects) == pytest.approx(22*16)


@pytest.mark.parametrize("p,q", [(1, 1), (3, 7), (22, 16), (89, 55)])
def test_euclid_network_resistance(p, q):
    assert impedance(euclid_network(p, q)) == pytest.approx(p/q)


def test_ladder_tiles_equal_ladder_impedance():
    a, sig, mu = .01, 1e6, 4e-7*math.pi
    R = {n: wire_cln_s0_zero(n, a, sig, mu) for n in range(0, 9, 2)}
    L = {n: wire_cln_s0_zero(n, a, sig, mu) for n in range(1, 9, 2)}
    s = 2*math.pi*1e4
    net = ladder_network(R, L, s)
    assert impedance(net) == pytest.approx(ladder_impedance(R, L, s, 0.0, 8))
    out = svg(net, title="wire, 9 elements")
    assert out.startswith("<svg") and out.count("<rect") >= 9


def test_shifted_type1_requires_s_above_s0():
    with pytest.raises(ValueError):
        ladder_network([1.0, 2.0, 1.0], {}, s=1.0, s0=2.0, ladder="type1")


def test_mcp_tiles_tool():
    import json
    from cln_mcp_server.server import cln_tiles, cln_knowledge
    out = json.loads(cln_tiles(euclid=[16, 22], title="16/22"))
    assert out["impedance"] == pytest.approx(16/22) and out["svg"].startswith("<svg") and len(out["tiles"]) == 6
    out = json.loads(cln_tiles(resistance=[1.0, 3.0], inductance=[0.5], s=2.0))
    assert out["impedance"] == pytest.approx(1 + 1/(1/1.0 + 1/3.0))
    assert "Euclid" in cln_knowledge("multiport", "euclid_tiles")
    assert "VCCS" in cln_knowledge("multiport", "multiport_y_vccs") or "voltage-controlled" in cln_knowledge("multiport", "multiport_y_vccs")


# Explicit rejection and resource caps are checked before construction.
def test_r7_unknown_ladder_rejected():
    from cln_mcp_server.server import cln_tiles
    with pytest.raises(ValueError, match="type2"):
        cln_tiles(resistance=[1.0, 2.0, 1.0], s=2.0, ladder="typo")
    with pytest.raises(ValueError):
        ladder_network({0: 1.0}, {}, s=1.0, ladder="typo")


def test_r8_euclid_cap_before_building(monkeypatch):
    import cln.tiles as T
    from cln_mcp_server.server import cln_tiles
    assert len(layout(euclid_network(64, 1))) == 64             # boundary: exactly the cap
    built = []
    monkeypatch.setattr(T, "Leaf", lambda *a, **k: built.append(1))
    for p, q in ((65, 1), (10**6, 1), (1, 10**6)):
        with pytest.raises(ValueError, match="at most 64"):
            T.euclid_network(p, q)
    assert built == []                                          # nothing allocated before the check
    monkeypatch.undo()
    with pytest.raises(ValueError):
        cln_tiles(euclid=[65, 1])
    with pytest.raises(ValueError):
        cln_tiles(euclid=[3, 2, 1])
    with pytest.raises(ValueError):
        euclid_network(0, 3)


@pytest.mark.parametrize("bad", [-1.0, 0.0, float("nan"), float("inf")])
def test_r9_nonpositive_or_nonfinite_elements_rejected(bad):
    from cln_mcp_server.server import cln_tiles
    res, ind = [1.0, bad], [0.5]
    with pytest.raises(ValueError):
        cln_tiles(resistance=res, inductance=ind, s=2.0)
    assert res == [1.0, bad] and ind == [0.5]                   # input not modified
    with pytest.raises(ValueError):
        cln_tiles(resistance=[2.0, bad, 1.0], s=2.0, ladder="type1")


def test_r9_type2_lengths_and_type1_positive_s():
    from cln_mcp_server.server import cln_tiles
    with pytest.raises(ValueError, match="alternates"):
        cln_tiles(resistance=[1.0], inductance=[0.5, 0.5, 0.5], s=2.0)
    with pytest.raises(ValueError):
        cln_tiles(resistance=[1.0, 2.0, 1.0], s=-3.0, s0=-4.0, ladder="type1")
