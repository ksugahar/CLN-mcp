"""Exact illustrative port identities; no FE or general passivity certificate.

Independence: wire fields follow Maxwell integrations; a separate Bessel
power series checks the terminal continued fraction. Boundary port identities
are checked against direct matrix inversion, not a second ladder recurrence.
"""
import argparse
import hashlib
import json
from pathlib import Path

import sympy as sp


def zero(actual, expected, label):
    difference = sp.simplify(sp.cancel(actual - expected))
    if difference != 0:
        raise AssertionError(f"{label}: actual={actual}, expected={expected}, delta={difference}")


def matched_elements(k, n, lift, s, zh):
    """Compare constrained B and full-space H recurrences with scalar extraction.

    The two recurrences are consistency checks of matched ports; rational
    extraction uses the independently inverted full terminal transfer.
    """
    free = k[1:, 1:]
    energy = lambda x, a: sp.simplify((x.T * a * x)[0])
    c = energy(lift, k)
    eb, ab = lift, sp.zeros(k.rows, 1)
    rb = 1 / energy(eb, n)
    b_elements = [("L", 1 / c), ("R", rb)]
    for _ in range(k.rows - 1):
        load = (n * eb)[1:, :]
        ab = ab + sp.Matrix([0, *(free.inv() * rb * load)])
        lb = energy(ab, k)
        if lb.is_positive is not True:
            raise AssertionError(f"B-mode energy: actual={lb}, expected > 0")
        eb = sp.simplify(eb - ab / lb)
        rb = sp.simplify(1 / energy(eb, n))
        b_elements.extend([("L", lb), ("R", rb)])
    ah = k.inv() * sp.Matrix([1, *([0] * (k.rows - 1))])
    eh = sp.zeros(k.rows, 1)
    lh = energy(ah, k)
    h_elements = [("L", lh)]
    for index in range(k.rows):
        eh = sp.simplify(eh - ah / lh)
        rh = sp.simplify(1 / energy(eh, n))
        h_elements.append(("R", rh))
        if index + 1 < k.rows:
            ah = sp.simplify(k.inv() * (k * ah + rh * n * eh))
            lh = energy(ah, k)
            h_elements.append(("L", lh))
    # Scalar Laurent/DC extraction, independent of either field recurrence.
    tail, extracted = zh, []
    for _ in range(k.rows):
        inductance = sp.simplify(1 / sp.limit(s / tail, s, 0))
        tail = sp.cancel(1 / (1 / tail - 1 / (s * inductance)))
        resistance = sp.limit(tail, s, 0)
        extracted.extend([("L", inductance), ("R", resistance)])
        tail = sp.cancel(tail - resistance)
    zero(tail, 0, "exact scalar extraction terminates")
    for index, ((kind_b, bv), (kind_h, hv), (kind_x, xv)) in enumerate(zip(b_elements, h_elements, extracted)):
        if not kind_b == kind_h == kind_x:
            raise AssertionError(f"Element {index}: kinds {kind_b}, {kind_h}, {kind_x}")
        zero(bv, hv, f"B/H index-shift element {index}")
        zero(bv, xv, f"recurrence vs independent scalar extraction {index}")
    return {"B_labels": ["L_-1", "R_0", "L_1", "R_2", "L_3", "R_4"],
            "H_labels": ["L_1", "R_2", "L_3", "R_4", "L_5", "R_6"],
            "B_elements": b_elements, "H_elements": h_elements,
            "independent_scalar_elements": extracted}


def port_checks():
    s = sp.Symbol("s")
    k = sp.Matrix([[4, 1, 1], [1, 3, 1], [1, 1, 2]])
    n = sp.diag(1, 2, 3)
    kff, nff = k[1:, 1:], n[1:, 1:]
    lift = sp.Matrix([1, *(-kff.inv() * k[1:, :1])])
    c = (lift.T * k * lift)[0]
    m = (lift.T * n * lift)[0]
    load = (n * lift)[1:, :]
    correction = -(kff + s * nff).inv() * s * load
    total = lift + sp.Matrix([0, *correction])
    residual = (k + s * n) * total
    for value in residual[1:]:
        zero(value, 0, "homogeneous free equations")
    zero((lift.T * k * sp.Matrix([0, *correction]))[0], 0, "static lift orthogonality")
    yb = sp.cancel(residual[0] / s)
    ye = sp.cancel(m - s * (load.T * (kff + s * nff).inv() * load)[0])
    zero(yb, c / s + ye, "boundary reaction / lift decomposition")
    port = sp.Matrix([1, 0, 0])
    zh = sp.cancel(s * (port.T * (k + s * n).inv() * port)[0])
    zero(zh * yb, 1, "same conjugate normalized port duality")
    matched = matched_elements(k, n, lift, s, zh)
    literal_full = sp.cancel((lift.T * k * (k + s * n).inv() * n * lift)[0])
    compact_gap = sp.cancel(literal_full - ye)
    if compact_gap == 0:
        raise AssertionError("Full-space compact-product counterexample collapsed")
    # A common boundary lift reused as an arbitrary RHS in the free space is
    # a different port problem; the static excitation inductance need not be L1.
    free_rhs = sp.Matrix([1, 1])
    l1 = (free_rhs.T * kff.inv() * free_rhs)[0]
    if sp.cancel(l1 - 1 / c) == 0:
        raise AssertionError("illustrative different-port counterexample collapsed")
    # Completing the square proves Re(s^-1 x* K x + x* N x) >= 0 for
    # Re(s)>0. Exact Sylvester minors establish definiteness in this example.
    minors = {name: [int(a[:i, :i].det()) for i in range(1, 4)]
              for name, a in (("K", k), ("N", n))}
    if not all(v > 0 for values in minors.values() for v in values):
        raise AssertionError(f"SPD leading minors: actual={minors}, expected all > 0")
    return {"K": [list(k.row(i)) for i in range(3)], "N": [list(n.row(i)) for i in range(3)],
            "static_lift": list(lift), "static_energy_c": c,
            "L_excitation": 1 / c, "different_port_L1": l1,
            "Y_B": yb, "Y_E_response": ye, "Z_H_same_port": zh,
            "positive_leading_minors": minors, "matched_index_shift": matched,
            "literal_full_compact_product": literal_full, "literal_full_minus_response": compact_gap,
            "identities": ["free residual=0", "lift K-orthogonality=0", "Y_B=c/s+Y_E", "Y_B Z_H=1"]}


def wire_checks():
    r, z = sp.symbols("r z")
    # Dimensionless r/a, mu=sigma=a=1; impedance is per unit axial length.
    weight = 2 * sp.pi * r
    inner = lambda f, g: sp.integrate(f * g * weight, (r, 0, 1))
    ampere = lambda e: sp.cancel(sp.integrate(e * r, (r, 0, r)) / r)
    faraday = lambda h: sp.integrate(h, (r, 1, r))
    e, h, rs, ls = [sp.Integer(1)], [], [], []
    for index in range(5):
        rs.append(sp.cancel(1 / inner(e[-1], e[-1])))
        if index == 4:
            break
        magnetic = sp.expand(ampere(rs[-1] * e[-1]) + (h[-1] if h else 0))
        h.append(magnetic)
        ls.append(sp.cancel(inner(magnetic, magnetic)))
        e.append(sp.expand(e[-1] + faraday(magnetic / ls[-1])))
    for fields, label in ((e, "electric"), (h, "magnetic")):
        for i, fi in enumerate(fields):
            for fj in fields[:i]:
                zero(inner(fi, fj), 0, f"{label} energy orthogonality")
    for i, magnetic in enumerate(h):
        a = -faraday(magnetic)
        zero(a.subs(r, 1), 0, "homogeneous A trace")
        zero(-sp.diff(a, r), magnetic, "A to H")
        # T is the integrated current of this step, before the previous H is added.
        t = ampere(e[i])
        zero(sp.diff(r * t, r) / r, e[i], "T to E")
        zero(rs[i] * t + (h[i-1] if i else 0), magnetic, "T to H")
        zero(e[i+1].subs(r, 1), 1, "accumulated electric trace")
    impedance = rs[-1]
    for index in reversed(range(len(ls))):
        impedance = sp.cancel(rs[index] + 1 / (1 / (z * ls[index]) + 1 / impedance))
    normalized = sp.cancel(sp.pi * impedance)
    # Independent analytic oracle: I0(sqrt(z))/(2 I1(sqrt(z))/sqrt(z)).
    numerator = sum((z / 4) ** j / sp.factorial(j) ** 2 for j in range(10))
    denominator = sum((z / 4) ** j / (sp.factorial(j) * sp.factorial(j + 1)) for j in range(10))
    oracle = sp.series(numerator / denominator, z, 0, 9).removeO()
    zero(sp.series(normalized, z, 0, 9).removeO(), oracle, "Bessel contact through z^8")
    # Exact zeros signal termination, not a positive coefficient to be manufactured.
    if not all(v.is_positive for v in rs + ls):
        raise AssertionError(f"Wire elements: actual={(rs, ls)}, expected all > 0")
    return {"normalization": "a=mu=sigma=1; Z per unit length; z=s*mu*sigma*a^2",
            "E_even": e, "H_odd": h, "R_even": rs, "L_odd": ls,
            "Z_over_Rdc": normalized, "Bessel_series_through_z8": oracle,
            "identities": ["energy orthogonality", "A/H and T/E field maps", "Bessel contact through z^8"]}


def caveat_checks():
    r, s = sp.symbols("r s", positive=True)
    # For Jz=1 the regular axis fixes Htheta=r/2. Subtracting a constant
    # to impose Htheta(1)=0 violates Ampere; the homogeneous solution is C/r.
    imposed = (r - 1) / 2
    defect = sp.simplify(sp.diff(r * imposed, r) / r - 1)
    zero(defect, -1 / (2 * r), "incompatible arbitrary magnetic boundary")
    zero(sp.diff(r * (r / 2), r) / r, 1, "regular Ampere solution")
    # Neither choosing a polynomial remainder sign nor positive coefficients
    # in the shifted variable can certify positive-realness in physical s.
    shifted_branch = s - 1  # q L, with s0=L=1
    zero(shifted_branch.subs(s, 0), -1, "negative physical DC of shifted branch")
    quotient, remainder = sp.div(s - 1, s + 1, s)
    zero(quotient, 1, "Euclidean quotient")
    zero(remainder, -2, "Euclidean remainder need not be positive")
    return {"regular_Ampere_H": r / 2, "incompatible_adjusted_H": imposed,
            "Ampere_defect": defect, "shifted_branch": shifted_branch,
            "shifted_branch_physical_DC": -1, "Euclidean_remainder": remainder}


def run():
    return {"schema": "cln.input-ports-exact.v1", "sympy_version": sp.__version__,
            "source_sha256_lf": hashlib.sha256(Path(__file__).read_text(encoding="utf-8-sig").encode("utf-8")).hexdigest(),
            "source_hash_convention": "UTF-8 without BOM; universal newlines normalized to LF; final newline preserved",
            "scope": "exact toy port and normalized round-wire illustration; no FE, shifted synthesis or general theorem",
            "port": port_checks(), "wire": wire_checks(), "caveats": caveat_checks()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="Verify the saved exact record without rewriting it")
    args = parser.parse_args()
    result = run()
    if args.check:
        stored = json.loads(args.output.read_text(encoding="utf-8-sig"))
        current = json.loads(json.dumps(result, default=str))
        stored.pop("sympy_version", None)
        current.pop("sympy_version", None)
        if stored != current:
            raise AssertionError("Saved exact record/source identity differs from recomputation")
    else:
        args.output.write_text(json.dumps(result, default=str, indent=2) + "\n", encoding="utf-8")
    print("PASS exact port duality/lift identities and independent Bessel-series wire check")
