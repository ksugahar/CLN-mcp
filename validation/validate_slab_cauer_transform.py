"""High-precision mpmath check of the Mathematica slab record.

The matrices come from Legendre orthogonality, independently of the
Mathematica integrals. The transformed circuit uses the same L-orthogonal
Krylov method in a separate implementation, so this is a cross-implementation
check of one method, not an independent second method.
"""
import argparse
import json
from pathlib import Path
import mpmath as mp

def verify(record):
    mp.mp.dps = 70
    n = 8
    r = mp.diag([mp.mpf(1)/(4*k+1) for k in range(n)])
    h = [{1: mp.mpf(1)}] + [
        {2*k+1: mp.mpf(1)/(4*k+1), 2*k-1: -mp.mpf(1)/(4*k+1)}
        for k in range(1, n)]
    l = mp.matrix(n)
    for i in range(n):
        for j in range(n):
            l[i,j] = sum(a*h[j].get(degree, 0)/(2*degree+1) for degree,a in h[i].items())
    b = mp.eye(n)[:,0]
    vectors = []
    q = mp.lu_solve(l,b)
    for _ in range(n):
        q /= mp.sqrt((q.T*l*q)[0])
        vectors.append(q.copy())
        q = mp.lu_solve(l,r*q)
        for v in vectors:
            q -= v * (v.T*l*q)[0]
    transform = mp.matrix(n)
    for j,v in enumerate(vectors):
        transform[:,j] = v

    def coefficients(w, m, high):
        v = transform[:,:m] if high else mp.eye(n)[:,:m]
        g = mp.lu_solve(v.T*(r+mp.j*w*l)*v,v.T*b)
        current = (b.T*v*g)[0]
        return v*g/current, 1/current

    maximum = mp.mpf(0)
    for row in record["impedance_samples"]:
        w = mp.mpf(str(row["omega_tau"]))
        k = mp.sqrt(mp.j*w)
        exact = k/mp.tanh(k)
        for high, key in [(False, "I_errors"),(True, "II_errors")]:
            for m in range(1,n+1):
                _,z = coefficients(w,m,high)
                error = abs(z/exact-1)
                maximum = max(maximum, abs(error-row[key][m-1]))
    if maximum > mp.mpf("2e-10"):
        raise AssertionError(f"impedance disagreement {maximum}")

    if record["band_threshold"] != 0.01:
        raise AssertionError("AC band threshold must be 1 percent")
    threshold = mp.mpf(".01")
    scan_low, scan_high, per_decade = -4, 5, 200
    if record["scan_points"] != (scan_high-scan_low)*per_decade+1:
        raise AssertionError(f"scan grid changed: {record['scan_points']}")

    def band_error(w, stages, high):
        _, z = coefficients(w, stages, high)
        k = mp.sqrt(mp.j*w)
        return abs(z/(k/mp.tanh(k))-1)

    def scan_point(index):
        return mp.mpf(10)**(scan_low+mp.mpf(index)/per_decade)

    def scan_index(endpoint):
        return (mp.log10(endpoint)-scan_low)*per_decade

    band_checks = 0
    for row in record["bands"]:
        stages = row["stages"]
        for high, key in [(False, "I"), (True, "II")]:
            for left, right in row[key]:
                lo, hi = mp.mpf(str(left)), mp.mpf(str(right))
                if not lo < hi:
                    raise AssertionError(f"empty band {stages} {key} {left} {right}")
                for endpoint in (left, right):
                    if endpoint in (10.0**scan_low, 10.0**scan_high):
                        continue
                    error = band_error(mp.mpf(str(endpoint)), stages, high)
                    if abs(error-threshold) > mp.mpf("2e-10"):
                        raise AssertionError(
                            f"band endpoint {stages} {key} {endpoint}: error {error}, expected 0.01")
                # Interior: geometric midpoint and the scan points next to each edge.
                inside = [mp.sqrt(lo*hi),
                          scan_point(mp.ceil(scan_index(lo)-mp.mpf("1e-9"))),
                          scan_point(mp.floor(scan_index(hi)+mp.mpf("1e-9")))]
                for w in inside:
                    if not lo <= w <= hi:
                        raise AssertionError(f"band {stages} {key} [{left}, {right}] holds no scan point")
                    error = band_error(w, stages, high)
                    if error > threshold:
                        raise AssertionError(
                            f"inside band {stages} {key} w={w}: error {error} > 0.01")
                    band_checks += 1
                # Exterior: the adjacent scan point beyond each refined edge.
                outside = []
                if left != 10.0**scan_low:
                    outside.append(scan_point(mp.floor(scan_index(lo)-mp.mpf("1e-9"))))
                if right != 10.0**scan_high:
                    outside.append(scan_point(mp.ceil(scan_index(hi)+mp.mpf("1e-9"))))
                for w in outside:
                    error = band_error(w, stages, high)
                    if error <= threshold:
                        raise AssertionError(
                            f"outside band {stages} {key} w={w}: error {error} <= 0.01")
                    band_checks += 1
    # Every recorded sample must lie inside a band exactly when its error is at most 1 percent.
    bands = {(row["stages"], key): row[key] for row in record["bands"] for key in ("I", "II")}
    for row in record["impedance_samples"]:
        w = row["omega_tau"]
        for key in ("I", "II"):
            for m in range(1, n+1):
                inside = any(left <= w <= right for left, right in bands[(m, key)])
                if (row[f"{key}_errors"][m-1] <= 0.01) != inside:
                    raise AssertionError(
                        f"sample w={w} stages={m} {key}: error {row[f'{key}_errors'][m-1]}, inside={inside}")
                band_checks += 1
    field_delta = mp.mpf(0)
    for row in record["field_errors"]:
        w = mp.mpf(str(row["omega_tau"]))
        k = mp.sqrt(mp.j*w)
        reference = [[],[],[]]
        for x in [mp.mpf(i)/100 for i in range(-100,101)]:
            reference[0].append(k*mp.cosh(k*x)/mp.sinh(k))
            reference[1].append((mp.cosh(k)-mp.cosh(k*x))/(k*mp.sinh(k)))
            reference[2].append(mp.sinh(k*x)/mp.sinh(k))
        for high, m, key in [(False,8,"full_I"),(True,8,"full_II"),(False,3,"three_I"),(True,3,"three_II")]:
            co,_ = coefficients(w,m,high)
            approximate = [[],[],[]]
            for x in [mp.mpf(i)/100 for i in range(-100,101)]:
                jval = sum(co[t]*mp.legendre(2*t,x) for t in range(n))
                hval = sum(co[t]*sum(a*mp.legendre(degree,x) for degree,a in h[t].items()) for t in range(n))
                # Integrate H via the Legendre antiderivative, enforcing A(1)=0.
                aval = -sum(co[t]*sum(a*(mp.legendre(degree+1,x)-mp.legendre(degree-1,x))/(2*degree+1) for degree,a in h[t].items()) for t in range(n))
                for out,value in zip(approximate,(jval,aval,hval)):
                    out.append(value)
            errs = [mp.sqrt(sum(abs(a-e)**2 for a,e in zip(out,ref))/sum(abs(e)**2 for e in ref)) for out,ref in zip(approximate,reference)]
            field_delta = max(field_delta,*(abs(a-b) for a,b in zip(errs,row[key])))
    if field_delta > mp.mpf("2e-10"):
        raise AssertionError(f"field disagreement {field_delta}")
    return {"ok": True, "independent_precision_digits":mp.mp.dps,
        "max_impedance_error_difference":float(maximum),
        "max_field_error_difference":float(field_delta),
        "band_membership_checks":band_checks}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path, nargs="?",
        default=Path(__file__).resolve().parents[1]/"docs"/"data"/"slab_cauer_transform.json")
    args = parser.parse_args()
    print(json.dumps(verify(json.loads(args.record.read_text(encoding="utf-8"))),indent=2))
