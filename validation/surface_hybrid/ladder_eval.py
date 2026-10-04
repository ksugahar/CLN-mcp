"""Input impedance of the shifted Type2 / Type1 ladders (q = s - s0), for either terminating element.

Type2  el = [R0, L1, R2, L3, ...]:   Z = R0 + ((q L1) || (R2 + ((q L3) || ...)))
Type1  el = [L-1, R0, L1, R2, ...]:  Z/s = L-1 || (R0/q + (L1 || (R2/q + ...)))

A ladder may end on either element: a final shunt q L (Type2) is the branch itself, a final series
R/q (Type1) is the branch itself.  Type1 is evaluated as the admittance g = s/Z, so the removable
divisions by q vanish and s = s0 is regular (R0/q opens and disconnects the rest: Z -> s0 L-1).
"""


def z_type2(el, s0, s):
    q = s - s0
    n = len(el) - 1
    z = q*el[n] if n % 2 else el[n]
    for i in range(n - 1, -1, -1):
        z = el[i] + z if i % 2 == 0 else (q*el[i])*z/(q*el[i] + z)
    return z


def z_type1(el, s0, s):
    q = s - s0
    n = len(el) - 1
    g = q/el[n] if n % 2 else 1/el[n]          # g = admittance of Z/s seen from the tail
    for i in range(n - 1, -1, -1):
        g = 1/el[i] + g if i % 2 == 0 else q*g/(el[i]*g + q)
    return s/g
