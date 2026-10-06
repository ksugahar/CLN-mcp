"""Exact synthesis of finite scalar rational RL terminal impedances.

Scope: simple negative real poles and low-frequency series-R/shunt-L Cauer
extraction at s0=0. Exact symbolic coefficients are required; Float inputs
are rejected (callers may explicitly rationalize an approximate model). Not shifted Type1/Type2 field-mode generation, nonlinear
synthesis, arbitrary positive-real RLC functions, or multiport conversion.
"""
import operator
import sympy as sp


def _exact(value):
    value=sp.sympify(value)
    if value.has(sp.Float):
        raise ValueError("Exact coefficients required; rationalize Float inputs explicitly")
    return value


def cauer_to_impedance(elements,s):
    if not elements:
        raise ValueError('Empty circuit')
    elements=[(kind,_exact(value)) for kind,value in elements]
    for index,(kind,value) in enumerate(elements):
        if kind != ('R' if index%2==0 else 'L'):
            raise ValueError('Expected alternating series R and shunt L')
        if value.is_nonnegative is not True or (kind=='L' and value.is_positive is not True):
            raise ValueError('Nonnegative R and positive L required')
    kind,value=elements[-1]
    result=value if kind=='R' else s*value
    for kind,value in reversed(elements[:-1]):
        result=sp.cancel(value+result if kind=='R' else (s*value*result)/(s*value+result))
    return sp.cancel(result)


def foster_to_cauer(z,s,max_elements=32):
    """Extract exact finite DC Cauer elements, including the last termination."""
    z=_exact(z)
    try:
        max_elements=operator.index(max_elements)
    except TypeError as exc:
        raise ValueError('Element budget must be a positive integer') from exc
    if max_elements<1:
        raise ValueError('Element budget must be a positive integer')
    current=sp.cancel(z);elements=[]
    def verified():
        if sp.cancel(cauer_to_impedance(elements,s)-z)!=0:
            raise ValueError('Extraction did not reconstruct the exact impedance')
        return elements
    while len(elements)<max_elements:
        r=sp.limit(current,s,0)
        if r.is_finite is not True or r.is_nonnegative is not True:
            raise ValueError('Not a regular scalar RL impedance at DC')
        elements.append(('R',r))
        remaining=sp.cancel(current-r)
        if remaining==0:return verified()
        if len(elements)==max_elements:break
        admittance=sp.cancel(1/remaining)
        inverse_l=sp.limit(s*admittance,s,0)
        if inverse_l.is_positive is not True:
            raise ValueError('No positive inductive extraction at DC')
        elements.append(('L',sp.cancel(1/inverse_l)))
        remaining_y=sp.cancel(admittance-inverse_l/s)
        if remaining_y==0:return verified()
        current=sp.cancel(1/remaining_y)
    raise ValueError('Extraction did not terminate within the element budget')


def cauer_to_foster(elements,s):
    """Partial fractions of the same terminal Z; return D, Ltail, and (p,r)."""
    z=cauer_to_impedance(elements,s)
    dc=sp.limit(z,s,0);tail=sp.limit(z/s,s,sp.oo)
    remainder=sp.cancel(z-dc-tail*s)
    if remainder==0:
        if dc.is_nonnegative is not True or tail.is_nonnegative is not True:
            raise ValueError('Negative static term')
        return dc,tail,[]
    numerator,denominator=sp.fraction(remainder)
    polynomial=sp.Poly(denominator,s)
    if sp.gcd(polynomial,polynomial.diff()).degree()!=0:
        raise ValueError('Expected simple negative real poles')
    try:
        roots=polynomial.real_roots(radicals=False)
        if len(roots)!=polynomial.degree():
            raise ValueError('Expected simple negative real poles')
        root_sum_check=True
    except (NotImplementedError, sp.polys.polyerrors.DomainError):
        # Preserve exact, sign-decidable parameterized examples.
        factored=sp.roots(denominator,s)
        if sum(factored.values())!=polynomial.degree():
            raise ValueError('Exact real-pole isolation unavailable')
        if any(multiplicity!=1 for multiplicity in factored.values()):
            raise ValueError('Expected simple negative real poles')
        roots=list(factored);root_sum_check=False
    derivative=sp.diff(denominator,s)
    branches=[]
    intervals=polynomial.intervals(eps=sp.Rational(1,10**8)) if (
        root_sum_check and (polynomial.domain.is_QQ or polynomial.domain.is_ZZ)) else None
    n_poly=sp.Poly(numerator,s); r_den=sp.Poly(s*derivative,s)
    for index,pole in enumerate(roots):
        if pole.is_negative is not True:
            raise ValueError('Expected simple negative real poles')
        r=numerator.subs(s,pole)/(pole*derivative.subs(s,pole))
        if intervals is not None:
            (lo,hi),_=intervals[index]
            # Certify the rational residue's sign on the isolated root interval.
            # Avoid expensive symbolic simplification of CRootOf expressions.
            for _ in range(100):
                if lo==hi or (n_poly.count_roots(lo,hi)==0 and r_den.count_roots(lo,hi)==0):
                    break
                lo,hi=polynomial.refine_root(lo,hi,eps=(hi-lo)/2)
            else:
                raise ValueError('Could not certify positive branch resistance')
            midpoint=(lo+hi)/2
            positive=(n_poly.eval(midpoint)/r_den.eval(midpoint)).is_positive
        else:
            r=sp.simplify(r);positive=r.is_positive
        if positive is not True:
            raise ValueError('Expected positive RL branch resistance')
        branches.append((-pole,r))
    if dc.is_nonnegative is not True or tail.is_nonnegative is not True:
        raise ValueError('Negative static term')
    if root_sum_check:
        # Exact rational RootSum identity avoids expanding sums of unrelated
        # CRootOf objects into huge radical expressions.
        t=sp.Dummy('pole')
        reconstructed=dc+tail*s+sp.RootSum(polynomial,
            sp.Lambda(t,numerator.subs(s,t)*s/(t*derivative.subs(s,t)*(s-t))))
    else:
        reconstructed=dc+tail*s+sum(r*s/(s+p) for p,r in branches)
    if sp.cancel(reconstructed-z)!=0:
        raise ValueError('Partial fractions did not reconstruct the same Z')
    return dc,tail,sorted(branches,key=lambda item:sp.default_sort_key(item[0]))



if __name__=='__main__':
    s=sp.Symbol('s')
    for z in [1+4*s/(s+2)+9*s/(s+3), 1+4*s/(s+2)+9*s/(s+3)+s/5]:
        elements=foster_to_cauer(z,s)
        assert sp.cancel(cauer_to_impedance(elements,s)-z)==0
        dc,tail,branches=cauer_to_foster(elements,s)
        back=dc+tail*s+sum(r*s/(s+p) for p,r in branches)
        assert sp.cancel(back-z)==0
        print('PASS exact round trip:',elements,branches)
