"""Exact synthesis of finite scalar rational RL terminal impedances.

Scope: simple negative real poles and low-frequency series-R/shunt-L Cauer
extraction at s0=0. Not shifted Type1/Type2 field-mode generation, nonlinear
synthesis, arbitrary positive-real RLC functions, or multiport conversion.
"""
import sympy as sp


def cauer_to_impedance(elements,s):
    if not elements:
        raise ValueError('Empty circuit')
    elements=[(kind,sp.sympify(value)) for kind,value in elements]
    for index,(kind,value) in enumerate(elements):
        if kind != ('R' if index%2==0 else 'L'):
            raise ValueError('Expected alternating series R and shunt L')
        if value.is_nonnegative is not True or (kind=='L' and value.is_positive is not True):
            raise ValueError('Nonnegative R and positive L required')
    kind,value=elements[-1]
    result=value if kind=='R' else s*value
    for kind,value in reversed(elements[:-1]):
        result=sp.cancel(value+result if kind=='R' else 1/(1/(s*value)+1/result))
    return sp.cancel(result)


def foster_to_cauer(z,s,max_elements=32):
    """Extract exact finite DC Cauer elements, including the last termination."""
    current=sp.cancel(z);elements=[]
    for _ in range(max_elements//2):
        r=sp.limit(current,s,0)
        if r.is_finite is not True or r.is_nonnegative is not True:
            raise ValueError('Not a regular scalar RL impedance at DC')
        elements.append(('R',r))
        remaining=sp.cancel(current-r)
        if remaining==0:return elements
        admittance=sp.cancel(1/remaining)
        inverse_l=sp.limit(s*admittance,s,0)
        if inverse_l.is_positive is not True:
            raise ValueError('No positive inductive extraction at DC')
        elements.append(('L',sp.cancel(1/inverse_l)))
        remaining_y=sp.cancel(admittance-inverse_l/s)
        if remaining_y==0:return elements
        current=sp.cancel(1/remaining_y)
    raise ValueError('Extraction did not terminate within the element budget')


def cauer_to_foster(elements,s):
    """Partial fractions of the same terminal Z; return D, Ltail, and (p,r)."""
    z=cauer_to_impedance(elements,s)
    dc=sp.limit(z,s,0);tail=sp.limit(z/s,s,sp.oo)
    remainder=sp.cancel(z-dc-tail*s)
    _,denominator=sp.fraction(remainder)
    roots=sp.roots(denominator,s)
    if sum(roots.values())!=sp.degree(denominator,s):
        raise ValueError('Exact pole factorization unavailable')
    branches=[]
    for pole,multiplicity in roots.items():
        if multiplicity!=1 or pole.is_negative is not True:
            raise ValueError('Expected simple negative real poles')
        p=-pole;ordinary_residue=sp.limit((s-pole)*remainder,s,pole)
        r=sp.cancel(-ordinary_residue/p)
        if r.is_positive is not True:
            raise ValueError('Expected positive RL branch resistance')
        branches.append((p,r))
    if dc.is_nonnegative is not True or tail.is_nonnegative is not True:
        raise ValueError('Negative static term')
    reconstructed=dc+tail*s+sum(r*s/(s+p) for p,r in branches)
    if sp.cancel(reconstructed-z)!=0:
        raise ValueError('Partial fractions did not reconstruct the same Z')
    return dc,tail,sorted(branches,key=lambda item:float(item[0]))


if __name__=='__main__':
    s=sp.Symbol('s')
    for z in [1+4*s/(s+2)+9*s/(s+3), 1+4*s/(s+2)+9*s/(s+3)+s/5]:
        elements=foster_to_cauer(z,s)
        assert sp.cancel(cauer_to_impedance(elements,s)-z)==0
        dc,tail,branches=cauer_to_foster(elements,s)
        back=dc+tail*s+sum(r*s/(s+p) for p,r in branches)
        assert sp.cancel(back-z)==0
        print('PASS exact round trip:',elements,branches)
