from pathlib import Path
import sys,numpy as np,sympy as sp
sys.path.insert(0,str(Path(__file__).resolve().parent))
from foster_cauer_exact import cauer_to_impedance,cauer_to_foster,foster_to_cauer
s=sp.Symbol('s');assert cauer_to_foster([('R',1),('L',2)],s)==(1,2,[]);z=1+4*s/(s+2)+9*s/(s+3)
for value in [1+4.0*s/(s+2)+9*s/(s+3),1+0.1*s/(s+0.3)+0.7*s/(s+1.9)]:
 try:foster_to_cauer(value,s)
 except ValueError as e:assert 'Exact coefficients' in str(e)
 else:raise AssertionError('Float accepted')
 assert sp.cancel(cauer_to_impedance(foster_to_cauer(sp.nsimplify(value,rational=True),s),s)-sp.nsimplify(value,rational=True))==0
assert len(foster_to_cauer(z,s,5))==5
for budget in [0,4,4.5]:
 try:foster_to_cauer(z,s,budget)
 except ValueError:pass
 else:raise AssertionError('Bad budget accepted')
assert cauer_to_impedance([('R',1),('L',1),('R',0)],s)==1
for values in [[1]*7,[1,2,3,1,2,3,1,1,5]]:
 elements=[('R' if j%2==0 else 'L',v) for j,v in enumerate(values)]
 dc,tail,branches=cauer_to_foster(elements,s)
 assert len(branches)==len(values)//2
 for omega in [.001,.3,1,10,1000]:
  q=sp.I*sp.Rational(str(omega))
  expected=complex(cauer_to_impedance(elements,s).subs(s,q).evalf(30))
  actual=complex(dc)+complex(tail)*complex(q)+sum(complex(r.evalf(30))*complex(q)/(complex(q)+complex(p.evalf(30))) for p,r in branches)
  assert abs(actual-expected)/max(abs(expected),1e-20)<1e-12
 print('PASS exact pole isolation and response:',len(values),'elements')
print('PASS Float rejection/rationalization, short termination and exact odd budgets')
