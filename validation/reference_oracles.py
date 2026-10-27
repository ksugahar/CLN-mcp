"""Independent fixed-case rational numerical oracles (stdlib Fraction, no CAS)."""
from fractions import Fraction as F
from math import factorial

def series_constant(value,n):return [F(value)]+[F(0)]*(n-1)
def add(a,b):return [x+y for x,y in zip(a,b)]
def mul(a,b):return [sum((a[j]*b[i-j] for j in range(i+1)),F(0)) for i in range(len(a))]
def inverse(a):
 if not a[0]:raise ValueError('regular series constant required')
 b=[1/a[0]]
 for i in range(1,len(a)):b.append(-sum((a[j]*b[i-j] for j in range(1,i+1)),F(0))/a[0])
 return b
def shift(a):return [F(0)]+a[:-1]
def scale(a,v):return [F(v)*x for x in a]
def type1_series(elements,n):
 g=shift(series_constant(1/elements[-1],n)) if len(elements)%2==0 else series_constant(1/elements[-1],n)
 for i in range(len(elements)-2,-1,-1):
  g=add(series_constant(1/elements[i],n),g) if i%2==0 else mul(shift(g),inverse(add(scale(g,elements[i]),shift(series_constant(1,n)))))
 return shift(inverse(g))
def bessel_series(n=9):
 numerator=[F(1,4**k*factorial(k)**2) for k in range(n)]
 denominator=[F(1,4**k*factorial(k)*factorial(k+1)) for k in range(n)]
 return mul(numerator,inverse(denominator))
def type2_series(rs,ls,n=9):
 z=series_constant(rs[-1],n)
 for r,l in reversed(list(zip(rs[:-1],ls))):
  sl=shift(series_constant(l,n));z=add(series_constant(r,n),mul(mul(sl,z),inverse(add(sl,z))))
 return z
def solve(matrix,rhs):
 n=len(rhs);a=[[F(x) for x in row]+[F(rhs[i])] for i,row in enumerate(matrix)]
 for i in range(n):
  pivot=next(j for j in range(i,n) if a[j][i]);a[i],a[pivot]=a[pivot],a[i]
  divisor=a[i][i];a[i]=[x/divisor for x in a[i]]
  for j in range(n):
   if j!=i:
    factor=a[j][i];a[j]=[x-factor*y for x,y in zip(a[j],a[i])]
 return [row[-1] for row in a]
def dot(a,b):return sum((x*y for x,y in zip(a,b)),F(0))
def truncated_energy(case):
 k=list(map(F,case['K_diagonal']));m=list(map(F,case['M_diagonal']));f=list(map(F,case['source']));count=case['states']
 a=[fi/ki for fi,ki in zip(f,k)];e=[F(0)]*len(f);modes=[];elements=[]
 for _ in range(count):
  l=dot(a,[ki*x for ki,x in zip(k,a)]);e=[x+y/l for x,y in zip(e,a)];res=1/dot(e,[mi*x for mi,x in zip(m,e)])
  modes.append(a);elements.extend([l,res]);a=[x-res*mi*y/ki for x,mi,y,ki in zip(a,m,e,k)]
 if elements!=list(map(F,case['elements_LR'])):raise AssertionError('independent rational energy elements differ')
 if not all(x>0 for x in elements):raise AssertionError('nonpositive energy element')
 n=2*count+2;kg=[[dot(q,[ki*x for ki,x in zip(k,p)]) for p in modes] for q in modes];mg=[[dot(q,[mi*x for mi,x in zip(m,p)]) for p in modes] for q in modes];fr=[dot(q,f) for q in modes]
 vec=solve(kg,fr);gal=[F(0)]
 for _ in range(n-1):
  gal.append(dot(fr,vec));vec=solve(kg,[-dot(row,vec) for row in mg])
 ladder=type1_series(elements,n)
 if ladder!=gal:raise AssertionError('rational ladder/Galerkin moments differ')
 full=[F(0)]+[sum((fi*fi/ki*(-mi/ki)**j for fi,ki,mi in zip(f,k,m)),F(0)) for j in range(n-1)]
 if any(full[j]!=gal[j] for j in range(2*count+1)) or full[2*count+1]==gal[2*count+1]:raise AssertionError('exact moment contact changed')
 if case['first_error_power']!=2*count+1:raise AssertionError('first error power differs')
 return elements
