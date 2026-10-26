"""Mathematica exact synthesis plus independent numerical circuit evaluation."""
from fractions import Fraction
import numpy as np
from reference_records import symbolic_record
from surface_hybrid.ladder_eval import z_type2

def main():
 data=symbolic_record()
 for model in data['conversion_examples']:
  elements=[float(Fraction(x[1])) for x in model['cauer']]
  for omega in [.001,.3,1,10,1000]:
   s=1j*omega
   expected=complex(float(Fraction(model['dc'])))+s*float(Fraction(model['tail']))+sum(float(Fraction(r))*s/(s+float(Fraction(p))) for p,r in model['branches'])
   actual=z_type2(elements,0,s)
   assert abs(actual-expected)/max(abs(expected),1e-20)<1e-12
 for model in data['foster_numeric_models']:
  elements=[float(Fraction(x)) for x in model['cauer_values']]
  assert len(model['branches'])==len(elements)//2
  for omega in [.001,.3,1,10,1000]:
   s=1j*omega;expected=z_type2(elements,0,s)
   actual=float(model['dc'])+s*float(model['tail'])+sum(float(r)*s/(s+float(p)) for p,r in model['branches'])
   assert abs(actual-expected)/max(abs(expected),1e-20)<1e-12
 print('PASS Mathematica exact synthesis and independent complex circuit responses (1e-12)')
if __name__=='__main__':main()
