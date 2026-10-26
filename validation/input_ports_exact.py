"""Check Mathematica symbolic records and an independent rational Bessel oracle."""
import argparse
from pathlib import Path
from fractions import Fraction
from reference_records import historical,symbolic_record,wire_record
from reference_oracles import bessel_series,type2_series

def run():
 record=symbolic_record();wire_record();legacy=historical('input_ports_exact.json')
 if record['port_elements']!=[x[1] for x in legacy['port']['matched_index_shift']['B_elements']]:raise AssertionError('Mathematica/legacy port elements differ')
 # Defining Bessel coefficients are independent of field recursion and CAS.
 if type2_series([1,3,5,7,9],[Fraction(1,8),Fraction(1,16),Fraction(1,24),Fraction(1,32)])!=bessel_series():raise AssertionError('exact Bessel contact differs')
 return legacy
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path);parser.add_argument('--check',action='store_true');args=parser.parse_args()
 if args.output and args.output.resolve()!=(Path(__file__).resolve().parents[1]/'docs/data/input_ports_exact.json').resolve():raise ValueError('historical record is read-only; use symbolic_cln.wls for new output')
 run();print('PASS Mathematica port/caveat record and independent exact Bessel-series wire check')
