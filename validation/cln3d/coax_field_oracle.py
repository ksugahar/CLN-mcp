"""Independent unit-current coaxial field integrals from Bessel solutions."""
import argparse,hashlib,json
from pathlib import Path
import mpmath as mp

def evaluate(hz):
    with mp.workdps(50):
        mu=4*mp.pi*mp.mpf('1e-7');sigma=mp.mpf('1e6');a=mp.mpf('.005');b=mp.mpf('.015');h=mp.mpf('.015')
        omega=2*mp.pi*hz;k=mp.sqrt(1j*omega*mu*sigma);den=2*mp.pi*a*mp.besseli(1,k*a)
        loss=2*mp.pi*h/sigma*mp.quad(lambda r:abs(k*mp.besseli(0,k*r)/den)**2*r,[0,a])
        inside=2*mp.pi*h*mu*mp.quad(lambda r:abs(mp.besseli(1,k*r)/den)**2*r,[0,a])
        air=mu*h*mp.log(b/a)/(2*mp.pi)
        z=h*k*mp.besseli(0,k*a)/(sigma*den)+1j*omega*air
        defect=abs((loss+1j*omega*(inside+air))/z-1)
        assert defect<mp.mpf('1e-35')
        return dict(hz=hz,unit_current_joule=float(loss),unit_current_magnetic_conductor=float(inside),unit_current_magnetic_air=float(air),power_identity_defect=float(defect))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);args=p.parse_args()
    ident=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out=dict(complete=True,source_sha256={'validation/cln3d/coax_field_oracle.py':ident},scope='Exact axially invariant MQS coax; unit current, no half factor; radial Bessel field integrals, not FE data',cases=[evaluate(hz) for hz in [1000,10000,100000,1000000]])
    Path(args.output).write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('PASS Bessel field-integral power identity at all four frequencies')

if __name__=='__main__':main()
