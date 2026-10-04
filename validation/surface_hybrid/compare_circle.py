"""Band and DC controls for positive four-pole Foster and shifted CLN on a circle.

The two bands and two static constraints are varied independently. Positive
Foster coefficients imply stable passive RL realizations. Fitted pole locations
are not the original diffusion eigenvalues. This is a circuit fit, not merely
selection of four physical eigenfields. Reference is the Bessel analytic solution.
"""
import argparse,json,math
from pathlib import Path
import numpy as np
import mpmath as mp
from scipy.special import jn_zeros
from positive_foster import foster_fit,response
from ladder_eval import z_type1,z_type2

mp.mp.dps=60
MU=4e-7*math.pi; SIGMA=1e6; A=.01; TAU=MU*SIGMA*A*A; SCALE=2*math.pi/SIGMA

def exact(f):
    u=2j*math.pi*f*TAU; x=mp.sqrt(mp.mpc(u))
    # Z/SCALE has unit static slope; avoids subtracting nearly equal Bessel values.
    return complex(2*u*mp.besseli(1,x)/(x*mp.besseli(0,x)))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--surface-results',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    surf=json.loads(Path(args.surface_results).read_text(encoding='utf-8'))
    freq=np.geomspace(.1,1e6,401); u=2j*np.pi*TAU*freq
    reference=np.array([exact(f) for f in freq])
    delta=np.sqrt(2/abs(u)); transition=(delta>=.3)&(delta<=3); high=freq>=1e3
    runs={}
    p=jn_zeros(0,4)**2
    seed0={'poles':p.tolist(),'residues':[4.]*4,'L_tail':1-float(np.sum(4/p)),'seconds':0.}
    for dc,band,label in [(True,[.1,3e4],'DC'),(True,[1e3,1e6],'high+DC'),
                           (False,[.1,3e4],'low-free-DC'),(False,[1e3,1e6],'band')]:
        seed=seed0 if dc else runs['Foster fitted '+('DC' if band[0]<1 else 'high+DC')]
        tr=np.geomspace(*band,90); utr=2j*np.pi*TAU*tr
        model,info=foster_fit(utr,np.array([exact(f) for f in tr]),4,p,dc,seed)
        z=model(u); error=abs(z/reference-1)
        info.update(relative_error=error.tolist(),transition_max=float(error[transition].max()),
                    high_band_max=float(error[high].max()),at_1MHz=float(error[-1]),rank=4,train_band_hz=band)
        runs['Foster fitted '+label]=info
    for typ,fun in [('Type1',z_type1),('Type2',z_type2)]:
        z=np.array([fun(surf['exact_elements'][typ],surf['s0'],2j*np.pi*f)/SCALE for f in freq])
        err=abs(z/reference-1)
        runs['analytic CLN '+typ+' 9 elements']={'relative_error':err.tolist(),'transition_max':float(err[transition].max()),
            'high_band_max':float(err[high].max()),'at_1MHz':float(err[-1]),'elements':surf['exact_elements'][typ]}
    # Foster's exact spectral expansion, checked with a positive low-frequency tail.
    p128=jn_zeros(0,128)**2; tail=1-float(np.sum(4/p128))
    assert tail>0
    z128=response(p128,4*np.ones(128),u)+tail*u
    check=float(np.max(abs(z128[transition]/reference[transition]-1)))
    assert check<1e-6
    for key,info in runs.items():
        print(key, 'transition',info['transition_max'],'high',info['high_band_max'],flush=True)
    d={'scope':'Linear 2D transverse circle; analytic oracle; 2 bands x 2 DC controls',
       'tau':TAU,'Z_scale':SCALE,'frequency_hz':freq.tolist(),'transition_delta_over_a':[.3,3],
       'runs':runs,'Foster_spectral_identity_check_transition':check,
       'caveat':'Fitted Foster poles and residues are optimized; CLN uses the fixed real expansion point. Representation alone has no accuracy ranking.'}
    Path(args.output).write_text(json.dumps(d,indent=2),encoding='utf-8')
    print('PASS: independent holdout grid and Foster/Bessel identity check')

if __name__=='__main__': main()
