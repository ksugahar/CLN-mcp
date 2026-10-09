"""Analytic AC-axis comparison; no FEM or digitized publication data."""
import hashlib,json,math
from pathlib import Path
import mpmath as mp
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def impedance(z):
    if z==0:return mp.mpf(1)
    k=mp.sqrt(z)
    return k*mp.besseli(0,k)/(2*mp.besseli(1,k))

def reciprocal(c):
    out=[1/c[0]]
    for n in range(1,len(c)):
        out.append(-mp.fsum(c[i]*out[n-i] for i in range(1,n+1))/c[0])
    return out

def extract(z0,count):
    seq=mp.taylor(impedance,z0,count+2);out=[]
    for n in range(count):
        out.append(seq[0] if n%2==0 else 1/seq[0])
        seq=seq[1:]
        if n<count-1:seq=reciprocal(seq)
    assert all(v>0 for v in out)
    return out

def ladder(c,q):
    if q==0:return c[0]
    out=c[-1] if len(c)%2 else q*c[-1]
    for n in range(len(c)-2,-1,-1):
        out=c[n]+out if n%2==0 else q*c[n]*out/(q*c[n]+out)
    return out

def type1_extract(z0,count=6):
    seq=mp.taylor(lambda v:impedance(v)/v,z0,count+2);out=[]
    for i in range(count):
        seq=reciprocal(seq)
        out.append(1/seq[0] if i%2==0 else seq[0])
        seq=seq[1:]
    assert all(v>0 for v in out)
    return out

def type1_h(c,q):
    value=1/c[-1] if len(c)%2 else c[-1]
    for i in range(len(c)-2,-1,-1):
        value=(1/c[i] if i%2==0 else c[i])+q/value
    return 1/value

def pade(function,z0,m,n):
    p,q=mp.pade(mp.taylor(function,z0,m+n),m,n)
    return lambda z:mp.polyval(p[::-1],z-z0)/mp.polyval(q[::-1],z-z0)

def components(a):return {'real':a.real.tolist(),'imag':a.imag.tolist()}

def intervals(f,mask):
    spans=[];start=None
    for i,good in enumerate(mask):
        if good and start is None:start=i
        if start is not None and (not good or i==len(mask)-1):
            end=i if good else i-1;spans.append([float(f[start]),float(f[end])]);start=None
    return spans

def study():
    mp.mp.dps=80
    mu=4*mp.pi*mp.mpf('1e-7');sigma=mp.mpf('5.8e7');radius=mp.mpf('.005')
    tau=mu*sigma*radius**2;rdc=1/(mp.pi*sigma*radius**2)
    freq=np.logspace(1,7,601);z=[mp.mpc(0,2*mp.pi*float(f)*tau) for f in freq]
    exact=np.array([complex(1/(rdc*impedance(v))) for v in z]);rows=[]
    for count in (6,7):
        for f0 in (1000,10000,100000):
            z0=2*mp.pi*f0*tau;c=extract(z0,count)
            # Independent Taylor Pade, rather than the continued-fraction code.
            independent=pade(impedance,z0,3,2 if count==6 else 3)
            inverse_check=pade(lambda v:1/impedance(v),z0,2 if count==6 else 3,3)
            assert max(abs(inverse_check(v)*ladder(c,v-z0)-1) for v in z[::20])<mp.mpf('1e-50')
            defect=max(float(abs(ladder(c,v-z0)/independent(v)-1)) for v in z[::20])
            assert defect<1e-50,defect
            approx=np.array([complex(1/(rdc*ladder(c,v-z0))) for v in z])
            e_y=abs(approx/exact-1);e_z=abs(exact/approx-1)
            assert np.max(abs(e_y-e_z/abs(exact/approx)))<1e-12
            rows.append({'family':'Type2','elements':count,'pairs':3,'f0_hz':f0,'z0':float(z0),
                         'coefficients':[float(v) for v in c],'admittance':components(approx),
                         'relative_y_error':e_y.tolist(),'relative_z_error':e_z.tolist(),
                         'relative_real_y_error':(abs(approx.real/exact.real-1)).tolist(),
                         'relative_imag_y_error':(abs(approx.imag/exact.imag-1)).tolist(),
                         'one_percent_intervals_hz':intervals(freq,e_y<=.01),
                         'pade_relative_defect':defect})
    for f0 in (1000,10000,100000):
        z0=2*mp.pi*f0*tau;c=type1_extract(z0)
        independent=pade(lambda v:impedance(v)/v,z0,2,3)
        defect=max(float(abs(type1_h(c,v-z0)/independent(v)-1)) for v in z[::20])
        assert defect<1e-50
        approx=np.array([complex(1/(rdc*v*type1_h(c,v-z0))) for v in z]);err=abs(approx/exact-1)
        rows.append({'family':'Type1 analytic H continued fraction','elements':6,'pairs':3,
                     'f0_hz':f0,'z0':float(z0),'coefficients':[float(v) for v in c],
                     'admittance':components(approx),'relative_y_error':err.tolist(),
                     'one_percent_intervals_hz':intervals(freq,err<=.01),
                     'pade_relative_defect':defect})
    # A DC-preserving rational control for the Type1 diffusion transfer.
    # This is not a field-energy-derived Type1 circuit or a stage-count equivalence proof.
    h=lambda v:(impedance(v)-1)/v if v!=0 else mp.mpf(1)/8
    for f0 in (1000,10000,100000):
        z0=2*mp.pi*f0*tau;ph=pade(h,z0,2,3)
        approx=np.array([complex(1/(rdc*(1+v*ph(v)))) for v in z]);err=abs(approx/exact-1)
        rows.append({'family':'DC-preserving H Pade control','f0_hz':f0,'z0':float(z0),
                     'admittance':components(approx),'relative_y_error':err.tolist(),
                     'one_percent_intervals_hz':intervals(freq,err<=.01)})
    original=[]
    for z0 in (1,10,80):
        c=extract(mp.mpf(z0),7)
        original.append({'z0':z0,'errors_at_wtau_0_1_100':[float(abs(ladder(c,mp.mpc(0,w)-z0)/impedance(mp.mpc(0,w))-1)) for w in (.1,100)]})
    return {'scope':'analytic round wire, unit length, internal impedance only; not a digitization of Kuriyama Fig.8',
            'mu_h_per_m':float(mu),'sigma_s_per_m':float(sigma),'radius_m':float(radius),
            'length_m':1,'tau_s':float(tau),'rdc_ohm':float(rdc),'frequency_hz':freq.tolist(),
            'exact_admittance':components(exact),'models':rows,'original_notebook_checks':original,
            'source_sha256':{'validation/ac_axis_study.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}

def plot(data):
    import matplotlib.pyplot as plt
    freq=np.array(data['frequency_hz']);ex=data['exact_admittance'];fig,ax=plt.subplots(2,3,figsize=(13,7))
    for row,count in enumerate((6,7)):
        ax[row,0].loglog(freq,ex['real'],color='black',ls=':',label='Bessel exact')
        ax[row,1].loglog(freq,-np.array(ex['imag']),color='black',ls=':',label='Bessel exact')
        for item in data['models']:
            if item['family']!='Type2' or item['elements']!=count:continue
            label=f"f0={item['f0_hz']/1000:g} kHz"
            ax[row,0].loglog(freq,item['admittance']['real'],label=label)
            im=-np.array(item['admittance']['imag']);assert (im>0).all()
            ax[row,1].loglog(freq,im,label=label)
            ax[row,2].loglog(freq,np.maximum(item['relative_y_error'],1e-16),label=label)
        ax[row,0].set_ylabel(f'{count} elements\nRe Y [S]')
        ax[row,1].set_ylabel('-Im Y [S]')
        ax[row,2].set_ylabel('|Yred/Yexact - 1|');ax[row,2].axhline(.01,color='k',ls=':',label='1%')
        for a in ax[row]:a.set_xlabel('Frequency [Hz]');a.grid(alpha=.25);a.legend(fontsize=8)
    fig.suptitle('Analytic 1 m round wire: real s0=2pi f0, three RL pairs\n6 elements vs 7 elements (additional terminal resistor)')
    fig.tight_layout();return fig

if __name__=='__main__':
    data=study();dest=ROOT/'docs/data/ac_axis_study.json'
    dest.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8',newline='\n')
    for row in data['models']:print(row['family'],row.get('elements'),row['f0_hz'],row['one_percent_intervals_hz'])