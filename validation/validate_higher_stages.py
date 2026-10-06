"""Sweep FE order, mesh and quadrature against an independent Bessel-series oracle.

Failures of the strict production load gate are retained as failures. Boundary
conditions, ICCG shift and load tolerance are unchanged. Verification covers
three stages of a linear cylindrical conductor at zero expansion point only.
"""
import argparse, contextlib, hashlib, io, json, time
from pathlib import Path
import numpy as np
import sympy as sp
import ngsolve
from cln3d import CLN_APhi,CLN_T_Omega,CLN_AT
ROOT=Path(__file__).resolve().parents[1]


def oracle():
    u=sp.Symbol('u')
    i0=sum(u**k/(4**k*sp.factorial(k)**2) for k in range(9))
    i1_over_root=sum(u**k/(2*4**k*sp.factorial(k)*sp.factorial(k+1)) for k in range(9))
    z=sp.cancel(i0/(2*i1_over_root));rs=[];ls=[]
    for j in range(4):
        r=sp.limit(z,u,0);rs.append(r)
        if j==3:break
        remainder=sp.cancel(z-r);l=sp.limit(remainder/u,u,0);ls.append(l)
        z=sp.cancel(1/(1/remainder-1/(u*l)))
    assert rs==[1,3,5,7] and ls==[sp.Rational(1,8),sp.Rational(1,16),sp.Rational(1,24)]
    return rs,ls


def hashes():
    paths=[Path(__file__),*(ROOT/'validation/cln3d'/n for n in ['CLN_APhi.py','CLN_T_Omega.py','CLN_AT.py','_iccg.py'])]
    return {str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_text(encoding='utf-8-sig').encode()).hexdigest() for p in paths}


def run(output):
    ngsolve.SetNumThreads(4);sources=hashes();rs,ls=oracle()
    out={'scope':'Three-stage linear internal round-wire s0=0; h/p/quadrature diagnosis',
         'oracle':'sqrt(u) I0(sqrt(u))/(2 I1(sqrt(u))); u=s*mu*sigma*r^2, Z normalized by R0',
         'oracle_reference':'https://dlmf.nist.gov/10.33.E1','oracle_R_normalized':[str(x) for x in rs],
         'oracle_L_normalized':[str(x) for x in ls], 'source_sha256_lf':sources,
         'runtime':{'ngsolve':ngsolve.__version__},'cases':[]}
    specs=[(1,4,.003),(1,8,.003),(2,4,.003),(2,8,.003),(3,8,.003),(4,8,.003),(2,8,.0015),(3,8,.0015)]
    for order,bonus,h in specs:
        for module in [CLN_APhi,CLN_T_Omega,CLN_AT]:
            row={'formulation':module.__name__.split('.')[-1],'order':order,'bonus_intorder':bonus,'maxh':h,'stages':3};start=time.perf_counter()
            log=io.StringIO()
            with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                try:
                    with ngsolve.TaskManager():d=module.run(maxh=h,order=order,stages=3,bonus_intorder=bonus)
                    base=1/(np.pi*d['params']['r']**2*d['params']['sigma']*d['params']['h'])
                    tau=d['params']['mu']*d['params']['sigma']*d['params']['r']**2
                    er=abs(np.asarray(d['Rn'])/(base*np.asarray(rs,dtype=float))-1)
                    el=abs(np.asarray(d['Ln'])/(base*tau*np.asarray(ls,dtype=float))-1)
                    assert np.all(np.isfinite(er)) and np.all(np.isfinite(el))
                    row.update(status='pass',R_relative_error=er.tolist(),L_relative_error=el.tolist(),result=d)
                    if order==3 and h==.0015:assert max(max(er),max(el))<.002
                except RuntimeError as e:row.update(status='load_or_solver_failure',error=str(e))
            row['seconds']=time.perf_counter()-start;row['execution_log']=log.getvalue().replace(str(ROOT)+'\\', '');out['cases'].append(row)
            output.write_text(json.dumps(out,indent=2),encoding='utf-8')
            print(row['formulation'],order,bonus,h,row['status'],row.get('R_relative_error',row.get('error')),flush=True)
    assert sources==hashes(),'Sources changed during sweep'
    fine=[v for v in out['cases'] if v['order']==3 and v['maxh']==.0015]
    assert len(fine)==3 and all(v['status']=='pass' for v in fine)
    out['acceptance']='All three formulations: seven elements at order3,maxh.0015 within0.2% of independently extracted Bessel coefficients'
    output.write_text(json.dumps(out,indent=2),encoding='utf-8')


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--output',required=True);args=a.parse_args();run(Path(args.output))

