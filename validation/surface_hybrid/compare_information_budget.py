"""Compare terminal fits and field-derived reductions at matched state counts.

Training responses are sampled at exactly the common-source POD frequencies.
All terminal errors use the same discrete pencil. A fitted Foster response has
no inferred field/Joule reconstruction. Timings include fitting only; no total
cost ranking is inferred. Failures are recorded, not accepted as converged fits.
"""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from ngsolve import SetNumThreads, TaskManager
from compare_3d import build, TAU
from positive_foster import response
from scipy.optimize import least_squares
from scipy.special import softmax
ROOT=Path(__file__).resolve().parents[2]


def proper_fit(points,target,n,dc,seed):
    """n finite poles, no ideal-inductor tail: proper equal-state terminal model."""
    poles=np.asarray(seed['poles']);residues=np.asarray(seed['residues'])
    if dc:
        weights=residues/poles;weights/=weights.sum()
        initial=np.r_[np.log(poles),np.log(weights[:-1]/weights[-1])]
        def unpack(theta):
            p=np.exp(theta[:n]);w=softmax(np.r_[theta[n:],0.]);return p,p*w,0.
    else:
        initial=np.r_[np.log(poles),np.log(residues),max(seed.get('DC_offset',0.),0.)]
        def unpack(theta):return np.exp(theta[:n]),np.exp(theta[n:2*n]),float(theta[-1])
    def model(theta,u):
        p,r,d=unpack(theta);return response(p,r,u)+d
    def residual(theta):
        e=(model(theta,points)-target)/abs(target);return np.r_[e.real,e.imag]
    start=time.perf_counter();bounds=(-30,30) if dc else (np.r_[np.full(2*n,-30.),0.],np.r_[np.full(2*n,30.),1e4]);fit=least_squares(residual,initial,bounds=bounds,max_nfev=5000,x_scale='jac',ftol=1e-10,gtol=1e-9,xtol=1e-10)
    if not fit.success:raise RuntimeError('Proper Foster optimizer failed: '+str(fit.message))
    p,r,d=unpack(fit.x)
    info={'poles':p.tolist(),'residues':r.tolist(),'DC_offset':d,'L_tail':0.,'DC_slope_error':float(abs(np.sum(r/p)-1)),
          'optimizer_converged':bool(fit.success),'nfev':int(fit.nfev),'seconds':time.perf_counter()-start,
          'training_relative_rms':float(np.sqrt(np.mean(residual(fit.x)**2))),
          'scope':'proper transfer, n finite poles, no ideal-inductor/infinite-pole tail; local optimizer only'}
    return lambda u:model(fit.x,u),info


def run(maxh, sizes=(4,8)):
    source_paths=[Path(__file__),Path(__file__).with_name('compare_3d.py'),Path(__file__).with_name('positive_foster.py')]
    identities=lambda:{str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_text(encoding='utf-8-sig').encode()).hexdigest() for p in source_paths}
    source_versions=identities()
    SetNumThreads(4)
    start=time.perf_counter()
    with TaskManager():k,m,f,p,r,mesh=build(maxh)
    size='coarse' if maxh==.004 else 'fine' if maxh==.003 else None
    if size is None:raise ValueError('Use .004 or .003 to match stored physical-source field evidence')
    path=ROOT/f'docs/data/same_excitation_3d_{size}.json'
    baseline=json.loads(path.read_text(encoding='utf-8'))
    freq=np.asarray(baseline['frequency_hz']);u=2j*np.pi*freq*TAU
    ref=response(p,r,u);stored=np.asarray(baseline['reference_real'])+1j*np.asarray(baseline['reference_imag'])
    defect=float(np.max(abs(ref/stored-1)));assert defect<1e-8
    train=np.asarray(baseline['field_pod']['training_frequency_hz']);ut=2j*np.pi*train*TAU
    target=response(p,r,ut)
    hold=np.all(abs(freq[:,None]/train[None,:]-1)>1e-9,axis=1)
    low=np.geomspace(.1,100,101);ul=2j*np.pi*low*TAU;lowref=response(p,r,ul)
    transition=(np.sqrt(2/abs(u))>=.3)&(np.sqrt(2/abs(u))<=3)
    out={'scope':'Matched-state terminal comparison; same 31-frequency physical-source POD training grid; linear internal 3D conductor',
         'mesh':mesh,'tau':TAU,'training_frequency_hz':train.tolist(),'frequency_hz':freq.tolist(),
         'low_frequency_hz':low.tolist(),'low_reference_real':lowref.real.tolist(),'low_reference_imag':lowref.imag.tolist(),'reference_real':ref.real.tolist(),'reference_imag':ref.imag.tolist(),
         'reference_vs_stored':defect,'holdout_points':int(hold.sum()),'runs':{},
         'information_contract':'Foster fits use terminal responses at the same 31 training frequencies as POD. Single/multipoint CLN use their stated shifts; solve budgets and field information differ.',
         'cost_scope':'Fit timings exclude reference, seed and full-model generation; no offline-cost ranking',
         'baseline_file':str(path.relative_to(ROOT)).replace('\\','/'),
         'baseline_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    def save(label,z,zlow,params):
        error=abs(z/ref-1);lowerr=abs(zlow/lowref-1)
        assert np.all(np.isfinite(error)) and np.all(np.isfinite(lowerr))
        params.update(relative_error=error.tolist(),low_relative_error=lowerr.tolist(),
                      holdout_high_band_max=float(error[hold & (freq>=1000)].max()),
                      transition_max=float(error[transition].max()),low_band_max=float(lowerr.max()),
                      high_band_max=float(error[freq>=1000].max()),at_1MHz=float(error[-1]))
        out['runs'][label]=params
        print(label,'holdout',params['holdout_high_band_max'],'DC slope',params['DC_slope_error'],'low',params['low_band_max'],flush=True)
    for n in sizes:
        for method in ['ordinary CLN','multipoint CLN','field POD','bulk2 + boundary-response POD']:
            name=f'{method} {n}';v=baseline['runs'][name];ps=np.asarray(v['poles']);rs=np.asarray(v['residues'])
            save(name,response(ps,rs,u),response(ps,rs,ul),{'rank':n,'poles':ps.tolist(),'residues':rs.tolist(),
                 'DC_slope_error':float(abs(np.sum(rs/ps)-1)),'DC_offset':0.,'L_tail':0.,
                 'field_holdout_max':v['holdout_field_high_band_max'],'joule_holdout_max':v['holdout_joule_high_band_max']})
        seed=baseline['runs'][f'multipoint CLN {n}']
        constrained_seed=seed
        for dc in [True,False]:
            name=f'Foster {"DC constrained" if dc else "DC free"} {n}'
            try:
                fitted,info=proper_fit(ut,target,n,dc,seed if dc else constrained_seed)
                if dc:constrained_seed=info.copy()
                info.update(rank=n,field_holdout_max=None,joule_holdout_max=None,
                            representation='Fitted scalar terminal Foster; no physical field modes inferred')
                save(name,fitted(u),fitted(ul),info)
            except RuntimeError as e:out['runs'][name]={'rank':n,'status':'optimizer_failure','error':str(e)};print(name,str(e),flush=True)
    assert identities()==source_versions,'Sources changed during fit comparison'
    out['source_sha256_lf']=source_versions
    out['total_seconds']=time.perf_counter()-start
    return out


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--maxh',type=float,required=True);a.add_argument('--output',required=True);args=a.parse_args()
    Path(args.output).write_text(json.dumps(run(args.maxh),indent=2),encoding='utf-8')

