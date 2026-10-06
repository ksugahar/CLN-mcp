"""Matched-rank nonlinear bases, including protected CLN + auxiliary surface modes.

The nonlinear full-model trajectory trains POD; half/1.5 amplitudes are held
out. CLN linear bases use the actual physical source. Enrichment additionally
uses auxiliary Steklov loads, retains the two bulk CLN directions, and rank
reveals residual directions. Diagonalization keeps nonlinear controlled sources;
it does not turn the nonlinear model into independent linear Foster branches.
"""
import argparse, ast, contextlib, hashlib, io, json, sys, time
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as spla
from scipy.linalg import eigh
ROOT=Path(__file__).resolve().parents[2]


def load_model(h):
    # Reuse definitions preceding the original standalone experiment driver.
    # AST selection changes no numerical expression and executes no old driver.
    path=Path(__file__).with_name('fp_cln_surface.py');tree=ast.parse(path.read_text(encoding='utf-8-sig'));nodes=[]
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='res' for t in node.targets):break
        nodes.append(node)
    else:raise RuntimeError('Standalone model/driver boundary not found')
    previous=sys.argv;sys.argv=['fp_model','2',str(h),'unused'];g={'__name__':'fp_model'}
    try:exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),g)
    finally:sys.argv=previous
    return g


def run(h):
    source_paths=[Path(__file__),Path(__file__).with_name('fp_cln_surface.py'),Path(__file__).with_name('fixed_point.py')]
    identities=lambda:{str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_text(encoding='utf-8-sig').encode()).hexdigest() for p in source_paths}
    source_versions=identities()
    g=load_model(h);ka=g['Kfull_matrix'](g['surface_K']('exact'));m=g['Mr'];f=g['fr'];dt=g['DT'];nstep=g['NSTEP_PER'];baseinput=g['uu'].copy();c0=g['c0'];s0=g['S0'];nr=g['nR']
    energy=(ka+s0*m).toarray();chol=np.linalg.cholesky(energy)
    def simulate(input_,q=None):
        kq=ka if q is None else q.T@(ka@q);mq=m if q is None else q.T@(m@q);fq=f if q is None else q.T@f
        if q is None:factor=spla.splu((kq+mq/dt).tocsc());solve=factor.solve;n=nr
        else:inv=np.linalg.inv(kq+mq/dt);solve=lambda v:inv@v;n=q.shape[1]
        y=np.zeros(n);trace=[];states=[];its=[];joule=0.
        for step,un in enumerate(input_):
            old=y.copy();rhs=fq*un+mq@y/dt
            y,it=g['anderson'](lambda w:solve(rhs-(g['g_of'](w) if q is None else q.T@g['g_of'](q@w))),y)
            field=y if q is None else q@y;trace.append(f@field+c0*un);states.append(field.copy());its.append(it)
            if step>=nstep:dy=y-old;joule+=float(dy@(mq@dy))/dt
        return np.asarray(trace),joule,np.column_stack(states),max(its)
    train,jtrain,snap,trainits=simulate(baseinput)
    old,_,_,jold=g['run_full'](g['surface_K']('exact'))
    assert np.max(abs(old-train))/np.max(abs(old))<1e-9 and abs(jold/jtrain-1)<1e-9
    left,sv,_=np.linalg.svd(chol.T@snap,full_matrices=False);pod=np.linalg.solve(chol.T,left)
    bulk,_=g['cln_basis'](g['surface_K']('exact'),2);assert bulk.shape[1]==2
    factor=spla.splu((ka+s0*m).tocsc());extras=[]
    for j in range(2,10):
        fe=np.zeros(nr);fe[g['Gr']]=g['Bgg']@g['Vs'][:,j];w=factor.solve(fe)
        for step in range(6):extras.append(w);w=factor.solve(m@w)
    white=chol.T@np.column_stack(extras);white/=np.linalg.norm(white,axis=0)
    wb=chol.T@bulk;residual=white-wb@(wb.T@white);extra,svextra,_=np.linalg.svd(residual,full_matrices=False)
    assert svextra[16]>svextra[0]*1e-11
    cols=[]
    for shift in s0*np.geomspace(.01,100,40):cols.append(spla.splu((ka+shift*m).tocsc()).solve(f))
    multipoint=g['energy_basis'](cols,ka)
    def stable_span(n):
        w=factor.solve(f);vectors=[]
        for j in range(n):
            before=float(w@energy@w)
            for _ in range(2):
                for v in vectors:w-=v*float(v@energy@w)
            after=float(w@energy@w)
            if after < before*1e-26:break
            w/=np.sqrt(after);vectors.append(w.copy());w=factor.solve(m@w)
        return np.column_stack(vectors)
    bases={};basis_audit=[]
    for n in [9,19]:
        single,_=g['cln_basis'](g['surface_K']('exact'),2*(n-1))
        for name,q in [('single CLN',single),('stabilized actual-source Krylov',stable_span(n)),('multipoint actual source',multipoint[:,:n]),('protected bulk2 + auxiliary surface',np.column_stack([bulk,np.linalg.solve(chol.T,extra[:,:n-2])])),('nonlinear training POD',pod[:,:n])]:
            label=f'{name} {n}';basis_audit.append({'label':label,'requested_rank':n,'rank':q.shape[1]})
            if q.shape[1]!=n:continue
            assert np.linalg.norm(q.T@energy@q-np.eye(n))<1e-7,label
            bases[label]=q
    out={'scope':'Nonlinear 2D full discrete surface operator; retained rank matched, actual physical source common, auxiliary loads disclosed',
         'order':2,'h_over_a':h,'nR':nr,'dt':dt,'training_amplitude':g['H0AMP'],'basis_audit':basis_audit,
         'information_contract':'POD uses full nonlinear training trajectory. Linear CLN uses the actual source and fixed linear pencil. Auxiliary enrichment uses eight extra Steklov forcing directions. Offline information differs.',
         'reference':{},'runs':{},'same_basis_modal_checks':{},'POD_relative_singular_values':(sv/sv[0]).tolist()}
    for amplitude in [1.,.5,1.5]:
        input_=amplitude*baseinput
        ref,jref,refstates,itref=(train,jtrain,snap,trainits) if amplitude==1 else simulate(input_)
        g['uu']=input_;out['reference'][str(amplitude)]={'trace':ref.tolist(),'joule':jref,'max_iterations':itref}
        for name,q in bases.items():
            label=f'{name} amplitude {amplitude}';start=time.perf_counter()
            try:
                lt,j,states,it=simulate(input_,q);metric=g['metrics'](lt,ref,j,jref)
                diff=states[:,nstep:]-refstates[:,nstep:];field=float(np.sqrt(np.sum(diff*(energy@diff))/np.sum(refstates[:,nstep:]*(energy@refstates[:,nstep:]))))
                metric.update(rank=q.shape[1],field_energy_rms=field,max_iterations=it,seconds=time.perf_counter()-start)
                out['runs'][label]=metric
                print(label,'rod',metric['rod_linkage_relative_error'],'Joule',metric['joule_relative_error'],'field',field,flush=True)
            except RuntimeError as e:out['runs'][label]={'rank':q.shape[1],'status':'failure','error':str(e)};print(label,'FAILED',str(e),flush=True)
        q=bases['protected bulk2 + auxiliary surface 19'];poles,v=eigh(q.T@(ka@q),q.T@(m@q));assert np.all(poles>0);modal=q@v
        a,j,_,_=simulate(input_,q);b,jb,_,_=simulate(input_,modal)
        check={'trace_relative_difference':float(np.max(abs(a-b))/np.max(abs(a))),'joule_relative_difference':float(abs(j/jb-1))}
        assert max(check.values())<1e-5;out['same_basis_modal_checks'][str(amplitude)]=check
    assert identities()==source_versions,'Sources changed during nonlinear comparison'
    out['source_sha256_lf']=source_versions
    return out


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--h-over-a',type=float,default=.3);a.add_argument('--output',required=True);args=a.parse_args()
    Path(args.output).write_text(json.dumps(run(args.h_over_a),indent=2),encoding='utf-8')

