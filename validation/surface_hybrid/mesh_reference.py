"""Sparse mixed A-phi reference for mesh uncertainty; no dense mode truncation."""
import argparse,json,math,time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from ngsolve import Mesh,H1,HCurl,GridFunction,BilinearForm,LinearForm,grad,curl,dx,SetNumThreads,TaskManager
from netgen.occ import Box,OCCGeometry,Z
A=.01;HEIGHT=.015;MU=4e-7*math.pi;SIGMA=1e6;TAU=MU*SIGMA*A*A

def csr(b):return sp.csr_matrix(b.mat.CSR()).copy()

def run(h,freq):
    shape=Box((-A,-A,0),(A,A,HEIGHT))-Box((.002,.002,HEIGHT/2),(.012,.012,HEIGHT+.002))
    shape.faces.name='surface';shape.faces.Min(Z).name='in';shape.faces.Max(Z).name='out'
    mesh=Mesh(OCCGeometry(shape).GenerateMesh(maxh=h));va=HCurl(mesh,order=1,nograds=False,dirichlet='surface|in|out');vp=H1(mesh,order=2,dirichlet='in|out')
    fa=np.fromiter(va.FreeDofs(),bool,va.ndof);fp=np.fromiter(vp.FreeDofs(),bool,vp.ndof)
    ua,wa=va.TnT();up,wp=vp.TnT();kp=BilinearForm(vp);kp+=SIGMA*grad(up)*grad(wp)*dx;kp.Assemble();ks=csr(kp)[fp][:,fp].tocsc();lu=sla.splu(ks)
    gf=GridFunction(vp);gf.Set(HEIGHT,definedon=mesh.Boundaries('in'));lift=gf.vec.CreateVector();lift.data=kp.mat*gf.vec;b=-lift.FV().NumPy()[fp].copy();gf.vec.FV().NumPy()[fp]=lu.solve(b)
    ka=BilinearForm(va);ka+=curl(ua)*curl(wa)/MU*dx;ka.Assemble();k=csr(ka)[fa][:,fa].tocsr()
    ma=BilinearForm(va);ma+=SIGMA*ua*wa*dx;ma.Assemble();m=csr(ma)[fa][:,fa].tocsr()
    lf=LinearForm(va);lf+=-SIGMA*grad(gf)*wa*dx;lf.Assemble();f=lf.vec.FV().NumPy()[fa].copy()
    mixed=va*vp;(a,p),(b,q)=mixed.TnT();bf=BilinearForm(mixed);bf+=SIGMA*a*grad(q)*dx;bf.Assemble();c=csr(bf)[va.ndof:,:va.ndof][fp][:,fa].tocsr()
    gm,h1=va.CreateGradient();freeh=np.fromiter(h1.FreeDofs(),bool,h1.ndof);g=sp.csr_matrix(gm.CSR()).copy()[fa][:,freeh];gg=g@g.T;scale=abs(k.diagonal()).max()/gg.diagonal().max()
    kernel_defect=float(sla.norm(k@g)/(sla.norm(k)*sla.norm(g)));load_defect=float(np.linalg.norm(g.T@f)/(sla.norm(g)*np.linalg.norm(f)));assert kernel_defect<1e-10 and load_defect<1e-10
    # Aphi gradients satisfy the mixed constraint through phi=-grad-coordinate.
    # Adding GG^T chooses the complement while the original residual is checked.
    kg=k+scale*gg;static=sla.spsolve(kg,f);h0=float(f@static);assert np.linalg.norm(k@static-f)/np.linalg.norm(f)<1e-7
    vals=[];residuals=[];check=[]
    for index,hz in enumerate(freq):
        s=2j*np.pi*hz;system=sp.bmat([[kg+s*m,c.T],[c,ks/s]],format='csc');rhs=np.r_[f,np.zeros(ks.shape[0])];scaling=sp.diags(1/np.sqrt(abs(system.diagonal())));sol=scaling@sla.spsolve((scaling@system@scaling).tocsc(),scaling@rhs)
        aa=sol[:len(f)];phi=sol[len(f):]/s;r1=(k+s*m)@aa+s*(c.T@phi)-f;r2=c@aa+ks@phi
        defect=max(np.linalg.norm(r1)/np.linalg.norm(f),np.linalg.norm(r2)/max(np.linalg.norm(c@aa),1e-300));assert defect<1e-6
        residuals.append(float(defect));vals.append(2j*np.pi*hz*TAU*(f@aa)/h0);print("frequency",h,hz,"residual",defect,flush=True)
        if index in [0,len(freq)-1]:
            other=sp.bmat([[k+.1*scale*gg+s*m,c.T],[c,ks/s]],format='csc');other_scale=sp.diags(1/np.sqrt(abs(other.diagonal())));alt=other_scale@sla.spsolve((other_scale@other@other_scale).tocsc(),other_scale@rhs)
            check.append(float(abs((f@alt[:len(f)])/(f@aa)-1)))
    assert max(check)<1e-6
    z=np.asarray(vals)
    return {'maxh':h,'ne':mesh.ne,'free_A':int(fa.sum()),'static_H':h0,'kernel_defect':kernel_defect,'load_defect':load_defect,'original_residual_max':max(residuals),'penalty_invariance_max':max(check),'reference_real':z.real.tolist(),'reference_imag':z.imag.tolist()}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args();SetNumThreads(4)
    freq=np.asarray([1e3,1e4,1e5,1e6]);out={'scope':'Sparse full mixed A-phi reference; physical susceptibility and normalized reaction; internal 3D model','frequency_hz':freq.tolist(),'meshes':[]}
    for h in [.003,.002,.0015]:
        t=time.perf_counter()
        with TaskManager():v=run(h,freq)
        v['seconds']=time.perf_counter()-t;out['meshes'].append(v);Path(a.output).write_text(json.dumps(out,indent=2),encoding='utf-8');print(h,v['ne'],v['static_H'],v['original_residual_max'],v['seconds'],flush=True)


