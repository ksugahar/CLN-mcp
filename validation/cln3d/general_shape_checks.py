"""Coarse notched-conductor quadrature and public ICCG robustness checks."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from general_shape import Model, recursion, SIGMA, source_identity
from general_shape_t import CurrentModel, run_current
from _iccg import solve_iccg, gradient_kernel


def identities():
    out=source_identity()
    for name in ["general_shape_checks.py","_iccg.py"]:
        path=Path(__file__).with_name(name)
        out["validation/cln3d/"+name]=hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    import ngsolve as ng
    ng.SetNumThreads(4)
    before=identities();start=time.perf_counter()
    values=[]
    with ng.TaskManager():
        for bonus in [6,8]:
            model=Model(.006,1,bonus=bonus)
            primary=[recursion(model,s0,4) for s0 in [0.,2*np.pi*1e4]]
            groups={"A-phi":primary}
            for name in ["A-T","T-Omega"]:
                current=CurrentModel(model,name)
                groups[name]=[run_current(current,s0,4) for s0 in [0.,2*np.pi*1e4]]
            values.append({name:[np.array([v for pair in zip(run["Rhat"],run["L"]) for v in pair])
                                 for run in runs] for name,runs in groups.items()})
        differences={name:[float(np.max(abs(b/a-1))) for a,b in zip(values[0][name],values[1][name])]
                     for name in values[0]}
        assert max(value for rows in differences.values() for value in rows)<1e-8
        # Independent fixed-shift ICCG checks only; production uses explicit sparse LU.
        p,q=model.vp.TnT()
        form=ng.BilinearForm(model.vp)
        form+=SIGMA*ng.grad(p)*ng.grad(q)*model.dx;form.Assemble()
        gf=ng.GridFunction(model.vp);gf.Set(1,definedon=model.mesh.Boundaries("in"))
        rhs=gf.vec.CreateVector();rhs.data=-form.mat*gf.vec
        solved,scalar_info=solve_iccg(form,rhs,model.vp,gf=gf,shift=1.,tol=1e-10,label="notched DC phi")
        difference=ng.grad(solved)-ng.grad(model.phi0)
        dc_defect=float(np.sqrt(ng.Integrate(SIGMA*difference*difference*model.dx,model.mesh)/model.G0))
        lf=ng.LinearForm(model.va)
        lf+=SIGMA*model.e0*model.va.TestFunction()*model.dx;lf.Assemble()
        agf,vector_info=solve_iccg(model.Kform,lf,model.va,
                                  kernel=gradient_kernel(model.va,model.Kform.mat),shift=1.1,tol=1e-10,
                                  label="notched DC A")
        initial=np.zeros(1+model.na+model.np);initial[0]=1
        aa=model.magnetic(initial)
        difference=ng.curl(agf)-ng.curl(model.gf_a(aa))
        magnetic_defect=float(np.sqrt(ng.Integrate(difference*difference*model.dx,model.mesh)/
                                       ng.Integrate(ng.curl(model.gf_a(aa))**2*model.dx,model.mesh)))
        assert scalar_info["converged"] and vector_info["converged"]
        assert max(dc_defect,magnetic_defect)<1e-8
    assert before==identities(),"Sources changed during robustness checks"
    out=dict(complete=True,source_sha256_lf=before,maxh=.006,order=1,seconds=time.perf_counter()-start,
             scope="Coarse quadrature 6/8 and fixed-shift ICCG vs sparse LU only; no all-mesh ICCG claim",
             quadrature_element_relative_changes=differences,
             iccg_phi=scalar_info,iccg_A=vector_info,
             iccg_LU_dc_field_relative=dc_defect,iccg_LU_magnetic_field_relative=magnetic_defect)
    Path(args.output).write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    print("PASS coarse quadrature and ICCG checks",differences,dc_defect,magnetic_defect)


if __name__=="__main__":
    main()
