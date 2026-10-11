"""Native differential controls for reused FE buffers and compiled forms."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import ngsolve as ng
import numpy as np
from threadpoolctl import threadpool_limits
import cached_cln_study as study
from cln3d.cached_checks import CachedScalar,CachedSweep,PhysicalCheck


def run(output):
    hashes=study.hashes()
    hashes['validation/cached_controls.py']=hashlib.sha256(Path(__file__).read_text(encoding='utf-8-sig').encode()).hexdigest()
    data=dict(schema=1,complete=False,host=socket.gethostname(),source_sha256_lf=hashes,cases=[])
    ng.SetNumThreads(4)
    rng=np.random.default_rng(729)
    with threadpool_limits(limits=1,user_api='blas'),ng.TaskManager():
        for curve in (1,3):
            mesh=study.baseline.CLN_AT.make_mesh(.003,curve)
            for order in (2,3,4):
                for magnetic in (False,True):
                    scalar=CachedScalar(mesh,order,magnetic,8)
                    potential_space=ng.HCurl(mesh,order=order,dirichlet='.*' if magnetic else '')
                    scalar.prepare(potential_space)
                    check=PhysicalCheck(mesh,order,scalar.space.components[0],potential_space,magnetic,8)
                    fields=[ng.GridFunction(scalar.space.components[0]) for _ in range(3)]
                    potential=ng.GridFunction(potential_space)
                    potential.vec.FV().NumPy()[:]=rng.normal(size=potential_space.ndof)
                    for gf in fields:gf.vec.FV().NumPy()[:]=rng.normal(size=gf.space.ndof)
                    recovery=CachedSweep(mesh,order,8,'.*' if magnetic else '')
                    recovery.prepare(scalar.space.components[0])
                    try:recovery.targets(fields[0])
                    except RuntimeError:pass  # an arbitrary flux need not be a curl
                    load=ng.LinearForm(recovery.broken)
                    load+=ng.curl(recovery.broken.TestFunction())*fields[0]*ng.dx(bonus_intorder=8)
                    load.Assemble()
                    curl_load_error=float(np.linalg.norm(load.vec.FV().NumPy()-recovery.load.vec.FV().NumPy())/load.vec.Norm())
                    q,u,uhat=scalar.space.TrialFunction()
                    v,w,what=scalar.space.TestFunction()
                    normal=ng.specialcf.normal(3)
                    reference=ng.BilinearForm(scalar.space,nonassemble=True)
                    reference+=(q*v-u*ng.div(v)-w*ng.div(q))*ng.dx(bonus_intorder=8)
                    reference+=(uhat*v*normal+what*q*normal)*ng.dx(element_boundary=True,bonus_intorder=8)
                    reference.Assemble()
                    arbitrary=ng.GridFunction(scalar.space)
                    arbitrary.vec.FV().NumPy()[:]=rng.normal(size=scalar.space.ndof)
                    old=arbitrary.vec.CreateVector();new=arbitrary.vec.CreateVector()
                    old.data=reference.mat*arbitrary.vec;new.data=scalar.original.mat*arbitrary.vec
                    scalar_operator_error=float(np.linalg.norm(old.FV().NumPy()-new.FV().NumPy())/old.Norm())
                    physical=[];loads=[]
                    # Repeated updates include sign/scale changes and clearing old inputs.
                    for k,(scale,force_scale) in enumerate(((2.,3.),(-.7,-2.),(4.,1.))):
                        previous=fields[1] if k!=2 else None
                        forcing=fields[2] if k!=2 else None
                        try:check.check(fields[0],potential,previous,scale,forcing,force_scale)
                        except RuntimeError:pass  # arbitrary data deliberately violates the equation
                        v0,v1,v2=check.space.TestFunction()
                        form=ng.LinearForm(check.space)
                        expr=fields[0]*ng.curl(v1)+(ng.curl(potential)-(previous if previous is not None else ng.CoefficientFunction((0,0,0))))/scale*ng.curl(v2)
                        if forcing is not None:expr+=force_scale*forcing*v0
                        else:form+=-ng.Cross(ng.CoefficientFunction((0,0,1)),v0.Trace())*ng.specialcf.normal(3)*ng.ds('conductorBND',bonus_intorder=8)
                        form+=expr*ng.dx(bonus_intorder=8);form.Assemble()
                        expected=form.vec.FV().NumPy()
                        physical.append(float(np.linalg.norm(check.form.vec.FV().NumPy()-expected)/np.linalg.norm(expected)))
                        scalar.load.vec.data=scale*(scalar.load_operator.mat*potential.vec)
                        ref=ng.LinearForm(scalar.space)
                        ref+=scale*potential*scalar.test*ng.dx(bonus_intorder=8);ref.Assemble()
                        expected=ref.vec.FV().NumPy()
                        loads.append(float(np.linalg.norm(scalar.load.vec.FV().NumPy()-expected)/np.linalg.norm(expected)))
                    assert max(physical+loads+[curl_load_error,scalar_operator_error])<1e-12
                    row=dict(curve=curve,order=order,magnetic=magnetic,physical_vector_relative_errors=physical,scalar_load_relative_errors=loads,
                             curl_load_relative_error=curl_load_error,scalar_operator_relative_error=scalar_operator_error)
                    data['cases'].append(row);print(json.dumps(row),flush=True)
    assert hashes=={n:hashlib.sha256((study.ROOT/n).read_text(encoding='utf-8-sig').encode()).hexdigest() for n in hashes}
    data['complete']=True
    output.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)

