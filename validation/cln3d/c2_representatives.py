"""Export independently assembled h and h+grad(eta) natural systems for C2.

Temporary proposed-Radia API control. This validates the natural magnetic
representative, not an independent full coil-driven T-Omega discretization.
"""
import hashlib,json,math
from pathlib import Path
import numpy as np
import ngsolve as ng
from c2_washer import run,csr
from c2_topology_temporary import region_cohomology,natural_magnetic_representative,insulated_loop_currents

ROOT=Path(__file__).resolve().parents[2]


def packed(matrix):
    matrix=matrix.tocsr().copy()
    return dict(shape=list(matrix.shape),data=matrix.data.tolist(),
                indices=matrix.indices.tolist(),indptr=matrix.indptr.tolist())


def produce():
    _,state=run(return_state=True);mesh=state['mesh'];dx=state['D']
    topology=region_cohomology(mesh,regions='conductor');h=topology.generators[0]
    space=ng.H1(mesh,order=2,definedon=mesh.Materials('conductor'))
    eta=ng.GridFunction(space);eta.Set(ng.sin(ng.x/.014)+.3*ng.cos(ng.z/.003))
    norm_h=ng.Integrate(h*h*dx,mesh);norm_g=ng.Integrate(ng.grad(eta)*ng.grad(eta)*dx,mesh)
    eta.vec.data*=math.sqrt(norm_h/norm_g);h2=h+ng.grad(eta)
    difference=math.sqrt(ng.Integrate((h2-h)*(h2-h)*dx,mesh)/norm_h)
    assert difference>.5
    first=natural_magnetic_representative(mesh,h,regions='conductor',order=2)
    second=natural_magnetic_representative(mesh,h2,regions='conductor',order=2)
    assert len(first.omega.vec)==len(eta.vec)==len(second.omega.vec)
    matrix=csr(type('Form',(),{'mat':first.matrix})()).toarray()
    matrix2=csr(type('Form',(),{'mat':second.matrix})()).toarray()
    rhs=np.array(first.rhs.FV().NumPy()).copy();rhs2=np.array(second.rhs.FV().NumPy()).copy()
    vector=eta.vec.FV().NumPy().copy();mask=first.free_dofs
    assert np.linalg.norm(matrix-matrix2)<1e-12*np.linalg.norm(matrix)
    assert np.linalg.norm((rhs2-rhs+matrix@vector)[mask])<1e-9*np.linalg.norm(rhs[mask])
    fields_gap=math.sqrt(ng.Integrate((first.field-second.field)*(first.field-second.field)*dx,mesh)
                         /ng.Integrate(first.field*first.field*dx,mesh))
    assert fields_gap<1e-8
    loops=insulated_loop_currents(mesh,conductor_regions='conductor',sigma=1e6,insulating_boundaries='surface')
    cuts=[]
    for angle in [0.,math.pi/3]:
        positions=[];values=[];normal=[-math.sin(angle),math.cos(angle),0.]
        for radius in .008+(np.arange(100)+.5)*.006/100:
            for z in .0085+(np.arange(50)+.5)*.003/50:
                point=[radius*math.cos(angle),radius*math.sin(angle),z];positions.append(point)
                values.append(list(loops.currents[0](mesh(*point))))
        area=.006/100*.003/50
        flux=float(np.sum(np.array(values)@normal)*area)
        assert abs(abs(flux)-1)<.01
        cuts.append(dict(angle=angle,positions=positions,J_A_per_m2=values,
                         normal=normal,area_weight_m2=area,flux_A=flux))
    return dict(complete=True,ne=mesh.ne,generator_relative_difference=difference,
                field_relative_gap=fields_gap,eta=vector.tolist(),free=mask.tolist(),
                h_D=topology.generators[0].vec.FV().NumPy().copy().tolist(),
                periods=topology.periods.tolist(),cochains=topology.cochains.tolist(),cycles=topology.cycles.tolist(),
                parent_vertices=topology.parent_vertices.tolist(),parent_edge_dofs=topology.parent_edge_dofs.tolist(),
                current_free=np.array(list(loops.space.FreeDofs()),bool).tolist(),
                j_loop=loops.coefficients[:,0].tolist(),cuts=cuts,
                mesh_points=[list(point.p) for point in mesh.ngmesh.Points()],
                tetrahedra=[dict(vertices=[int(v.nr) for v in e.vertices],material=e.mat) for e in mesh.Elements(ng.VOL)],
                S=packed(csr(type('Form',(),{'mat':first.matrix})())),
                S_changed=packed(csr(type('Form',(),{'mat':second.matrix})())),
                rhs=rhs.tolist(),rhs_changed=rhs2.tolist(),
                Omega=first.omega.vec.FV().NumPy().copy().tolist(),
                Omega_changed=second.omega.vec.FV().NumPy().copy().tolist(),
                source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [Path(__file__),Path(__file__).with_name('c2_washer.py'),
                                         Path(__file__).with_name('c2_topology_temporary.py')]})


if __name__=='__main__':
    ng.SetNumThreads(4)
    with ng.TaskManager():result=produce()
    (ROOT/'docs/data/c2_representatives.json').write_text(json.dumps(result,separators=(',',':'))+'\n',encoding='utf-8',newline='\n')
    print('PASS independently assembled C2 changed-representative export')
