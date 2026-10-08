"""Washer loop/curl current control, coupled through the total magnetic field.

This is an A-T current-space check of A-phi, not an independent T-Omega
discretization. The temporary topology adapter has the proposed Radia contract.
"""
import math
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla
import ngsolve as ng
from c2_topology_temporary import insulated_loop_currents, natural_magnetic_representative
from c2_washer import csr, SIGMA, MU
from surface_hybrid.ladder_eval import z_type1


def _orth(matrix):
    diagonal = matrix.diagonal()
    return float(np.max(abs(matrix - np.diag(diagonal)) /
                        np.sqrt(np.outer(diagonal, diagonal))))


def physical_type1(k, m, f, s0):
    """Energy-normalized field recursion: magnetic increments / Joule fields."""
    factor = la.cho_factor(k + s0*m)
    solve = lambda b: la.cho_solve(factor, b)
    a = solve(f); electric = np.zeros(len(a)); modes = []; currents = []; elements = []
    for _ in range(4):
        ell = float(a @ (k+s0*m) @ a)
        electric += a/ell
        resistance = 1/float(electric @ m @ electric)
        assert min(ell, resistance) > 0
        elements.extend([ell, resistance]); modes.append(a.copy()); currents.append(electric.copy())
        a -= resistance*solve(m@electric)
    q = np.column_stack(modes); kp = q.T@k@q; mp = q.T@m@q; bp = q.T@f
    orth = max(_orth(kp+s0*mp), _orth(np.array(currents)@m@np.array(currents).T))
    assert orth < 1e-8
    full = solve(f); small = la.solve(kp+s0*mp, bp, assume_a='pos'); tq = []; rq = []
    for _ in range(9):
        tq.append(float(f@full)); rq.append(float(bp@small))
        full = -solve(m@full); small = -la.solve(kp+s0*mp, mp@small, assume_a='pos')
    errors = [abs(x-y)/max(abs(x),1e-300) for x,y in zip(tq,rq)]
    assert max(errors[:8]) < 1e-7
    rows = []
    for hz in [1e3,1e4,1e5,1e6]:
        s = 2j*math.pi*hz; z = s*(f@la.solve(k+s*m,f,assume_a='sym'))
        galerkin = s*(bp@la.solve(kp+s*mp,bp,assume_a='sym'))
        ladder = z_type1(elements,s0,s)
        assert abs(ladder/galerkin-1) < 1e-8
        rows.append(dict(hz=hz,Z=[z.real,z.imag],ladder=[ladder.real,ladder.imag],
                         reduction_error=float(abs(ladder/z-1))))
    return dict(s0=s0,elements=elements,orthogonality=orth,frequency=rows,
                flux_taylor_full=tq,flux_taylor_reduced=rq,flux_taylor_relative=errors,
                magnetic_modes=q.tolist(),electric_modes=np.array(currents).T.tolist(),
                K_reduced=kp.tolist(),M_reduced=mp.tolist(),f_reduced=bp.tolist())


def evaluate(state, primary, quotient_tolerance=1e-11):
    mesh=state['mesh']; A=state['A']; fa=state['fa']; K=state['K']; Kg=state['Kg']; f=state['f']; D=state['D']
    loops=insulated_loop_currents(mesh,conductor_regions='conductor',sigma=SIGMA,
                                insulating_boundaries='surface')
    assert loops.topology.b1 == 1
    current=loops.space; fj=np.array(list(current.FreeDofs()),bool)
    scalar=ng.L2(mesh,order=0,definedon=mesh.Materials('conductor')); fs=np.array(list(scalar.FreeDofs()),bool)
    j,w=current.TnT(); loss=ng.BilinearForm(current);loss+=j*w/SIGMA*D;loss.Assemble()
    raw=csr(loss)[fj][:,fj].toarray()
    pair=current*scalar;(j,p),(w,q)=pair.TnT();div=ng.BilinearForm(pair);div+=ng.div(j)*q*D;div.Assemble()
    dmat=csr(div)[current.ndof:,:current.ndof][fs][:,fj].toarray(); n=la.null_space(dmat,rcond=1e-10)
    generator=loops.topology.generators[0]
    load=ng.LinearForm(current);load+=generator*current.TestFunction()*D;load.Assemble()
    gamma=load.vec.FV().NumPy().copy()[fj]; zero=la.null_space((gamma@n)[None,:])
    basis=np.column_stack([loops.coefficients[fj,0],n@zero]);r=basis.T@raw@basis
    assert la.norm(gamma@basis-np.r_[1.,np.zeros(len(r)-1)])<1e-8
    assert len(r)==loops.divergence_free_dimension and len(r)-1==loops.curl_dimension
    pair=A*current;(a,j),(v,w)=pair.TnT();bf=ng.BilinearForm(pair);bf+=j*v*D;bf.Assemble()
    coupling=csr(bf)[:A.ndof,A.ndof:][fa][:,fj].toarray()@basis
    gradient=state['G'];gradient_defect=la.norm(gradient.T@coupling)/(la.norm(gradient.toarray())*la.norm(coupling))
    assert gradient_defect<1e-10
    factor=sla.splu(Kg);magnetic=factor.solve(coupling);coil=factor.solve(f)
    l=coupling.T@magnetic;port=coupling.T@coil;self_l=float(f@coil)
    fields=np.column_stack([coil,magnetic]);gram=fields.T@K@fields
    scale=np.sqrt(gram.diagonal());values,vectors=la.eigh(gram/scale[:,None]/scale[None,:])
    keep=values>quotient_tolerance*values[-1]
    qb=fields@(vectors[:,keep]/scale[:,None]/np.sqrt(values[keep])[None,:])
    kt=qb.T@K@qb;F=qb.T@coupling;mt=F@la.solve(r,F.T,assume_a='pos');ft=qb.T@f
    assert la.norm(kt-np.eye(len(kt)))<1e-7
    runs=[physical_type1(kt,mt,ft,s0) for s0 in [0.,2*math.pi*1e4]]
    rows=[]
    for row in primary['runs'][0]['frequency']:
        s=2j*math.pi*row['hz'];induced=-s*la.solve(r+s*l,port,assume_a='sym')
        z=s*self_l+s*(port@induced)
        omitted=-s*la.solve(r[1:,1:]+s*l[1:,1:],port[1:],assume_a='sym')
        zo=s*self_l+s*(port[1:]@omitted);power=float(np.real(induced.conj()@r@induced))
        assert abs(power/z.real-1)<1e-8
        pencil_z=s*(ft@la.solve(kt+s*mt,ft,assume_a='sym'))
        assert abs(pencil_z/z-1)<1e-9
        rows.append(dict(hz=row['hz'],Z=[z.real,z.imag],joule=power,
                         eddy_increment=[(z-s*self_l).real,(z-s*self_l).imag],
                         Aphi_gap=float(abs(z/complex(*row['Z'])-1)),
                         Aphi_loss_gap=float(abs(z.real/row['Z'][0]-1)),
                         loop_current=[induced[0].real,induced[0].imag],
                         omitted_loop_Z=[zo.real,zo.imag],omitted_loop_relative=float(abs(zo/z-1)),
                         omitted_eddy_increment_relative=float(abs((zo-z)/(z-s*self_l))),
                         omitted_loss_relative=float(abs(zo.real/z.real-1))))
    cuts=[]
    for angle in [0.,math.pi/3]:
        total=0.;nr,nz=100,50
        for radius in .008+(np.arange(nr)+.5)*.006/nr:
            for z in .0085+(np.arange(nz)+.5)*.003/nz:
                value=np.array(loops.currents[0](mesh(radius*math.cos(angle),radius*math.sin(angle),z)))
                total+=(value@np.array([-math.sin(angle),math.cos(angle),0]))*.006/nr*.003/nz
        cuts.append(float(total));assert abs(abs(total)-1)<.01
    # A genuinely different representative, with an independently assembled
    # natural scalar system. No vertex reordering / identical-field control.
    scalar=ng.H1(mesh,order=2,definedon=mesh.Materials('conductor'))
    eta=ng.GridFunction(scalar);eta.Set(ng.sin(ng.x/.014)+.3*ng.cos(ng.z/.003))
    gnorm=math.sqrt(ng.Integrate(ng.grad(eta)*ng.grad(eta)*D,mesh))
    hnorm=math.sqrt(ng.Integrate(generator*generator*D,mesh));eta.vec.data*=hnorm/gnorm
    changed=generator+ng.grad(eta)
    first=natural_magnetic_representative(mesh,generator,regions='conductor',order=2)
    second=natural_magnetic_representative(mesh,changed,regions='conductor',order=2)
    gap=math.sqrt(ng.Integrate((first.field-second.field)*(first.field-second.field)*D,mesh)
                  /ng.Integrate(first.field*first.field*D,mesh))
    assert gap<1e-8
    alt=ng.LinearForm(current);alt+=changed*current.TestFunction()*D;alt.Assemble()
    gamma2=alt.vec.FV().NumPy().copy()[fj]
    rnull=n.T@raw@n;v=la.solve(rnull,n.T@gamma2,assume_a='pos');v/=(gamma2@n)@v
    alternative=n@v;loop_gap=la.norm(alternative-basis[:,0])/la.norm(basis[:,0]);assert loop_gap<1e-8
    # Reassemble the coupling of the alternative current; compare full port
    # response and CLN from that independent current-space basis.
    basis2=np.column_stack([alternative,n@zero]);r2=basis2.T@raw@basis2
    pair=A*current;(a,j),(v,w)=pair.TnT();bf2=ng.BilinearForm(pair);bf2+=j*v*D;bf2.Assemble()
    coupling2=csr(bf2)[:A.ndof,A.ndof:][fa][:,fj].toarray()@basis2
    F2=qb.T@coupling2;mt2=F2@la.solve(r2,F2.T,assume_a='pos')
    alt_runs=[physical_type1(kt,mt2,ft,s0) for s0 in [0.,2*math.pi*1e4]]
    element_gaps=[float(np.max(abs(np.array(a['elements'])/b['elements']-1))) for a,b in zip(runs,alt_runs)]
    assert max(element_gaps)<1e-7
    analytic=2*math.pi/(SIGMA*.003*math.log(.014/.008))
    summary=dict(b1=1,current_dimension=len(r),curl_dimension=loops.curl_dimension,
                 loop_R=float(r[0,0]),analytic_loop_R=analytic,loop_R_error=float(abs(r[0,0]/analytic-1)),
                 cut_flux=loops.cut_flux.tolist(),sampled_cut_flux=cuts,
                 divergence_residual=loops.divergence_residual,stationarity_residual=loops.stationarity_residual,
                 source_gradient_defect=float(gradient_defect),self_inductance=self_l,
                 magnetic_quotient_nullity=int(sum(~keep)),quotient_tolerance=quotient_tolerance,
                 frequency=rows,runs=runs,representative=dict(generator_relative_difference=1.,
                 field_relative_gap=gap,loop_relative_gap=float(loop_gap),element_relative_gaps=element_gaps,
                 scalar_residuals=[first.relative_residual,second.relative_residual]),
                 limitation='A-T uses the same A magnetic inverse; this is not an independent T-Omega check.')
    original=dict(K=kt.tolist(),M=mt.tolist(),f=ft.tolist(),R_current=r.tolist(),
                  L_current=l.tolist(),current_port=port.tolist(),coil_self_inductance=self_l,
                  magnetic_spectrum=values.tolist(),current_basis=basis.tolist(),
                  divergence=dmat.tolist(),cut_pairing=gamma.tolist(),R_raw=raw.tolist(),
                  changed_cut_pairing=gamma2.tolist(),changed_current_basis=basis2.tolist(),
                  changed_M=mt2.tolist(),runs=runs)
    return summary,original
