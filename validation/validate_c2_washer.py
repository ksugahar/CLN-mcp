"""Independent NumPy-only C2 matrix, physical recursion and evidence gates."""
import copy,hashlib,json,math
from pathlib import Path,PureWindowsPath
import numpy as np
from surface_hybrid.ladder_eval import z_type1

ROOT=Path(__file__).resolve().parents[1]


def check_source(name,digest):
    # Native hashes identify the bytes actually executed. Git can change only
    # line endings of the existing shared ladder dependency on checkout;
    # Python itself reads both through universal-newline decoding. Do not
    # accept whitespace edits, changed symbols or any other normalization.
    raw=(ROOT/Path(*PureWindowsPath(name).parts)).read_bytes()
    candidates=[raw]
    if name=='validation/surface_hybrid/ladder_eval.py':
        lf=raw.replace(b'\r\n',b'\n');candidates.extend([lf,lf.replace(b'\n',b'\r\n')])
    assert any(hashlib.sha256(value).hexdigest()==digest for value in candidates),name


def dense(item):
    a=np.zeros(item['shape'])
    for row in range(len(a)):
        lo,hi=item['indptr'][row:row+2]
        a[row,np.array(item['indices'][lo:hi],int)]=item['data'][lo:hi]
    return a


def check_each(actual,reference,tolerance=1e-7):
    actual,reference=np.array(actual),np.array(reference)
    assert actual.shape==reference.shape and np.all(np.isfinite(actual))
    assert np.max(abs(actual-reference)/np.maximum(abs(reference),1e-300))<tolerance


def coefficients(k,m,f,s0,count=9):
    # Normalize the matrix before inversion. Each Taylor coefficient is checked
    # individually, not against the dominant DC coefficient's scale.
    a=k+s0*m;scale=1/np.sqrt(a.diagonal());inverse=np.linalg.inv(a*scale[:,None]*scale[None,:])
    solve=lambda b:scale*(inverse@(scale*b))
    v=solve(f);out=[]
    for _ in range(count):out.append(float(f@v));v=-solve(m@v)
    return out


def check_run(k,m,f,run):
    s0=run['s0'];q=np.array(run['magnetic_modes']);e=np.array(run['electric_modes'])
    kp=q.T@k@q;mp=q.T@m@q;bp=q.T@f
    assert min(run['elements'])>0 and len(run['elements'])==8
    for actual,key in [(kp,'K_reduced'),(mp,'M_reduced'),(bp,'f_reduced')]:
        reference=np.array(run[key]);assert np.linalg.norm(actual-reference)<1e-8*np.linalg.norm(reference)
    energy=kp+s0*mp;loss=e.T@m@e
    for physical in [kp,mp]:
        assert np.linalg.eigvalsh(physical).min()>-1e-10*np.linalg.norm(physical)
    for matrix in [energy,loss]:
        diagonal=matrix.diagonal()
        assert np.max(abs(matrix-np.diag(diagonal))/np.sqrt(np.outer(diagonal,diagonal)))<1e-8
    check_each(energy.diagonal(),run['elements'][::2])
    check_each(1/loss.diagonal(),run['elements'][1::2])
    # Verify the actual alternating field recursion from original matrices.
    assert np.linalg.norm((k+s0*m)@q[:,0]-f)<1e-7*np.linalg.norm(f)
    for index in range(4):
        previous=np.zeros(len(f)) if index==0 else e[:,index-1]
        assert np.linalg.norm(e[:,index]-previous-q[:,index]/run['elements'][2*index])<1e-8*np.linalg.norm(e[:,index])
        if index<3:
            residual=(k+s0*m)@(q[:,index+1]-q[:,index])+run['elements'][2*index+1]*(m@e[:,index])
            assert np.linalg.norm(residual)<1e-7*np.linalg.norm((k+s0*m)@q[:,index])
    full=coefficients(k,m,f,s0);small=coefficients(kp,mp,bp,s0)
    check_each(full,run['flux_taylor_full'],tolerance=2e-6)
    check_each(small,run['flux_taylor_reduced'],tolerance=2e-6)
    check_each(full[:8],small[:8],tolerance=2e-6)
    assert abs(small[0]/run['elements'][0]-1)<1e-8
    assert abs(full[0]/run['elements'][0]-1)<1e-8
    for row in run['frequency']:
        s=2j*math.pi*row['hz'];z=s*(f@np.linalg.solve(k+s*m,f))
        ladder=z_type1(run['elements'],s0,s)
        galerkin=s*(bp@np.linalg.solve(kp+s*mp,bp))
        assert abs(z/complex(*row['Z'])-1)<2e-7
        assert abs(ladder/galerkin-1)<1e-8
        assert abs(ladder/complex(*row['ladder'])-1)<1e-8
        assert abs(abs(ladder/z-1)-row['reduction_error'])<2e-7


def verify(data,matrices):
    assert data['complete'] and data['claim'].startswith('Same-mesh')
    primary=matrices['primary'];k=dense(primary['K']);kg=dense(primary['Kg']);m=dense(primary['m']);g=dense(primary['G']);f=np.array(matrices['f'])
    assert np.linalg.norm(g.T@f)<1e-8*np.linalg.norm(g)*np.linalg.norm(f)
    assert np.linalg.norm(m@g)<1e-8*np.linalg.norm(m)*np.linalg.norm(g)
    mass=dense(primary['M']);c=dense(primary['C']);s=dense(primary['S'])
    assert np.linalg.norm(m-(mass-c.T@np.linalg.solve(s,c)))<1e-8*np.linalg.norm(m)
    # Gauge does not enter physical energies; its action on exported field
    # modes is separately bounded before kg is used as the solvable pencil.
    for run in matrices['primary_runs']:
        q=np.array(run['magnetic_modes'])
        assert np.linalg.norm((kg-k)@q)<1e-7*np.linalg.norm(k@q)
        check_run(kg,m,f,run)
    dual=matrices['dual'];kd,md,fd=[np.array(dual[key]) for key in ['K','M','f']]
    for run in dual['runs']:check_run(kd,md,fd,run)
    basis=np.array(dual['current_basis']);div=np.array(dual['divergence']);gamma=np.array(dual['cut_pairing']);raw=np.array(dual['R_raw'])
    assert np.linalg.norm(div@basis)<1e-8*np.linalg.norm(div)*np.linalg.norm(basis)
    assert np.max(abs(gamma@basis-np.r_[1.,np.zeros(basis.shape[1]-1)]))<1e-8
    assert np.linalg.norm(basis.T@raw@basis-dual['R_current'])<1e-8*np.linalg.norm(dual['R_current'])
    changed=np.array(dual['changed_current_basis']);gamma2=np.array(dual['changed_cut_pairing'])
    assert np.linalg.norm(changed-basis)<1e-8*np.linalg.norm(basis)
    assert np.max(abs(gamma2@changed-np.r_[1.,np.zeros(basis.shape[1]-1)]))<1e-8
    assert np.linalg.norm(np.array(dual['changed_M'])-md)<1e-8*np.linalg.norm(md)
    assert data['filled_hole']['b1']==0 and data['filled_hole']['loop_channels']==0
    assert data['filled_hole']['current_dimension']==data['filled_hole']['curl_dimension']>0
    d=data['dual'];assert d['current_dimension']==d['curl_dimension']+1
    assert max(abs(abs(np.array(d['sampled_cut_flux']))-1))<.01
    analytic=2*math.pi/(1e6*.003*math.log(.014/.008))
    assert abs(d['analytic_loop_R']/analytic-1)<1e-12
    assert abs(d['loop_R']/analytic-1)<.02  # coarse curved RT0, not a convergence claim
    assert d['representative']['generator_relative_difference']>.5
    assert max(d['representative']['element_relative_gaps'])<1e-7
    r,l,b=np.array(dual['R_current']),np.array(dual['L_current']),np.array(dual['current_port'])
    self_l=dual['coil_self_inductance']
    for row in d['frequency']:
        ss=2j*math.pi*row['hz'];j=-ss*np.linalg.solve(r+ss*l,b);z=ss*self_l+ss*(b@j)
        omit=-ss*np.linalg.solve(r[1:,1:]+ss*l[1:,1:],b[1:]);zo=ss*self_l+ss*(b[1:]@omit)
        power=float(np.real(j.conj()@r@j))
        assert abs(z/complex(*row['Z'])-1)<1e-8 and abs(power/z.real-1)<1e-8
        assert abs(power/row['joule']-1)<1e-8
        assert abs(zo/complex(*row['omitted_loop_Z'])-1)<1e-8
        assert abs((z-ss*self_l)/complex(*row['eddy_increment'])-1)<1e-8
        assert abs(abs((zo-z)/(z-ss*self_l))-row['omitted_eddy_increment_relative'])<1e-8
    references=data['axisymmetric'];assert references[0]['oracle']
    assert max(row['relative_error'] for row in references[0]['frequency'])<.002
    assert references[-1]['triangles']>references[-2]['triangles']
    assert len(data['primary'])==3
    assert data['primary'][0]['ne']<data['primary'][1]['ne']==data['primary'][2]['ne']
    assert data['primary'][0]['dofs']<data['primary'][1]['dofs']<data['primary'][2]['dofs']
    for case in data['primary']:
        assert abs(case['self_inductance']/references[-1]['self_inductance']-1-case['self_inductance_signed_gap'])<1e-10
        for row,ref,gap in zip(case['runs'][0]['frequency'],references[-1]['frequency'],case['axisymmetric_comparison']):
            z,zref=complex(*row['Z']),complex(*ref['Z'])
            assert abs(z.real/zref.real-1-gap['loss_signed_gap'])<1e-10
            assert abs(abs(z/zref-1)-gap['full_Z_gap'])<1e-10


def main():
    data=json.loads((ROOT/'docs/data/c2_washer.json').read_text())
    path=ROOT/'docs/data/c2_washer_matrices.json';assert hashlib.sha256(path.read_bytes()).hexdigest()==data['matrix_sha256']
    for name,digest in data['source_sha256'].items():check_source(name,digest)
    matrices=json.loads(path.read_text());verify(data,matrices)
    rep=json.loads((ROOT/'docs/data/c2_representatives.json').read_text());assert rep['complete']
    for name,digest in rep['source_sha256'].items():assert hashlib.sha256((ROOT/Path(*PureWindowsPath(name).parts)).read_bytes()).hexdigest()==digest,name
    s,sc=dense(rep['S']),dense(rep['S_changed']);eta=np.array(rep['eta']);free=np.array(rep['free'],bool)
    rhs,rhsc=np.array(rep['rhs']),np.array(rep['rhs_changed'])
    assert np.max(abs(np.array(rep['periods'])@np.array(rep['h_D'])-1))<1e-8
    assert np.max(abs(np.array(rep['cycles'])@np.array(rep['cochains'])-1))<1e-8
    assert max(abs(np.array(rep['j_loop'])[~np.array(rep['current_free'],bool)]),default=0.)<1e-12
    for cut in rep['cuts']:
        flux=float(np.sum(np.array(cut['J_A_per_m2'])@cut['normal'])*cut['area_weight_m2'])
        assert abs(flux-cut['flux_A'])<1e-12 and abs(abs(flux)-1)<.01
    assert rep['generator_relative_difference']>.5
    assert np.linalg.norm(s-sc)<1e-12*np.linalg.norm(s)
    assert np.linalg.norm((rhsc-rhs+s@eta)[free])<1e-9*np.linalg.norm(rhs[free])
    for b,key in [(rhs,'Omega'),(rhsc,'Omega_changed')]:
        solution=np.zeros(len(b));solution[free]=np.linalg.solve(s[free][:,free],b[free])
        assert np.linalg.norm(solution-rep[key])<1e-8*np.linalg.norm(solution)
    ranks=json.loads((ROOT/'docs/data/c2_rank_controls.json').read_text());assert ranks['complete']
    for name,digest in ranks['source_sha256'].items():assert hashlib.sha256((ROOT/Path(*PureWindowsPath(name).parts)).read_bytes()).hexdigest()==digest,name
    assert ranks['ne']==data['primary'][0]['ne']
    reference=ranks['records'][1]
    for record in ranks['records']:
        spectrum=np.array(record['spectrum'])
        assert sum(spectrum<=record['tolerance']*max(spectrum))==record['nullity']==4
        gap=np.max(abs(np.array(record['elements'])/reference['elements']-1))
        assert gap<1e-8 and abs(gap-record['max_element_relative_gap'])<1e-12
    for record,run in zip(reference['elements'],data['dual']['runs']):
        check_each(record,run['elements'],tolerance=1e-7)
    # Negative control on the load-bearing tail coefficient, using the actual
    # independent checker. A global norm alone would hide this corruption.
    for index in [7,8]:
        bad=copy.deepcopy(matrices);bad['dual']['runs'][0]['flux_taylor_full'][index]*=1.01
        try:check_run(np.array(bad['dual']['K']),np.array(bad['dual']['M']),np.array(bad['dual']['f']),bad['dual']['runs'][0])
        except AssertionError:pass
        else:raise AssertionError(f'q{index} corruption was accepted')
    print('PASS C2 original pencils, physical Type1 fields, contact q0..q7, loops and disclosed FE gaps')


if __name__=='__main__':main()
