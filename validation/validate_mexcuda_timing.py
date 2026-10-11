"""Portable gate for formal full-CLN mexcuda timings, including startup and transfers."""
import copy
import json
import math
from pathlib import Path
from unittest.mock import patch
import mexcuda_host_timing as monitor
from validate_scalar_cln import sources,verify

ROOT=Path(__file__).resolve().parents[1]


def check(data,host,controls,initialization):
    assert data['complete'] is host['complete'] is controls['complete'] is initialization['complete'] is True
    sources(data,20);sources(host,2);sources(controls,3)
    assert host['returncode']==0 and 0<=host['preflight_cpu_busy_fraction']<=.05
    assert host['preflight_gpu']['utilization_percent']==0
    assert not any(host['foreign_compute_job_samples'])
    assert not host.get('monitor_errors')
    assert host['observed_owned_processes']
    assert len({(p['pid'],p['creation']) for p in host['observed_owned_processes']})==len(host['observed_owned_processes'])
    assert host['preflight_gpu']['name']==data['matlab_engine']['device']
    build=json.loads((ROOT/'docs/data/mexcuda_build.json').read_text())
    assert build['binary_sha256']==data['matlab_engine']['binary_sha256']
    assert host['artifact_sha256']['cln_projection_mex.mexw64']==build['binary_sha256']
    assert len(host['artifact_sha256']['radia-5.2.3-cp312-cp312-win_amd64.whl'])==64
    sources(initialization,2)
    assert initialization['binary_sha256']==build['binary_sha256']
    assert 0<=initialization['preflight_cpu_busy_fraction']<=.05 and initialization['preflight_gpu']['utilization_percent']==0
    assert not any(initialization['foreign_compute_job_samples'])
    assert len(initialization['runs'])==4
    for row in initialization['runs']:
        assert row['device']==data['matlab_engine']['device']
        parts=[row[k] for k in ('engine_start_seconds','configuration_seconds','device_start_seconds')]
        assert all(math.isfinite(v) and v>0 for v in parts)
        assert sum(parts)<=row['initialization_seconds']
        assert 0<row['quit_call_seconds']<=row['shutdown_seconds'] and row['matlab_pid']>0
    assert len(controls['cases'])==10 and all(c['rejected'] for c in controls['cases'])
    assert {r['control'] for r in controls['cases']}=={'foreign_handle','wrong_handle_type','wrong_target_size','nonfinite_target','overlapping_writes','cross_element_read_write','index_overflow','double_destroy','stale_handle_after_recreate','stale_handle_after_unload'}
    assert controls['parity']=={'value':[2.,0.,3.,0.],'history':[[1.,0.]]}
    for key in ('start_seconds','device_start_seconds'):
        assert math.isfinite(data['matlab_engine'][key]) and data['matlab_engine'][key]>0
    original=json.loads((ROOT/'docs/data/scalar_cln_study.json').read_text())
    nodal=json.loads((ROOT/'docs/data/scalar_nodal_control.json').read_text())
    for route in ('scalar-mexcuda','scalar-bulk'):
        view=copy.deepcopy(data)
        view['source_sha256_lf']={n:data['source_sha256_lf'][n] for n in original['source_sha256_lf']}
        for cohort in view['cohorts']:
            cohort['repeats']=[r for r in cohort['repeats'] if r['route'] in ('original','condensed',route)]
            cohort['comparisons']=[r for r in cohort['comparisons'] if r['route'] in ('condensed',route)]
            for row in cohort['repeats']+cohort['comparisons']:
                if row['route']==route:row['route']='scalar-VPhi'
        # The old nodal incompatibility experiment is a mathematical control,
        # not a timing reference on this host. Its source/physics gates remain.
        verify(view,nodal,same_host=False)
    for cohort in data['cohorts']:
        assert len(cohort['repeats'])==16 and len(cohort['comparisons'])==12
        for row in cohort['repeats']:
            if row['route']!='scalar-mexcuda':continue
            assert row['setup_seconds']+sum(s['mex_call_seconds'] for s in row['solves'])<=row['seconds']
            for step in row['solves']:
                assert step['backend']=='mexcuda-local/cpu-fe'
                assert step['projection_backend']=='mexcuda FP64 fused element kernels'
                assert step['physical_integral_order']==13 and step['gpu_dtype']=='float64'
                parts=[step[k] for k in ('device_upload_seconds','mex_internal_projection_seconds','curl_check_seconds','device_download_seconds','matlab_bridge_seconds')]
                assert all(math.isfinite(x) and x>=0 for x in parts)
                assert abs(sum(parts)-step['mex_call_seconds'])<1e-6
                assert step['mex_call_seconds']<=step['recovery_seconds']
                assert 0<=step['mex_setup_transfer_seconds']<=row['setup_seconds']


def ownership_control():
    monitor.OWNED_IDENTITIES.clear()
    parent=dict(ProcessId=10,ParentProcessId=1,Name='python.exe',CreationDate='one')
    child=dict(ProcessId=20,ParentProcessId=10,Name='python.exe',CreationDate='two')
    matlab=dict(ProcessId=30,ParentProcessId=20,Name='MATLAB.exe',CreationDate='three')
    with patch.object(monitor.os,'getpid',return_value=10):
        with patch.object(monitor,'processes',return_value=[parent,child,matlab]):assert monitor.foreign_jobs()==[]
        with patch.object(monitor,'processes',return_value=[parent,matlab]):assert monitor.foreign_jobs()==[]
        reused=dict(matlab,ParentProcessId=999,CreationDate='new')
        with patch.object(monitor,'processes',return_value=[parent,reused]):assert monitor.foreign_jobs()==[dict(name='MATLAB.exe',pid=30,parent=999)]
    monitor.OWNED_IDENTITIES.clear()


if __name__=='__main__':
    ownership_control()
    data=json.loads((ROOT/'docs/data/mexcuda_cln_timing.json').read_text())
    host=json.loads((ROOT/'docs/data/mexcuda_timing_host.json').read_text())
    controls=json.loads((ROOT/'docs/data/mexcuda_timing_controls.json').read_text())
    initialization=json.loads((ROOT/'docs/data/mexcuda_initialization.json').read_text())
    check(data,host,controls,initialization)
    for name in ('busy','overlap','transfer','residual','initialization'):
        bad,h,init=copy.deepcopy(data),copy.deepcopy(host),copy.deepcopy(initialization)
        row=next(r for r in bad['cohorts'][0]['repeats'] if r['route']=='scalar-mexcuda')
        if name=='busy':h['preflight_cpu_busy_fraction']=.5
        elif name=='overlap':h['foreign_compute_job_samples'].append(['python.exe'])
        elif name=='transfer':row['solves'][0]['device_upload_seconds']=-1
        elif name=='residual':row['solves'][0]['original_physical_residual']=.01
        else:init['runs'][0]['initialization_seconds']=0
        try:check(bad,h,controls,init)
        except AssertionError:pass
        else:raise AssertionError('Accepted corruption: '+name)
    print('PASS formal mexcuda timing, startup/transfer accounting, full physical parity and five corruptions')
