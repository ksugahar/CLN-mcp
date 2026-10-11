"""Run formal mexcuda timings on an idle Windows GPU host, retaining host gates."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

OWNED_IDENTITIES=set()


def processes():
    command="Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,CreationDate | ConvertTo-Json -Compress"
    return json.loads(subprocess.check_output(['powershell','-NoProfile','-Command',command],text=True))


def foreign_jobs():
    rows=processes();owned={os.getpid()}
    # Engine shutdown may outlive its Python parent. Retain observed ownership,
    # pairing PID with creation time so PID reuse never inherits ownership.
    owned|={r['ProcessId'] for r in rows if (r['ProcessId'],r['CreationDate']) in OWNED_IDENTITIES}
    while True:
        more={r['ProcessId'] for r in rows if r['ParentProcessId'] in owned}
        if more<=owned:break
        owned|=more
    # A Windows venv can retain a Python launcher parent of this interpreter.
    parents={r['ProcessId']:r for r in rows};parent=parents.get(os.getpid(),{}).get('ParentProcessId')
    while parent in parents and parents[parent]['Name'].lower().startswith('python'):
        owned.add(parent);parent=parents[parent]['ParentProcessId']
    OWNED_IDENTITIES.update((r['ProcessId'],r['CreationDate']) for r in rows if r['ProcessId'] in owned)
    return [dict(name=r['Name'],pid=r['ProcessId'],parent=r['ParentProcessId'])
            for r in rows if r['ProcessId'] not in owned and
            r['Name'].lower().startswith(('python','matlab','pytest','nvcc'))]


def gpu():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    values=raw.split(', ')
    if len(values)!=5:raise RuntimeError('Exactly one GPU required')
    return dict(name=values[0],driver=values[1],total_mib=int(values[2]),used_mib=int(values[3]),utilization_percent=int(values[4]))


def cpu_times():
    values=[ctypes.c_ulonglong() for _ in range(3)]
    if not ctypes.windll.kernel32.GetSystemTimes(*(ctypes.byref(v) for v in values)):
        raise ctypes.WinError()
    return [v.value for v in values]


def run(directory,output):
    attempts=[]
    for attempt in range(20):
        before=cpu_times();time.sleep(3);after=cpu_times()
        delta=[b-a for a,b in zip(before,after)]
        busy=1-delta[0]/(delta[1]+delta[2])
        pre=gpu();jobs=foreign_jobs()
        attempts.append(dict(cpu_busy_fraction=busy,gpu=pre,foreign_jobs=jobs))
        print(json.dumps(dict(idle_probe=attempt,cpu_busy_fraction=busy,foreign_jobs=jobs)),flush=True)
        if not jobs and pre['utilization_percent']==0 and busy<=.05:break
        time.sleep(30)
    else:
        output.write_text(json.dumps(dict(complete=False,idle_probes=attempts),indent=2)+'\n')
        raise RuntimeError('Host did not become idle within the preflight window')
    root=Path(__file__).resolve().parents[1]
    result=dict(schema=1,complete=False,idle_probes=attempts,preflight_cpu_busy_fraction=busy,
                preflight_gpu=pre,foreign_compute_job_samples=[jobs],
                artifact_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in directory.iterdir() if p.suffix in ('.whl','.mexw64')},
                monitor_interval_seconds=30,monitor_scope='Python/MATLAB/pytest/nvcc processes; own process tree excluded; PID/parent retained',
                source_sha256_lf={str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_text(encoding='utf-8-sig').encode()).hexdigest()
                    for p in (Path(__file__),root/'validation/mexcuda_cln_study.py')})
    stop=threading.Event()
    def monitor():
        while not stop.wait(30):
            try:result['foreign_compute_job_samples'].append(foreign_jobs())
            except Exception as error:
                result.setdefault('monitor_errors',[]).append(type(error).__name__+': '+str(error))
    thread=threading.Thread(target=monitor);thread.start()
    begin=time.perf_counter()
    try:
        with (directory/'formal.log').open('w',encoding='utf8') as log:
            code=subprocess.call([sys.executable,'-u',str(root/'validation/mexcuda_cln_study.py'),
                '--directory',str(directory),'--output',str(directory/'formal.json')],stdout=log,stderr=subprocess.STDOUT)
        result['whole_experiment_seconds']=time.perf_counter()-begin
        result['returncode']=code
    finally:
        stop.set();thread.join()
    result['foreign_compute_job_samples'].append(foreign_jobs())
    result['observed_owned_processes']=[dict(pid=pid,creation=creation) for pid,creation in sorted(OWNED_IDENTITIES)]
    result['postflight_gpu']=gpu()
    result['complete']=code==0 and not any(result['foreign_compute_job_samples']) and not result.get('monitor_errors')
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    if not result['complete']:raise RuntimeError('Formal timing failed or overlapped foreign work')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.directory,args.output)
