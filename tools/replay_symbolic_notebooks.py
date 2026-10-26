"""Replay selected portable notebook cells only after source-bound Mathematica records exist."""
from pathlib import Path
import contextlib,io,json,os,sys,base64
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'validation'))
from reference_records import symbolic_record
symbolic_record()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plan={'02_formulations_gauge_boundary': [1], '04_termination_conversion_tiles':[1],
      '09_foster_cauer_conversion_error':[2,4], '16_input_ports_and_passivity':[2,4,6]}
os.chdir(ROOT/'docs')
for name,indices in plan.items():
 path=ROOT/'docs'/(name+'.ipynb');notebook=json.loads(path.read_text(encoding='utf-8'));scope={}
 for i in indices:
  cell=notebook['cells'][i];output=io.StringIO()
  with contextlib.redirect_stdout(output):exec(compile(''.join(cell['source']),str(path)+':cell'+str(i),'exec'),scope)
  results=[]
  if output.getvalue():results.append({'output_type':'stream','name':'stdout','text':output.getvalue().splitlines(True)})
  for figure in plt.get_fignums():
   data=io.BytesIO();plt.figure(figure).savefig(data,format='png',bbox_inches='tight')
   results.append({'output_type':'display_data','metadata':{},'data':{'image/png':base64.b64encode(data.getvalue()).decode(),'text/plain':['Matplotlib figure']}})
  plt.close('all');cell['outputs']=results;cell['execution_count']=i+1
 notebook['metadata']['symbolic_migration_status']='portable cell replay from source-bound Mathematica output'
 path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
 print('Replayed',name,indices)
