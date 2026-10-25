"""Bounded Module 4A benchmarks only; never launches a 279-node experiment."""
from pathlib import Path
import subprocess,sys,json
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'graphs/causal_split0'
records=[]
for algorithm in ['pc','boss']:
    for nodes,cells in [(20,2000),(40,2000),(80,2000),(40,10691)]:
        name=f'bench_n{nodes}_cells{cells}_seed20'
        folder=BASE/algorithm/name
        status_path=folder/'status.json'
        if status_path.exists():
            status=json.loads(status_path.read_text())
            if status['status']!='complete':
                raise SystemExit(f'Existing incomplete benchmark: recover explicitly first: {folder}')
        else:
            command=[sys.executable,str(ROOT/'scripts/causal/discover.py'),'--algorithm',algorithm,
              '--run-id',name,'--nodes',str(nodes),'--cells',str(cells),'--seed','20','--time-limit-seconds','90']
            print('BENCHMARK',algorithm,nodes,cells,flush=True)
            subprocess.run(command,cwd=ROOT,check=False)
        status=json.loads(status_path.read_text())
        row={'algorithm':algorithm,'nodes':nodes,'cells':cells,'status':status['status'],'attempt_seconds':status['attempt_seconds']}
        if row['status']=='complete': row.update(json.loads((folder/'statistics.json').read_text()))
        records.append(row)
        pd.DataFrame(records).to_csv(BASE/'small_benchmarks.csv',index=False)
print('Completed bounded benchmarks; see graphs/causal_split0/small_benchmarks.csv')
