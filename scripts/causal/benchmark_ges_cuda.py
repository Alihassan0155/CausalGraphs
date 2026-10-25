"""Bounded paired CPU/CUDA GES pilots; never full discovery."""
import argparse
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time
import discover as b


def main():
    p=argparse.ArgumentParser();p.add_argument('--prefix',required=True);args=p.parse_args()
    if not args.prefix or not all(c.isalnum() or c in '_-' for c in args.prefix):p.error('Simple prefix required')
    root=b.BASE/'ges_cuda_pilot';report=root/f'{args.prefix}_report'
    if report.exists():raise SystemExit('Report exists')
    report.mkdir(parents=True);results=[]
    for nodes in (20,40,80):
        pair=[]
        for backend in ('cpu','cuda'):
            name=f'{args.prefix}_n{nodes}_{backend}';folder=root/name
            command=[sys.executable,'-u',str(Path(__file__).with_name('discover_ges_cuda.py')),'--run-id',name,'--backend',backend,'--nodes',str(nodes),'--cells','2000','--seed','20','--bic-penalty','2']
            start=time.monotonic()
            with open(report/f'{name}.log','x') as stream:
                child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
                try:child.wait(timeout=60)
                except BaseException:b.stop_worker_tree(child);raise
            if child.returncode:raise RuntimeError(f'Failed {name}')
            stat=json.loads((folder/'statistics.json').read_text());stat.update(nodes=nodes,backend=backend,process_seconds=time.monotonic()-start)
            with open(folder/'record.pkl','rb') as stream:record=pickle.load(stream)
            pair.append((stat,record));print(name,stat['algorithm_seconds'],flush=True)
        a,c=pair[0][1],pair[1][1]
        comparison=dict(nodes=nodes,skeleton_equal=bool(b.np.array_equal(a['G'].graph!=0,c['G'].graph!=0)),
            endpoints_equal=bool(b.np.array_equal(a['G'].graph,c['G'].graph)),
            forward_updates_equal=repr(a['update1'])==repr(c['update1']),backward_updates_equal=repr(a['update2'])==repr(c['update2']),
            score_difference=float(c['score']-a['score']),cpu=pair[0][0],cuda=pair[1][0])
        results.append(comparison);b.atomic_json(report/'summary.json',results)
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
