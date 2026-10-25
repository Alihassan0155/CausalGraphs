"""90-second bounded GES feasibility runs; never launches 279 nodes."""
import json
from pathlib import Path
import subprocess
import sys
import time
import discover as b


def main():
    results=[]
    root=b.BASE/'ges'
    for nodes in (20,40,80):
        name=f'bench_n{nodes}_cells2000_seed20';out=root/name
        if out.exists():raise SystemExit(f'Benchmark exists: {out}')
        log=root/f'{name}_terminal.log';root.mkdir(parents=True,exist_ok=True)
        start=time.monotonic();peak=0
        with open(log,'x') as stream:
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).with_name('discover_ges.py')),
                '--run-id',name,'--nodes',str(nodes),'--cells','2000','--seed','20','--bic-penalty','2'],stdout=stream,stderr=subprocess.STDOUT)
            timed=False
            try:
                while child.poll() is None:
                    try:
                        process=b.psutil.Process(child.pid);mem=[p.memory_info() for p in [process]+process.children(recursive=True)]
                        peak=max(peak,sum(max(m.rss,getattr(m,'peak_wset',0)) for m in mem))
                    except b.psutil.NoSuchProcess:pass
                    if time.monotonic()-start>=90:
                        timed=True;b.stop_worker_tree(child);break
                    time.sleep(.2)
            finally:
                if child.poll() is None:b.stop_worker_tree(child)
        if timed:
            status=json.loads((out/'status.json').read_text());status.update(status='benchmark_timed_out',budget_seconds=90)
            b.atomic_json(out/'status.json',status)
        result=dict(nodes=nodes,cells=2000,elapsed_seconds=time.monotonic()-start,peak_process_tree_rss_bytes=peak,timed_out=timed,exit_code=child.returncode)
        if not timed:
            if child.returncode:raise RuntimeError(f'Benchmark failed: {log}')
            result['statistics']=json.loads((out/'statistics.json').read_text())
            with open(root/f'{name}_validation.log','x') as logstream:
                subprocess.run([sys.executable,str(Path(__file__).with_name('validate_ges.py')),'--run-id',name],stdout=logstream,stderr=subprocess.STDOUT,check=True,timeout=60)
        results.append(result);print(json.dumps(result),flush=True)
        b.atomic_json(root/'feasibility.json',results)


if __name__=='__main__':main()
