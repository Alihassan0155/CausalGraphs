"""Small isolated subprocess benchmarks, telemetry, exact trajectories and BLAS controls."""
import argparse
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import discover as b


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prefix',required=True);args=parser.parse_args()
    if not args.prefix or not all(c.isalnum() or c in '_-' for c in args.prefix):parser.error('Simple prefix required')
    report=b.BASE/'ges_cpu_fast'/f'{args.prefix}_report'
    if report.exists():raise SystemExit('Use NEW prefix')
    report.mkdir(parents=True);results=[]
    cases=[(n,backend,1,False) for n in (20,40,80) for backend in ('reference','fast')]
    cases += [(80,'fast',threads,False) for threads in (2,4)]
    cases += [(80,backend,1,True) for backend in ('reference','fast')]
    for nodes,backend,threads,profile in cases:
        name=f'{args.prefix}_n{nodes}_{backend}_t{threads}{"_profile" if profile else ""}'
        script=Path(__file__).with_name('discover_ges.py' if backend=='reference' else 'discover_ges_fast.py')
        command=[sys.executable,'-u']
        if profile:command+=['-m','cProfile','-o',str(report/f'{name}.pstats')]
        command += [str(script),'--run-id',name,'--nodes',str(nodes),'--cells','2000','--seed','20','--bic-penalty','2']
        env=os.environ.copy()
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):env[key]=str(threads)
        start=time.monotonic();peak=0;cpu=0;cpus={}
        with open(report/f'{name}.log','x') as stream:
            child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env)
            try:
                while child.poll() is None:
                    try:
                        root=b.psutil.Process(child.pid);processes=[root]+root.children(recursive=True)
                        memory=[]
                        for process in processes:
                            m=process.memory_info();memory.append(max(m.rss,getattr(m,'peak_wset',0)))
                            times=process.cpu_times();cpus[(process.pid,process.create_time())]=times.user+times.system
                        peak=max(peak,sum(memory));cpu=sum(cpus.values())
                    except b.psutil.NoSuchProcess:pass
                    if time.monotonic()-start>60:raise TimeoutError('Lightweight budget exceeded')
                    time.sleep(.05)
                if child.returncode:raise RuntimeError(f'Failed {name}; inspect log')
            finally:
                if child.poll() is None:b.stop_worker_tree(child)
        folder=b.BASE/('ges' if backend=='reference' else 'ges_cpu_fast')/name
        stats=json.loads((folder/'statistics.json').read_text());elapsed=time.monotonic()-start
        result=dict(nodes=nodes,backend=backend,threads=threads,profile=profile,run_id=name,
            process_seconds=elapsed,algorithm_seconds=stats['algorithm_seconds'],peak_rss_bytes=max(peak,stats['peak_rss_bytes']),
            sampled_cpu_seconds=cpu,average_core_equivalents=cpu/elapsed,
            total_machine_cpu_percent=100*cpu/elapsed/b.psutil.cpu_count(),statistics=stats)
        results.append(result);b.atomic_json(report/'partial_results.json',results)
        print(name,round(elapsed,3),round(stats['algorithm_seconds'],3),flush=True)
    reference={}
    for result in results:
        folder=b.BASE/('ges' if result['backend']=='reference' else 'ges_cpu_fast')/result['run_id']
        with open(folder/'ges_record.pkl','rb') as stream:record=pickle.load(stream)
        if result['backend']=='reference' and not result['profile']:reference[result['nodes']]=record
        native=reference[result['nodes']]
        checks=dict(endpoints_equal=bool(b.np.array_equal(record['G'].graph,native['G'].graph)),
            score_equal=bool(record['score']==native['score']),
            forward_moves_equal=repr(record['update1'])==repr(native['update1']),
            backward_moves_equal=repr(record['update2'])==repr(native['update2']))
        for key in ('G_step1','G_step2'):
            checks[key+'_equal']=len(record[key])==len(native[key]) and all(b.np.array_equal(a.graph,c.graph) for a,c in zip(record[key],native[key]))
        result['checks']=checks;result['accepted']=all(checks.values())
    import pstats
    for path in report.glob('*.pstats'):
        with open(path.with_suffix('.txt'),'w') as stream:pstats.Stats(str(path),stream=stream).strip_dirs().sort_stats('cumulative').print_stats(45)
    b.atomic_json(report/'summary.json',dict(results=results,note='Single measurements under existing background workloads; sampled CPU telemetry can miss final interval. No full experiment launched.'))
    print('PASS flags',[(r['run_id'],r['accepted']) for r in results])


if __name__=='__main__':main()
