"""Separate CPU/CUDA GES feasibility runner with genuine loop-state recovery."""
import argparse
import cProfile
import hashlib
import json
import os
from pathlib import Path
import pickle
import pstats
import threading
import time
import traceback
import discover as b
from discover_pc_batched import validate_inputs
from ges_recovery import build
from ges_cuda_backend import CudaScore,torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id',required=True);p.add_argument('--backend',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--nodes',type=int,default=40);p.add_argument('--cells',type=int,default=2000)
    p.add_argument('--seed',type=int,default=20);p.add_argument('--bic-penalty',type=float,default=2.)
    p.add_argument('--profile',action='store_true')
    group=p.add_mutually_exclusive_group();group.add_argument('--resume',action='store_true');group.add_argument('--restart-stage',action='store_true')
    args=p.parse_args()
    if not args.run_id or not all(c.isalnum() or c in '_-' for c in args.run_id) or not 2<=args.nodes<=80 or not 4<=args.cells<=10691:p.error('Pilot restricted to 2–80 genes')
    if not b.np.isfinite(args.bic_penalty) or args.bic_penalty<=0:p.error('Invalid penalty')
    if args.backend=='cuda' and not torch.cuda.is_available():raise SystemExit('CUDA unavailable; stopping')
    folder=b.BASE/'ges_cuda_pilot'/args.run_id
    paths=[Path(__file__),Path(__file__).with_name('ges_cuda_backend.py'),Path(__file__).with_name('ges_recovery.py'),Path(b.__file__),Path(__file__).with_name('discover_pc_batched.py')]
    paths += [b.ISOLATED/'causallearn'/s for s in ['search/ScoreBased/GES.py','utils/GESUtils.py','score/LocalScoreFunction.py','score/LocalScoreFunctionClass.py','utils/DAG2CPDAG.py','utils/PDAG2DAG.py']]
    cfg=dict(nodes=args.nodes,cells=args.cells,seed=args.seed,backend=args.backend,lambda_value=args.bic_penalty,profile=args.profile,
        sources={str(path):b.sha(path) for path in paths},inputs={q.name:b.sha(q) for q in (b.MATRIX,b.ORDER,b.CONTROL_ROWS)},
        versions={**b.versions(),'torch':torch.__version__,'cuda':torch.version.cuda},python=b.sys.version,
        threads={key:os.environ.get(key) for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']})
    prior={}
    if folder.exists():
        prior=json.loads((folder/'status.json').read_text())
        if prior['status']=='complete':raise SystemExit('Complete run protected')
        pid=prior.get('pid',-1)
        if b.psutil.pid_exists(pid) and abs(b.psutil.Process(pid).create_time()-prior.get('created',0))<1:raise SystemExit('Run active')
        if not(args.resume or args.restart_stage):raise SystemExit('Use --resume or --restart-stage')
        if json.loads((folder/'config.json').read_text())!=cfg:raise SystemExit('Configuration changed')
    else:
        if args.resume or args.restart_stage:raise SystemExit('Run missing')
        folder.mkdir(parents=True);rng=b.np.random.default_rng(args.seed)
        cols=b.np.sort(rng.choice(279,args.nodes,replace=False));rows=b.np.sort(rng.choice(10691,args.cells,replace=False))
        b.np.save(folder/'selected_columns.npy',cols);b.np.save(folder/'selected_rows.npy',rows)
        b.np.save(folder/'selected_source_rows.npy',b.np.load(b.CONTROL_ROWS)[rows])
        names=b.pd.read_csv(b.ORDER).iloc[cols].rename(columns={'node_index':'original_node_index'}).copy()
        names.insert(0,'discovery_node_index',range(len(cols)));names.to_csv(folder/'node_order.csv',index=False)
        b.atomic_json(folder/'config.json',cfg)
    x,names=validate_inputs(folder);b.np.random.seed(args.seed)
    start=time.monotonic();status=dict(status='running',pid=os.getpid(),created=b.psutil.Process().create_time(),attempt=prior.get('attempt',0)+1)
    b.atomic_json(folder/'status.json',status);stop=threading.Event();latest={}
    def log(kind,**values):b.event(folder,kind,elapsed=time.monotonic()-start,**values)
    def heartbeat():
        while not stop.wait(5):log('heartbeat',rss=b.psutil.Process().memory_info().rss,**latest)
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    scoring=[None];profile=cProfile.Profile()
    def saved(phase,state):
        scoring[0]=state['score_func'];latest.update(phase=phase,forward_moves=len(state['update1']),backward_moves=len(state.get('update2',[])))
        log('checkpoint_saved',**latest)
    try:
        func=build(lambda *a:None,folder/'checkpoint.pkl',hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),args.resume,saved)
        if args.backend=='cuda':
            torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.reset_peak_memory_stats()
            func.__globals__['LocalScoreClass']=CudaScore
        if args.profile:profile.enable()
        record=func(x,parameters={'lambda_value':args.bic_penalty},score_func='local_score_BIC_from_cov',node_names=names.symbol.tolist(),maxP=args.nodes)
        if args.backend=='cuda':torch.cuda.synchronize()
        duration=time.monotonic()-start
        if args.profile:
            profile.disable();profile.dump_stats(folder/'profile.pstats')
            with open(folder/'profile.txt','w') as stream:pstats.Stats(profile,stream=stream).strip_dirs().sort_stats('cumulative').print_stats(35)
        stats=b.graph_artifacts(folder,record['G'],names)
        stats.update(algorithm_seconds=duration,score=float(record['score']),forward_moves=len(record['update1']),backward_moves=len(record['update2']))
        if args.backend=='cuda':stats.update(score_audit=scoring[0].audit_cache(),vram_allocated_peak=torch.cuda.max_memory_allocated(),vram_reserved_peak=torch.cuda.max_memory_reserved(),device=torch.cuda.get_device_name())
        mem=b.psutil.Process().memory_info();stats['peak_rss']=max(mem.rss,getattr(mem,'peak_wset',0))
        b.atomic_pickle(folder/'record.pkl',record);b.atomic_json(folder/'statistics.json',stats);status['status']='complete'
    except KeyboardInterrupt:status['status']='interrupted';raise SystemExit(130)
    except BaseException:
        status['status']='failed';(folder/'error.txt').write_text(traceback.format_exc());raise
    finally:
        stop.set();thread.join(timeout=3);status['elapsed']=time.monotonic()-start;b.atomic_json(folder/'status.json',status)


if __name__=='__main__':main()
