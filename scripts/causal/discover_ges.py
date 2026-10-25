"""Genuine causal-learn GES; manual full run, progress snapshots, explicit restart only."""
import argparse
import inspect
import json
import os
from pathlib import Path
import random
import sys
import threading
import time
import traceback
import hashlib
from ges_recovery import build
import discover as b
from discover_pc_batched import validate_inputs
np,pd=b.np,b.pd
ROOT=b.BASE/'ges'


def instrument(report):
    import importlib
    module=importlib.import_module('causallearn.search.ScoreBased.GES')
    source=inspect.getsource(module.ges)
    for phase in (1,2):
        anchor=f'            G_step{phase}.append(G)'
        assert source.count(anchor)==1
        source=source.replace(anchor,anchor+f'\n            _report({phase}, G, score_new, len(update{phase}))')
    namespace=dict(module.__dict__,_report=report)
    exec(compile(source,inspect.getfile(module.ges),'exec'),namespace)
    return namespace['ges']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id',required=True)
    p.add_argument('--nodes',type=int,default=279)
    p.add_argument('--cells',type=int,default=10691)
    p.add_argument('--seed',type=int,default=20)
    p.add_argument('--bic-penalty',type=float,default=2.,help='Match BOSS penalty; conventional BIC is 0.5 in this implementation')
    recovery=p.add_mutually_exclusive_group()
    recovery.add_argument('--restart-stage',action='store_true')
    recovery.add_argument('--resume',action='store_true',help='Restore genuine loop-boundary checkpoint; unfinished candidate scan repeats')
    args=p.parse_args()
    if not(args.run_id and all(c.isalnum() or c in '_-' for c in args.run_id)) or not(2<=args.nodes<=279 and 4<=args.cells<=10691 and np.isfinite(args.bic_penalty) and args.bic_penalty>0):p.error('Invalid parameters')
    out=ROOT/args.run_id
    pkg=b.ISOLATED/'causallearn'
    cfg=dict(algorithm='GES',nodes=args.nodes,cells=args.cells,seed=args.seed,bic_penalty=args.bic_penalty,
        score='local_score_BIC_from_cov',maxP=args.nodes,dtype='float64',covariance_ddof=0,
        inputs={path.name:b.sha(path) for path in (b.MATRIX,b.ORDER,b.CONTROL_ROWS)},versions=b.versions(),python=sys.version,
        sources={str(path):b.sha(path) for path in (Path(__file__),Path(__file__).with_name('ges_recovery.py'),Path(b.__file__),Path(__file__).with_name('discover_pc_batched.py'),
            pkg/'search/ScoreBased/GES.py',pkg/'utils/GESUtils.py',pkg/'score/LocalScoreFunction.py',
            pkg/'score/LocalScoreFunctionClass.py',pkg/'utils/DAG2CPDAG.py',pkg/'utils/PDAG2DAG.py')},
        threads={key:os.environ.get(key) for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})
    prior={}
    if out.exists():
        prior=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
        if prior.get('status')=='complete':raise SystemExit('Completed run protected; use a NEW run ID')
        pid=prior.get('pid',-1)
        if b.psutil.pid_exists(pid) and abs(b.psutil.Process(pid).create_time()-prior.get('created',0))<1:raise SystemExit('Run active; concurrent writer refused')
        if not (args.restart_stage or args.resume):raise SystemExit('Run exists. Use --resume or --restart-stage.')
        if json.loads((out/'config.json').read_text())!=cfg:raise SystemExit('Changed inputs/configuration/software; use a NEW run ID')
        if args.resume and not (out/'ges_checkpoint.pkl').exists():raise SystemExit('No genuine checkpoint; legacy snapshots cannot resume. Use a NEW run ID.')
    else:
        if args.restart_stage or args.resume:raise SystemExit('Run missing; omit recovery flags')
        out.mkdir(parents=True)
        rng=np.random.default_rng(args.seed)
        cols=np.sort(rng.choice(279,args.nodes,replace=False)) if args.nodes<279 else np.arange(279)
        rows=np.sort(rng.choice(10691,args.cells,replace=False)) if args.cells<10691 else np.arange(10691)
        np.save(out/'selected_columns.npy',cols);np.save(out/'selected_rows.npy',rows)
        np.save(out/'selected_source_rows.npy',np.load(b.CONTROL_ROWS)[rows])
        names=pd.read_csv(b.ORDER).iloc[cols].rename(columns={'node_index':'original_node_index'}).copy()
        names.insert(0,'discovery_node_index',np.arange(len(cols)));names.to_csv(out/'node_order.csv',index=False)
        b.atomic_json(out/'config.json',cfg)
        b.atomic_json(out/'provenance.json',dict(command=sys.argv,git_commit=b.subprocess.check_output(['git','rev-parse','HEAD'],cwd=b.ROOT,text=True).strip(),
            limitations='Only training controls. Inherited whole-file normalize_total/log1p is transductive, not strictly inductive. Observational Gaussian SEM assumptions, latent confounding and cell/batch dependence limit causal interpretation.',
            recovery='Genuine loop-boundary checkpoints retain graph, scores, phase, counters, histories, score caches and random states. An unfinished candidate scan repeats; completed moves are retained.'))
    x,names=validate_inputs(out);assert x.shape==(args.cells,args.nodes)
    np.random.seed(args.seed);random.seed(args.seed)
    status=dict(status='running',pid=os.getpid(),created=b.psutil.Process().create_time(),attempt=prior.get('attempt',0)+1,timeout_enabled=False)
    b.atomic_json(out/'status.json',status)
    start=time.monotonic();stop=threading.Event();latest={}
    def event(kind,**fields):b.event(out,kind,elapsed_seconds=time.monotonic()-start,attempt=status['attempt'],**fields)
    def report(phase,graph,score,moves):
        latest.update(phase='forward' if phase==1 else 'backward',moves=moves,score=float(np.asarray(score).item()))
        # Stage snapshot retained per attempt; not claimed as resumable checkpoint.
        b.atomic_pickle(out/f'latest_snapshot_attempt{status["attempt"]}.pkl',dict(graph=graph,**latest,resumable=False))
        event('accepted_move',**latest,edges=int(np.count_nonzero(graph.graph)//2))
    def heartbeat():
        while not stop.wait(5):event('heartbeat',rss_bytes=b.psutil.Process().memory_info().rss,**latest)
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    try:
        event('search_started',restart=args.restart_stage)
        fingerprint=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
        def saved(phase,state):
            latest.update(phase='forward' if phase==1 else 'backward',moves=len(state['update1' if phase==1 else 'update2']))
            event('checkpoint_saved',**latest,bytes=(out/'ges_checkpoint.pkl').stat().st_size)
        record=build(report,out/'ges_checkpoint.pkl',fingerprint,args.resume,saved)(x,score_func='local_score_BIC_from_cov',parameters={'lambda_value':args.bic_penalty},maxP=args.nodes,node_names=names.symbol.tolist())
        stats=b.graph_artifacts(out,record['G'],names)
        stats.update(algorithm_seconds=time.monotonic()-start,score=float(np.asarray(record['score']).item()),
            forward_moves=len(record['update1']),backward_moves=len(record['update2']))
        mem=b.psutil.Process().memory_info();stats['peak_rss_bytes']=max(mem.rss,getattr(mem,'peak_wset',0))
        b.atomic_pickle(out/'ges_record.pkl',record)
        b.atomic_json(out/'statistics.json',stats)
        status['status']='complete';event('search_complete',**stats)
    except KeyboardInterrupt:
        status['status']='interrupted';event('interrupted',recovery='--resume; unfinished candidate scan repeats');raise SystemExit(130)
    except BaseException:
        status['status']='failed';(out/'worker_error.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
    finally:
        stop.set();thread.join(timeout=3)
        status.update(attempt_seconds=time.monotonic()-start,cumulative_seconds=prior.get('cumulative_seconds',0)+time.monotonic()-start)
        b.atomic_json(out/'status.json',status)


if __name__=='__main__':main()
