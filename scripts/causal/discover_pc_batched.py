"""Full-size CPU-only PC-stable runner. Unlimited runtime; manual launch and resume."""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import threading
import time
import traceback

import discover as base
import pc_batched as batch
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.PCUtils import SkeletonDiscovery as sd

np,pd=base.np,base.pd


def ensure_idle(status):
    pid=status.get('pid',-1)
    if base.psutil.pid_exists(pid) and abs(base.psutil.Process(pid).create_time()-status.get('created',0))<1:
        raise SystemExit('Run is active; refusing concurrent writer')


def validate_inputs(out):
    order=pd.read_csv(base.ORDER)
    assert order.equals(pd.read_csv(base.ROOT/'graphs/correlation_split0/gene_order.csv'))
    assert np.array_equal(order.node_index,np.arange(279))
    source=np.load(base.CONTROL_ROWS)
    cells=pd.read_csv(base.ROOT/'processed/gnn_split0/cells.csv.gz',usecols=['source_row','split','training_control'])
    assert np.array_equal(cells.iloc[source].source_row.to_numpy(),source)
    assert (cells.iloc[source]['split']=='train').all() and cells.iloc[source].training_control.all()
    cols=np.load(out/'selected_columns.npy');rows=np.load(out/'selected_rows.npy')
    assert np.array_equal(cols,np.sort(np.unique(cols))) and np.array_equal(rows,np.sort(np.unique(rows)))
    assert ((0<=cols)&(cols<279)).all() and ((0<=rows)&(rows<10691)).all()
    assert np.array_equal(np.load(out/'selected_source_rows.npy'),source[rows])
    names=pd.read_csv(out/'node_order.csv')
    assert names.symbol.tolist()==order.iloc[cols].symbol.tolist()
    assert names.ensembl_id.tolist()==order.iloc[cols].ensembl_id.tolist()
    assert np.array_equal(names.expression_column.to_numpy(),order.iloc[cols].expression_column.to_numpy())
    matrix=np.load(base.MATRIX,mmap_mode='r');assert matrix.shape==(10691,279)
    x=np.asarray(matrix[np.ix_(rows,cols)],dtype=np.float64)
    assert np.isfinite(x).all() and (x.std(axis=0)>0).all()
    return x,names


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--nodes',type=int,default=279)
    parser.add_argument('--cells',type=int,default=10691)
    parser.add_argument('--seed',type=int,default=20)
    parser.add_argument('--batch-size',type=int,default=4096)
    parser.add_argument('--max-depth',type=int,default=-1)
    parser.add_argument('--audit-ci',action='store_true')
    recovery=parser.add_mutually_exclusive_group()
    recovery.add_argument('--resume',action='store_true')
    recovery.add_argument('--restart-stage',action='store_true')
    args=parser.parse_args()
    if not(args.run_id and all(c.isalnum() or c in '_-' for c in args.run_id)):
        parser.error('Use a simple run ID')
    if not(2<=args.nodes<=279 and 4<=args.cells<=10691 and args.batch_size>0 and args.max_depth>=-1):
        parser.error('Invalid parameters')
    np.random.seed(args.seed);random.seed(args.seed)
    cfg=batch.configuration(args.nodes,args.cells,args.seed,args.batch_size,
                            None if args.max_depth==-1 else args.max_depth,args.audit_ci)
    out=batch.BASE/args.run_id
    prior={}
    if out.exists():
        prior=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
        if prior.get('status')=='complete':
            raise SystemExit('Completed experiment protected; choose a NEW run ID')
        ensure_idle(prior)
        if not(args.resume or args.restart_stage):
            raise SystemExit('Run exists; use --resume or --restart-stage')
        if json.loads((out/'config.json').read_text())!=cfg:
            raise SystemExit('Input/settings/source/software fingerprints changed; use a NEW run ID')
        if args.resume and not(out/'pc_depth_checkpoint.pkl').exists():
            raise SystemExit('No complete-depth checkpoint; explicit --restart-stage starts from zero')
    else:
        if args.resume or args.restart_stage:
            raise SystemExit('No run exists; omit recovery flags')
        out.mkdir(parents=True)
        rng=np.random.default_rng(args.seed)
        cols=np.sort(rng.choice(279,args.nodes,replace=False)) if args.nodes<279 else np.arange(279)
        rows=np.sort(rng.choice(10691,args.cells,replace=False)) if args.cells<10691 else np.arange(10691)
        np.save(out/'selected_columns.npy',cols);np.save(out/'selected_rows.npy',rows)
        np.save(out/'selected_source_rows.npy',np.load(base.CONTROL_ROWS)[rows])
        chosen=pd.read_csv(base.ORDER).iloc[cols].rename(columns={'node_index':'original_node_index'}).copy()
        chosen.insert(0,'discovery_node_index',np.arange(len(cols)))
        chosen.to_csv(out/'node_order.csv',index=False)
        base.atomic_json(out/'config.json',cfg)
        base.atomic_json(out/'provenance.json',dict(command=sys.argv,git_commit=base.subprocess.check_output(
            ['git','rev-parse','HEAD'],cwd=base.ROOT,text=True).strip(),
            limitations='Training controls only. Inherited whole-file normalize_total/log1p is transductive. Observational linear-Gaussian PC assumptions apply.'))
    x,names=validate_inputs(out)
    assert x.shape==(args.cells,args.nodes)
    start=time.monotonic();stop=threading.Event();engine=[None]
    status=dict(status='running',pid=os.getpid(),created=base.psutil.Process().create_time(),
                attempt=prior.get('attempt',0)+1,mode='resume' if args.resume else 'restart' if args.restart_stage else 'new',
                timeout_enabled=False)
    base.atomic_json(out/'status.json',status)
    def report(kind,**fields):
        base.event(out,kind,elapsed_seconds=time.monotonic()-start,**fields)
    def heartbeat():
        while not stop.wait(5):
            current=engine[0]
            report('heartbeat',rss_bytes=base.psutil.Process().memory_info().rss,
                   ci=current.stats.copy() if current else {},position=current.position.copy() if current else {})
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    original=sd.skeleton_discovery
    try:
        track,engine,original=batch.install_hooks(out/'pc_depth_checkpoint.pkl',x,cfg,report,args.resume)
        result=pc(x,alpha=.05,stable=True,uc_rule=0,uc_priority=2,node_names=names.symbol.tolist(),
                  indep_test='lean_fisherz',show_progress=False,max_k=cfg['max_depth'])
        stats=base.graph_artifacts(out,result.G,names)
        stats.update(algorithm_seconds=time.monotonic()-start,last_completed_conditioning_depth=track['last_completed_depth'],
            resumed_from_depth=track['initial_depth'],
            conditioning_truncated=bool(cfg['max_depth'] is not None and track['max_degree']-1>track['last_completed_depth']),
            ci=engine[0].stats if engine[0] else {},
            sepset_sha256=batch.summary(result)['sepset_sha256'],
            checkpoint_bytes=(out/'pc_depth_checkpoint.pkl').stat().st_size)
        memory=base.psutil.Process().memory_info()
        stats['peak_rss_bytes']=max(memory.rss,getattr(memory,'peak_wset',0))
        base.atomic_json(out/'statistics.json',stats)
        status['status']='complete';report('algorithm_complete',seconds=stats['algorithm_seconds'])
    except KeyboardInterrupt:
        status['status']='interrupted';report('interrupted',recovery='--resume restores last complete depth; repeats interrupted depth')
        raise SystemExit(130)
    except BaseException:
        status['status']='failed'
        (out/'worker_error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
    finally:
        sd.skeleton_discovery=original;stop.set();thread.join(timeout=3)
        status.update(attempt_seconds=time.monotonic()-start,
                      cumulative_seconds=prior.get('cumulative_seconds',0)+time.monotonic()-start)
        base.atomic_json(out/'status.json',status)


if __name__=='__main__':
    main()
