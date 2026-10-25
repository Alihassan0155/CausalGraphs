"""Manual CUDA PC pilot (40-80 genes), isolated from existing experiments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import threading
import time
import traceback

import discover as base
import pc_cuda
from pc_memory import MemoryFisherZ
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.PCUtils import SkeletonDiscovery as sd

np,pd,torch = base.np,base.pd,pc_cuda.torch
HERE = Path(__file__).resolve()
OUTPUT = base.BASE/'pc_cuda_pilot'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id',required=True)
    p.add_argument('--backend',choices=['cpu','cpu-batch','cuda'],default='cuda')
    p.add_argument('--nodes',type=int,default=40)
    p.add_argument('--cells',type=int,default=2000)
    p.add_argument('--seed',type=int,default=20)
    p.add_argument('--max-depth',type=int,default=-1)
    p.add_argument('--batch-size',type=int,default=4096)
    p.add_argument('--audit-ci',action='store_true')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--resume',action='store_true')
    g.add_argument('--restart-stage',action='store_true')
    args = p.parse_args()
    if not (args.run_id and all(c.isalnum() or c in '_-' for c in args.run_id)):
        p.error('Use a simple run ID')
    if not (2<=args.nodes<=80 and 4<=args.cells<=10691 and args.max_depth>=-1 and args.batch_size>0):
        p.error('Pilot limited to <=80 nodes; invalid parameters')
    np.random.seed(args.seed); random.seed(args.seed)
    if args.backend=='cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable: STOP. No CPU fallback.')
    cfg = dict(nodes=args.nodes,cells=args.cells,seed=args.seed,alpha=.05,stable=True,
               uc_rule=0,uc_priority=2,dtype='float64',backend=args.backend,
               batch_size=args.batch_size,audit_ci=args.audit_ci,
               max_depth=None if args.max_depth==-1 else args.max_depth,
               versions={**base.versions(),'torch':torch.__version__,'cuda_runtime':torch.version.cuda},
               python=sys.version,
               inputs={q.name:base.sha(q) for q in (base.MATRIX,base.ORDER,base.CONTROL_ROWS)},
               sources={q.name:base.sha(q) for q in (HERE,HERE.with_name('pc_cuda.py'),HERE.with_name('pc_memory.py'),HERE.with_name('discover.py'))},
               upstream=base.package_fingerprints(),
               ci_orientation={name:base.sha(base.ISOLATED/'causallearn'/path) for name,path in (
                   ('cit','utils/cit.py'),('uc','utils/PCUtils/UCSepset.py'),('meek','utils/PCUtils/Meek.py'))},
               thread_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})
    out = OUTPUT/args.run_id
    prior = {}
    if out.exists():
        prior = json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
        if prior.get('status')=='complete':
            raise SystemExit('Completed experiment protected; choose a NEW run ID.')
        pid = prior.get('pid',-1)
        if base.psutil.pid_exists(pid) and abs(base.psutil.Process(pid).create_time()-prior.get('created',0))<1:
            raise SystemExit('Run is active; refusing concurrent writer.')
        if not (args.resume or args.restart_stage):
            raise SystemExit('Run exists; use --resume or --restart-stage.')
        if json.loads((out/'config.json').read_text())!=cfg:
            raise SystemExit('Configuration/input/software changed; use a NEW run ID.')
        if args.resume and not (out/'pc_depth_checkpoint.pkl').exists():
            raise SystemExit('No complete-depth checkpoint. Use --restart-stage.')
    else:
        if args.resume or args.restart_stage:
            raise SystemExit('No run exists; omit recovery flags.')
        matrix = np.load(base.MATRIX,mmap_mode='r'); order = pd.read_csv(base.ORDER)
        assert matrix.shape==(10691,279) and np.array_equal(order.node_index,np.arange(279))
        assert order.equals(pd.read_csv(base.ROOT/'graphs/correlation_split0/gene_order.csv'))
        source_rows = np.load(base.CONTROL_ROWS)
        cells = pd.read_csv(base.ROOT/'processed/gnn_split0/cells.csv.gz',usecols=['source_row','split','training_control'])
        assert np.array_equal(cells.iloc[source_rows].source_row.to_numpy(),source_rows)
        assert (cells.iloc[source_rows]['split']=='train').all() and cells.iloc[source_rows].training_control.all()
        rng = np.random.default_rng(args.seed)
        cols = np.sort(rng.choice(279,args.nodes,replace=False))
        rows = np.sort(rng.choice(10691,args.cells,replace=False)) if args.cells<10691 else np.arange(10691)
        out.mkdir(parents=True)
        np.save(out/'selected_columns.npy',cols); np.save(out/'selected_rows.npy',rows)
        np.save(out/'selected_source_rows.npy',source_rows[rows])
        names = order.iloc[cols].rename(columns={'node_index':'original_node_index'}).copy()
        names.insert(0,'discovery_node_index',np.arange(len(cols)))
        names.to_csv(out/'node_order.csv',index=False)
        base.atomic_json(out/'config.json',cfg)
        base.atomic_json(out/'provenance.json',dict(command=sys.argv,git_commit=base.subprocess.check_output(
            ['git','rev-parse','HEAD'],cwd=base.ROOT,text=True).strip(),
            limitations='Training controls only; inherited whole-file normalize_total median/log1p transductive preprocessing. Observational Gaussian CI assumptions, not interventional truth.'))
    x = np.asarray(np.load(base.MATRIX,mmap_mode='r')[np.ix_(np.load(out/'selected_rows.npy'),np.load(out/'selected_columns.npy'))],dtype=np.float64)
    assert np.isfinite(x).all() and (x.std(axis=0)>0).all()
    names = pd.read_csv(out/'node_order.csv')
    start = time.perf_counter()
    stop = threading.Event()
    status = dict(status='running',pid=os.getpid(),created=base.psutil.Process().create_time(),
                  attempt=prior.get('attempt',0)+1,mode='resume' if args.resume else 'restart' if args.restart_stage else 'new')
    base.atomic_json(out/'status.json',status)
    def report(kind,**fields):
        base.event(out,kind,elapsed_seconds=time.perf_counter()-start,**fields)
    def heartbeat():
        while not stop.wait(2):
            report('heartbeat',rss_bytes=base.psutil.Process().memory_info().rss)
    thread = threading.Thread(target=heartbeat,daemon=True); thread.start()
    original = sd.skeleton_discovery
    try:
        track,engine,_ = pc_cuda.install_hooks(out/'pc_depth_checkpoint.pkl',report,backend=args.backend,
            batch_size=args.batch_size,audit=args.audit_ci,resume=args.resume)
        result = pc(x,alpha=.05,stable=True,uc_rule=0,uc_priority=2,show_progress=False,
                    indep_test='memory_fisherz',cache_entries=100000 if args.backend=='cpu' else 0,
                    node_names=names.symbol.tolist(),max_k=cfg['max_depth'])
        if args.backend=='cuda':
            torch.cuda.synchronize()
        stats = base.graph_artifacts(out,result.G,names)
        depth = track['last_completed_depth']
        stats.update(algorithm_seconds=time.perf_counter()-start,last_completed_conditioning_depth=depth,
            conditioning_truncated=bool(cfg['max_depth'] is not None and track['max_degree']-1>depth),
            ci=engine[0].stats if engine[0] else {},
            input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
            sepset_sha256=hashlib.sha256(repr(result.sepset.tolist()).encode()).hexdigest(),
            checkpoint_bytes=(out/'pc_depth_checkpoint.pkl').stat().st_size)
        if args.backend=='cuda':
            stats.update(gpu_name=torch.cuda.get_device_name(),vram_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                         vram_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                         gpu_free_total_bytes=list(torch.cuda.mem_get_info()))
        mem = base.psutil.Process().memory_info()
        stats['peak_rss_bytes'] = max(mem.rss,getattr(mem,'peak_wset',0))
        base.atomic_json(out/'statistics.json',stats)
        status['status']='complete'
        report('algorithm_complete',algorithm_seconds=stats['algorithm_seconds'])
    except KeyboardInterrupt:
        status['status']='interrupted'
        report('interrupted',recovery='--resume repeats interrupted depth; completed depths preserved')
        raise SystemExit(130)
    except BaseException:
        status['status']='failed'
        (out/'worker_error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
    finally:
        sd.skeleton_discovery = original
        stop.set(); thread.join(timeout=3)
        status.update(attempt_seconds=time.perf_counter()-start,
                      cumulative_seconds=prior.get('cumulative_seconds',0)+time.perf_counter()-start)
        base.atomic_json(out/'status.json',status)


if __name__=='__main__':
    main()
