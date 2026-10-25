"""Separate memory-bounded PC runner. Never reads/writes legacy PC/BOSS runs."""
import argparse
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time
import traceback
import inspect

import discover as base
from pc_memory import MemoryFisherZ  # registers the custom memoization backend

np, pd, psutil = base.np, base.pd, base.psutil
HERE = Path(__file__).resolve()
OUTPUT = base.BASE / 'pc_memory'


def install_hooks(checkpoint, report, resume=False):
    """Instrument the unchanged upstream loop only at complete depth boundaries."""
    from causallearn.utils.PCUtils import SkeletonDiscovery as sd
    original = sd.skeleton_discovery
    source = inspect.getsource(original)
    start_anchor = '    depth = -1'
    end_anchor = '    if show_progress:\n        pbar.close()'
    assert source.count(start_anchor) == source.count(end_anchor) == 1
    initial = None
    if resume:
        with open(checkpoint, 'rb') as stream:
            initial = pickle.load(stream)
        assert isinstance(initial['cg'].test, MemoryFisherZ)
    tracking = {'last_completed_depth': -1, 'max_degree': None}
    def restore(cg, depth):
        nonlocal initial
        if initial is not None:
            cg, depth = initial['cg'], initial['completed_depth']
            initial = None  # do not keep an additional checkpoint graph alive
            tracking.update(last_completed_depth=depth, max_degree=int(cg.max_degree()))
            report('pc_state_resumed', depth=depth, cache=cg.test.pvalue_cache.statistics())
        return cg, depth
    def save(cg, depth):
        base.atomic_pickle(checkpoint, {'cg':cg, 'completed_depth':depth})
        tracking.update(last_completed_depth=depth, max_degree=int(cg.max_degree()))
        report('pc_depth_complete', depth=depth, max_degree=tracking['max_degree'],
               cache=cg.test.pvalue_cache.statistics(), checkpoint_bytes=checkpoint.stat().st_size,
               skeleton_edges=int(np.count_nonzero(cg.G.graph)//2))
    source = source.replace(start_anchor, start_anchor+'\n    cg, depth = _memory_restore(cg, depth)')
    source = source.replace(end_anchor, '        _memory_save(cg, depth)\n\n'+end_anchor)
    namespace = dict(sd.__dict__, _memory_restore=restore, _memory_save=save)
    exec(compile(source, inspect.getfile(original), 'exec'), namespace)
    sd.skeleton_discovery = namespace['skeleton_discovery']
    return tracking, original


def run_worker(out, resume):
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.PCUtils import SkeletonDiscovery as sd
    cfg = json.loads((out / 'config.json').read_text())
    np.random.seed(cfg['seed']); random.seed(cfg['seed'])
    x = np.asarray(np.load(base.MATRIX, mmap_mode='r')[np.ix_(
        np.load(out / 'selected_rows.npy'), np.load(out / 'selected_columns.npy'))], dtype=np.float64)
    assert np.isfinite(x).all() and (x.std(axis=0) > 0).all()
    nodes = pd.read_csv(out / 'node_order.csv')
    start = time.monotonic()
    def report(kind, **fields):
        fields.setdefault('algorithm_seconds', time.monotonic()-start)
        base.event(out, kind, **fields)
    track, original = install_hooks(out / 'pc_depth_checkpoint.pkl', report, resume)
    try:
        result = pc(x, alpha=cfg['alpha'], stable=True, indep_test='memory_fisherz',
                    cache_entries=cfg['cache_entries'], uc_rule=0, uc_priority=2,
                    max_k=cfg['max_depth'], show_progress=False, node_names=nodes.symbol.tolist())
        depth = track['last_completed_depth']
        remaining = track['max_degree']-1 > depth
        truncated = cfg['max_depth'] is not None and remaining and depth >= cfg['max_depth']
        stats = base.graph_artifacts(out, result.G, nodes)
        stats.update(algorithm_seconds=time.monotonic()-start,
                     last_completed_conditioning_depth=depth, conditioning_truncated=bool(truncated),
                     cache=result.test.pvalue_cache.statistics(),
                     checkpoint_bytes=(out / 'pc_depth_checkpoint.pkl').stat().st_size,
                     resume_semantics='Genuine skeleton/sepset recovery at completed depth; interrupted depth and orientation repeat.')
        mem = psutil.Process().memory_info()
        stats['worker_peak_rss_bytes'] = max(mem.rss, getattr(mem, 'peak_wset', 0))
        base.atomic_json(out / 'statistics.json', stats)
        report('algorithm_complete', **stats)
    except BaseException:
        (out / 'worker_error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    finally:
        sd.skeleton_discovery = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--nodes', type=int, default=279)
    parser.add_argument('--cells', type=int, default=10691)
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--alpha', type=float, default=.05)
    parser.add_argument('--max-depth', type=int, default=-1)
    parser.add_argument('--cache-entries', type=int, default=100000)
    parser.add_argument('--time-limit-seconds', type=float)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--resume', action='store_true')
    group.add_argument('--restart-stage', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (args.run_id and all(c.isalnum() or c in '_-' for c in args.run_id)):
        parser.error('Use a simple run ID.')
    if not (2 <= args.nodes <= 279 and 4 <= args.cells <= 10691 and
            0 < args.alpha < 1 and args.max_depth >= -1 and args.cache_entries >= 0):
        parser.error('Invalid discovery/cache parameters.')
    if args.time_limit_seconds is not None and (not base.math.isfinite(args.time_limit_seconds) or args.time_limit_seconds < 0):
        parser.error('Time budget must be finite and nonnegative; omit for unlimited.')
    out = OUTPUT / args.run_id
    if args.worker:
        run_worker(out, args.resume)
        return
    cfg = dict(nodes=args.nodes, cells=args.cells, seed=args.seed, alpha=args.alpha,
               stable=True, independence_test='fisherz', uc_rule=0, uc_priority=2,
               max_depth=None if args.max_depth == -1 else args.max_depth,
               cache_entries=args.cache_entries, cache_policy='compact uint16 keys, bounded LRU',
               input_dtype='float64 cast from prepared float32', extra_standardization=False,
               versions=base.versions(), implementation_files=base.package_fingerprints(),
               python=sys.version, thread_environment={key:os.environ.get(key) for key in
                   ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
               input_sha256={p.name:base.sha(p) for p in (base.MATRIX, base.ORDER, base.CONTROL_ROWS)},
               runner_sha256={p.name:base.sha(p) for p in (HERE, HERE.with_name('pc_memory.py'), HERE.with_name('discover.py'))},
               cit_sha256=base.sha(base.ISOLATED / 'causallearn/utils/cit.py'))
    cfg['orientation_sha256'] = {name:base.sha(base.ISOLATED / 'causallearn/utils/PCUtils' / name)
                                 for name in ('UCSepset.py','Meek.py')}
    prior = {}
    if out.exists():
        prior = json.loads((out / 'status.json').read_text()) if (out / 'status.json').exists() else {}
        if prior.get('status') == 'complete':
            raise SystemExit('Completed run protected; choose a NEW run ID.')
        pid = prior.get('supervisor_pid', -1)
        if psutil.pid_exists(pid) and abs(psutil.Process(pid).create_time()-prior.get('supervisor_created', 0)) < 1:
            raise SystemExit('Supervisor is still active; refusing concurrent writer.')
        # A supervisor killed outside Ctrl+C may leave a live worker. Never overwrite it.
        for pid, created in prior.get('worker_processes', []):
            if psutil.pid_exists(pid) and abs(psutil.Process(pid).create_time()-created) < 1:
                raise SystemExit('Worker is still active; refusing concurrent writer.')
        if not (args.resume or args.restart_stage):
            raise SystemExit('Run exists; specify --resume or --restart-stage.')
        if json.loads((out / 'config.json').read_text()) != cfg:
            raise SystemExit('Input/configuration/software fingerprint changed. Use a NEW run ID.')
        if args.resume and not (out / 'pc_depth_checkpoint.pkl').exists():
            raise SystemExit('No completed-depth checkpoint. Use --restart-stage.')
    else:
        if args.resume or args.restart_stage:
            raise SystemExit('No run exists. Omit recovery flags.')
        matrix = np.load(base.MATRIX, mmap_mode='r')
        nodes = pd.read_csv(base.ORDER)
        assert matrix.shape == (10691, 279) and np.array_equal(nodes.node_index, np.arange(279))
        assert nodes.equals(pd.read_csv(base.ROOT / 'graphs/correlation_split0/gene_order.csv'))
        rowids = np.load(base.CONTROL_ROWS)
        cells = pd.read_csv(base.ROOT / 'processed/gnn_split0/cells.csv.gz', usecols=['source_row','split','training_control'])
        assert (cells.iloc[rowids]['split'] == 'train').all() and cells.iloc[rowids].training_control.all()
        assert np.array_equal(cells.iloc[rowids].source_row.to_numpy(), rowids)
        rng = np.random.default_rng(args.seed)
        cols = np.sort(rng.choice(279, args.nodes, replace=False)) if args.nodes < 279 else np.arange(279)
        rows = np.sort(rng.choice(10691, args.cells, replace=False)) if args.cells < 10691 else np.arange(10691)
        out.mkdir(parents=True)
        np.save(out / 'selected_columns.npy', cols); np.save(out / 'selected_rows.npy', rows)
        np.save(out / 'selected_source_rows.npy', rowids[rows])
        chosen = nodes.iloc[cols].rename(columns={'node_index':'original_node_index'}).copy()
        chosen.insert(0, 'discovery_node_index', np.arange(len(cols)))
        chosen.to_csv(out / 'node_order.csv', index=False)
        base.atomic_json(out / 'config.json', cfg)
        base.atomic_json(out / 'provenance.json', dict(
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=base.ROOT, text=True).strip(),
            command=sys.argv, python=sys.version, platform=base.platform.platform(),
            normalization_limitation='Inherited whole-file normalize_total median and log1p; transductive preprocessing. Only training controls enter graph discovery; no test-response fitting.',
            assumptions='Linear Gaussian CI, causal Markov/faithfulness, sufficiency, acyclicity; latent confounding, compositionality and cell/batch dependence can violate these.'))
    status = dict(status='running', attempt=prior.get('attempt', 0)+1,
                  supervisor_pid=os.getpid(), supervisor_created=psutil.Process().create_time(),
                  timeout_enabled=bool(args.time_limit_seconds), time_limit_seconds=args.time_limit_seconds)
    base.atomic_json(out / 'status.json', status)
    start = time.monotonic(); child = None; peak = 0; state = 'failed'
    base.event(out, 'attempt_started', **status, mode='resume' if args.resume else 'restart' if args.restart_stage else 'new')
    try:
        with open(out / 'worker.log', 'a', encoding='utf-8') as stream:
            child = subprocess.Popen([sys.executable, '-u', str(HERE), *sys.argv[1:], '--worker'],
                                     cwd=base.ROOT, stdout=stream, stderr=subprocess.STDOUT)
            while child.poll() is None:
                try:
                    procs = [psutil.Process(child.pid)]
                    procs += procs[0].children(recursive=True)
                    status['worker_processes'] = [[p.pid, p.create_time()] for p in procs]
                    mem = [p.memory_info() for p in procs]
                    peak = max(peak, sum(max(m.rss, getattr(m, 'peak_wset', 0)) for m in mem))
                    base.atomic_json(out / 'status.json', status)
                except psutil.NoSuchProcess:
                    pass
                elapsed = time.monotonic()-start
                base.event(out, 'heartbeat', elapsed_seconds=elapsed, peak_worker_rss_bytes=peak)
                if base.time_budget_exhausted(args.time_limit_seconds, elapsed):
                    state = 'timed_out'; base.stop_worker_tree(child); break
                time.sleep(2)
            child.wait()
            if state != 'timed_out':
                state = 'complete' if child.returncode == 0 else 'failed'
    except KeyboardInterrupt:
        state = 'interrupted'
        if child and child.poll() is None:
            base.stop_worker_tree(child)
    finally:
        if state == 'complete':
            stats = json.loads((out / 'statistics.json').read_text())
            peak = max(peak, stats['worker_peak_rss_bytes'])
            stats.update(supervised_seconds=time.monotonic()-start,
                         cumulative_supervised_seconds=prior.get('cumulative_supervised_seconds', 0)+time.monotonic()-start,
                         peak_worker_rss_bytes=peak)
            base.atomic_json(out / 'statistics.json', stats)
        status.update(status=state, attempt_seconds=time.monotonic()-start,
                      cumulative_supervised_seconds=prior.get('cumulative_supervised_seconds', 0)+time.monotonic()-start,
                      peak_worker_rss_bytes=peak, exit_code=child.returncode if child else None)
        base.atomic_json(out / 'status.json', status)
        base.event(out, 'attempt_finished', **status)
    if state != 'complete':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
