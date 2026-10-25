"""Lightweight, isolated-process cache/PC equivalence and recovery benchmarks."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import time

import discover as base
import discover_pc_memory as optimized
from pc_memory import MemoryFisherZ, BoundedCache
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import FisherZ
from causallearn.utils.PCUtils import SkeletonDiscovery as sd

np = base.np


def subset(nodes, cells):
    rng = np.random.default_rng(20)
    cols = np.sort(rng.choice(279, nodes, replace=False))
    rows = np.sort(rng.choice(10691, cells, replace=False))
    return np.asarray(np.load(base.MATRIX, mmap_mode='r')[np.ix_(rows, cols)], dtype=np.float64)


def signature(value):
    # Sepset entries are lists of tuples or None; compare exact structure and order.
    return hashlib.sha256(repr(value.tolist()).encode()).hexdigest()


def child(out, nodes, mode):
    x = subset(nodes, 2000)
    snapshots = []
    original = sd.skeleton_discovery
    start = time.perf_counter()
    def report(kind, **fields):
        snapshots.append(dict(event=kind, **fields))
    if mode == 'original':
        track, _ = base.install_pc_hooks(out/'checkpoint.pkl', report)
        kwargs = dict(indep_test='fisherz')
    else:
        track, _ = optimized.install_hooks(out/'checkpoint.pkl', report)
        kwargs = dict(indep_test='memory_fisherz', cache_entries=1000)
    try:
        graph = pc(x, alpha=.05, stable=True, uc_rule=0, uc_priority=2, show_progress=False, **kwargs)
    finally:
        sd.skeleton_discovery = original
    elapsed = time.perf_counter()-start
    cache = graph.test.pvalue_cache
    if isinstance(cache, BoundedCache):
        cache_stats = cache.statistics()
    else:
        cache_stats = dict(entries=len(cache)-3, live_cache_object_bytes=sys.getsizeof(cache)+sum(
            sys.getsizeof(k)+sys.getsizeof(v) for k,v in cache.items()))
    mem = base.psutil.Process().memory_info()
    np.save(out/'endpoint_matrix.npy', graph.G.graph)
    base.atomic_json(out/'measurement.json', dict(
        nodes=nodes, cells=2000, seed=20, mode=mode, algorithm_seconds=elapsed,
        peak_rss_bytes=max(mem.rss, getattr(mem,'peak_wset',0)), cache=cache_stats,
        checkpoint_bytes=(out/'checkpoint.pkl').stat().st_size,
        sepset_sha256=signature(graph.sepset), input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
        last_depth=track['last_completed_depth'], depth_profiles=snapshots,
        edges=int(np.count_nonzero(graph.G.graph)//2)))


def unit_checks(out):
    assert MemoryFisherZ.__call__ is FisherZ.__call__
    x = subset(12, 500)
    native, compact = FisherZ(x), MemoryFisherZ(x, cache_entries=2)
    comparisons = 0
    for depth in range(4):
        for i in range(9):
            s = list(range(9,9+depth))
            for a,b,cond in ((i,(i+1)%9,s), ((i+1)%9,i,list(reversed(s))+s)):
                assert native(a,b,cond) == compact(a,b,cond)
                comparisons += 1
    assert compact.pvalue_cache.high_water <= 2 and compact.pvalue_cache.evictions > 0
    # Canonical keys preserve symmetry and condition-set deduplication; no collisions.
    keys = set()
    import itertools
    for i,j in itertools.combinations(range(12),2):
        others = [v for v in range(12) if v not in (i,j)]
        for d in range(3):
            for s in itertools.combinations(others,d):
                k = compact.get_formatted_XYZ_and_cachekey(i,j,s)[3]
                assert k not in keys
                keys.add(k)
    original = sd.skeleton_discovery
    native_graph = pc(x, alpha=.05, stable=True, uc_rule=0, uc_priority=2, show_progress=False)
    for bound in (0,2,1000):
        graph = pc(x, alpha=.05, stable=True, uc_rule=0, uc_priority=2, show_progress=False,
                   indep_test='memory_fisherz', cache_entries=bound)
        assert np.array_equal(native_graph.G.graph, graph.G.graph)
        assert signature(native_graph.sepset) == signature(graph.sepset)
    class Interrupted(Exception):
        pass
    path = out/'recovery_checkpoint.pkl'
    def interrupt(kind, **fields):
        if fields.get('depth') == 0:
            raise Interrupted()
    try:
        optimized.install_hooks(path, interrupt)
        try:
            pc(x, alpha=.05, stable=True, uc_rule=0, uc_priority=2, show_progress=False,
               indep_test='memory_fisherz', cache_entries=2)
            raise AssertionError('Expected depth-boundary interruption')
        except Interrupted:
            pass
    finally:
        sd.skeleton_discovery = original
    # A partially written temporary checkpoint must not replace the valid one.
    path.with_suffix('.tmp').write_bytes(b'interrupted temporary pickle')
    events = []
    try:
        optimized.install_hooks(path, lambda k,**v: events.append(dict(event=k,**v)), resume=True)
        graph = pc(x, alpha=.05, stable=True, uc_rule=0, uc_priority=2, show_progress=False,
                   indep_test='memory_fisherz', cache_entries=2)
    finally:
        sd.skeleton_discovery = original
    assert np.array_equal(native_graph.G.graph, graph.G.graph)
    assert signature(native_graph.sepset) == signature(graph.sepset)
    assert events[0]['event']=='pc_state_resumed' and events[0]['depth']==0
    return dict(exact_pvalue_comparisons=comparisons, unique_compact_keys_checked=len(keys),
                bounded_eviction=True, zero_cache_equivalent=True, resumed_endpoints_equal=True,
                resumed_sepsets_equal=True, partial_temporary_checkpoint_ignored=True)


def storage_profile():
    # Controlled 100k depth-2-like unique keys; no CI searches or dense data copies.
    import itertools
    import struct
    values = {}
    compact_unbounded = {}
    compact = BoundedCache({}, 1000)
    start = time.perf_counter()
    for i,j,a,b in itertools.islice(itertools.combinations(range(279),4),100000):
        values[f'{i};{j}|{a}.{b}'] = float(.123456789)
        key = struct.pack('<HHHH',i,j,a,b)
        compact[key] = float(.123456789)
        compact_unbounded[key] = float(.123456789)
    return dict(generated_entries=100000, seconds=time.perf_counter()-start,
                original_live_cache_object_bytes=sys.getsizeof(values)+sum(sys.getsizeof(k)+sys.getsizeof(v) for k,v in values.items()),
                compact_unbounded_live_cache_object_bytes=sys.getsizeof(compact_unbounded)+sum(
                    sys.getsizeof(k)+sys.getsizeof(v) for k,v in compact_unbounded.items()),
                optimized=compact.statistics(), note='Shallow live object accounting, not process RSS; no full PC launched.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--child', choices=['original','optimized'])
    p.add_argument('--nodes', type=int, choices=[20,40], default=20)
    args = p.parse_args()
    if args.child:
        child(args.output, args.nodes, args.child)
        return
    if args.output.exists():
        raise SystemExit('Output exists; choose a NEW benchmark directory.')
    args.output.mkdir(parents=True)
    checks = unit_checks(args.output)
    measurements = []
    for nodes in (20,40):
        pair = []
        for mode in ('original','optimized'):
            folder = args.output/f'n{nodes}_{mode}'
            folder.mkdir()
            with open(folder/'worker.log','w') as stream:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), '--child', mode,
                                '--nodes', str(nodes), '--output', str(folder)],
                               stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=60)
            pair.append(json.loads((folder/'measurement.json').read_text()))
        assert pair[0]['input_sha256'] == pair[1]['input_sha256']
        assert pair[0]['sepset_sha256'] == pair[1]['sepset_sha256']
        assert np.array_equal(np.load(args.output/f'n{nodes}_original/endpoint_matrix.npy'),
                              np.load(args.output/f'n{nodes}_optimized/endpoint_matrix.npy'))
        measurements.extend(pair)
        print(json.dumps(pair), flush=True)
    base.atomic_json(args.output/'summary.json', dict(checks=checks, measurements=measurements,
                     storage_profile=storage_profile(), versions=base.versions(),
                     matrix_sha256=base.sha(base.MATRIX), runner_sha256=base.sha(optimized.HERE)))
    print('PASS: exact p-values, endpoint matrices, separating sets, bounded eviction and genuine resume.')


if __name__ == '__main__':
    main()
