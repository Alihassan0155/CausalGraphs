"""Bounded 40/80-node CPU/GPU comparisons and small legacy-state migration test."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time

import discover as base
import pc_cuda
from pc_memory import MemoryFisherZ
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.PCUtils import SkeletonDiscovery as sd

np,torch = base.np,pc_cuda.torch


def migration_test(out):
    rng = np.random.default_rng(20)
    cols = np.sort(rng.choice(279,40,replace=False))
    rows = np.sort(rng.choice(10691,2000,replace=False))
    x = np.asarray(np.load(base.MATRIX,mmap_mode='r')[np.ix_(rows,cols)],dtype=np.float64)
    class Interrupted(Exception):
        pass
    def interrupt(kind,**fields):
        if fields.get('depth')==1:
            raise Interrupted()
    original = sd.skeleton_discovery
    legacy = out/'small_legacy_checkpoint.pkl'
    try:
        base.install_pc_hooks(legacy,interrupt)
        try:
            pc(x,alpha=.05,stable=True,uc_rule=0,uc_priority=2,show_progress=False)
            raise AssertionError('Expected checkpoint interruption')
        except Interrupted:
            pass
    finally:
        sd.skeleton_discovery=original
    old_sha = base.sha(legacy)
    with open(legacy,'rb') as stream:
        state = pickle.load(stream)
    assert state['completed_depth']==1
    endpoints = state['cg'].G.graph.copy()
    sepsets = repr(state['cg'].sepset.tolist())
    state = pc_cuda.migrate_state(state,x)
    assert np.array_equal(endpoints,state['cg'].G.graph)
    assert repr(state['cg'].sepset.tolist())==sepsets
    target = out/'small_migrated_checkpoint.pkl'
    base.atomic_pickle(target,state)
    baseline = pc(x,alpha=.05,stable=True,uc_rule=0,uc_priority=2,show_progress=False)
    events = []
    try:
        track,engine,_ = pc_cuda.install_hooks(target,lambda kind,**kw:events.append(dict(event=kind,**kw)),
                                                backend='cuda',audit=True,resume=True)
        result = pc(x,alpha=.05,stable=True,uc_rule=0,uc_priority=2,show_progress=False,
                    indep_test='memory_fisherz',cache_entries=0)
    finally:
        sd.skeleton_discovery=original
    assert events[0]['event']=='pc_state_resumed' and events[0]['depth']==1
    assert np.array_equal(baseline.G.graph,result.G.graph)
    assert repr(baseline.sepset.tolist())==repr(result.sepset.tolist())
    assert old_sha==base.sha(legacy)
    return dict(source_depth=1,next_depth=2,final_depth=track['last_completed_depth'],
                endpoints_equal=True,sepsets_equal=True,source_unchanged=True,
                ci=engine[0].stats,full_legacy_checkpoint_loaded=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-prefix',required=True)
    args = p.parse_args()
    if not (args.run_prefix and all(c.isalnum() or c in '_-' for c in args.run_prefix)):
        p.error('Use a simple prefix')
    out = base.BASE/'pc_cuda_pilot'/f'{args.run_prefix}_report'
    if out.exists():
        raise SystemExit('Report exists; choose a NEW prefix.')
    out.mkdir(parents=True)
    results = []
    for nodes,depth in ((40,-1),(80,2)):
        pair = []
        for backend,audit in (('cpu',False),('cpu-batch',False),('cuda',False),('cuda',True)):
            name=f'{args.run_prefix}_n{nodes}_{backend.replace("-","_")}{"_audit" if audit else ""}'
            command=[sys.executable,'-u',str(Path(__file__).with_name('discover_pc_cuda.py')),
                     '--run-id',name,'--backend',backend,'--nodes',str(nodes),'--cells','2000',
                     '--seed','20','--max-depth',str(depth),'--batch-size','4096']
            if audit:
                command.append('--audit-ci')
            print('START',name,flush=True)
            started=time.perf_counter()
            with open(out/f'{name}.log','w') as log:
                child=subprocess.Popen(command,cwd=base.ROOT,stdout=log,stderr=subprocess.STDOUT)
                try:
                    child.wait(timeout=90)
                except BaseException:
                    base.stop_worker_tree(child)
                    raise
                if child.returncode:
                    raise RuntimeError(f'{name} failed; inspect {out/name}.log')
            stat=json.loads((base.BASE/'pc_cuda_pilot'/name/'statistics.json').read_text())
            stat.update(run_id=name,backend=backend,audit=audit,nodes=nodes,cells=2000,
                        process_end_to_end_seconds=time.perf_counter()-started)
            pair.append(stat); results.append(stat)
            print('DONE',name,round(stat['algorithm_seconds'],3),round(stat['process_end_to_end_seconds'],3),flush=True)
            base.atomic_json(out/'partial_results.json',results)
        reference=pair[0]
        for candidate in pair[1:]:
            assert candidate['input_sha256']==reference['input_sha256']
            assert candidate['sepset_sha256']==reference['sepset_sha256']
            assert np.array_equal(np.load(base.BASE/'pc_cuda_pilot'/reference['run_id']/'endpoint_matrix.npy'),
                                  np.load(base.BASE/'pc_cuda_pilot'/candidate['run_id']/'endpoint_matrix.npy'))
        assert pair[-1]['ci']['returned_decision_mismatches']==0
    migration=migration_test(out)
    base.atomic_json(out/'summary.json',dict(results=results,migration=migration,
        source_sha256={path.name:base.sha(path) for path in (Path(__file__),Path(pc_cuda.__file__),Path(__file__).with_name('discover_pc_cuda.py'))},
        cuda_device=torch.cuda.get_device_name(),driver=base.subprocess.check_output(
            ['nvidia-smi','--query-gpu=driver_version,memory.total,memory.free','--format=csv,noheader'],text=True).strip(),
        notes='40-node unrestricted, 80-node depth-2-truncated feasibility only. CPU-batch isolates batching/search optimization from CUDA. Audit runs are excluded from performance speedups.'))
    print('PASS: CPU/batched CPU/CUDA endpoints and sepsets; all audited decisions; small legacy migration.',flush=True)


if __name__=='__main__':
    main()
