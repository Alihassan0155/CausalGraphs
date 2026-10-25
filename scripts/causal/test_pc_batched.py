"""Small exact CPU equivalence, streaming migration and genuine resume validations."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import discover as base
import pc_batched as batch
from migrate_pc_batched import strip_checkpoint
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.PCUtils import SkeletonDiscovery as sd

np=base.np


def inputs(nodes):
    rng=np.random.default_rng(20)
    columns=np.sort(rng.choice(279,nodes,replace=False))
    rows=np.sort(rng.choice(10691,2000,replace=False))
    return np.asarray(np.load(base.MATRIX,mmap_mode='r')[np.ix_(rows,columns)],dtype=np.float64)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    if args.output.exists():
        raise SystemExit('Output exists; choose a NEW path')
    args.output.mkdir(parents=True)
    results=[]
    original=sd.skeleton_discovery
    for nodes,max_depth in ((20,None),(40,None),(80,2)):
        x=inputs(nodes)
        folder=args.output/f'n{nodes}';folder.mkdir()
        start=time.monotonic()
        native=pc(x,alpha=.05,stable=True,uc_rule=0,uc_priority=2,show_progress=False,max_k=max_depth)
        native_seconds=time.monotonic()-start
        cfg=batch.configuration(nodes,2000,20,4096,max_depth,True)
        start=time.monotonic()
        try:
            track,engine,_=batch.install_hooks(folder/'batch.pkl',x,cfg,lambda *a,**k:None)
            optimized=pc(x,alpha=.05,stable=True,uc_rule=0,uc_priority=2,show_progress=False,
                         max_k=max_depth,indep_test='lean_fisherz')
        finally:
            sd.skeleton_discovery=original
        assert batch.summary(native)==batch.summary(optimized)
        assert engine[0].stats['decision_mismatches']==0
        # Compare raw state before vs after streaming surgery, not only final graphs.
        class Interrupted(Exception):
            pass
        def interrupt(kind,**fields):
            if fields.get('depth')==1:
                raise Interrupted()
        legacy=folder/'legacy_depth1.pkl'
        try:
            base.install_pc_hooks(legacy,interrupt)
            try:
                pc(x,alpha=.05,stable=True,uc_rule=0,uc_priority=2,show_progress=False,max_k=max_depth)
                raise AssertionError('Expected depth-1 interruption')
            except Interrupted:
                pass
        finally:
            sd.skeleton_discovery=original
        source_sha=base.sha(legacy)
        profile=strip_checkpoint(legacy,folder/'stripped.pkl',folder/'strip.log')
        with open(legacy,'rb') as stream:
            old=pickle.load(stream)
        with open(folder/'stripped.pkl','rb') as stream:
            stripped=pickle.load(stream)
        assert old['completed_depth']==stripped['completed_depth']==1
        assert batch.summary(old['cg'])==batch.summary(stripped['cg'])
        assert np.array_equal(old['cg'].test.data,stripped['cg'].test.data)
        assert np.array_equal(old['cg'].test.correlation_matrix,stripped['cg'].test.correlation_matrix)
        assert stripped['cg'].test.pvalue_cache=={} and old['cg'].test.method==stripped['cg'].test.method
        assert profile['CIKeys']==len(old['cg'].test.pvalue_cache)-3
        cg=stripped['cg'];cg.set_ind_test(batch.LeanFisherZ(x))
        resume_path=folder/'migrated.pkl'
        batch.save_checkpoint(resume_path,cg,1,{},batch.config_hash(cfg))
        loaded=batch.load_checkpoint(resume_path,x,batch.config_hash(cfg))
        assert loaded['completed_depth']==1 and batch.summary(loaded['cg'])==batch.summary(old['cg'])
        events=[]
        try:
            track,resumed_engine,_=batch.install_hooks(resume_path,x,cfg,
                lambda kind,**kw:events.append(dict(event=kind,**kw)),resume=True)
            resumed=pc(x,alpha=.05,stable=True,uc_rule=0,uc_priority=2,show_progress=False,
                       max_k=max_depth,indep_test='lean_fisherz')
        finally:
            sd.skeleton_discovery=original
        assert events[0]['event']=='pc_state_resumed' and events[0]['depth']==1
        assert batch.summary(resumed)==batch.summary(native)
        assert source_sha==base.sha(legacy)
        results.append(dict(nodes=nodes,cells=2000,max_depth=max_depth,native_seconds=native_seconds,
            optimized_and_validation_seconds=time.monotonic()-start,
            endpoints_equal=True,sepsets_equal=True,input_equal=True,correlation_equal=True,
            resumed_from=1,next_depth=2,final_depth=track['last_completed_depth'],
            ci=engine[0].stats,resumed_ci=resumed_engine[0].stats,stream=profile,source_unchanged=True))
        base.atomic_json(args.output/'partial_results.json',results)
        print('PASS',nodes,engine[0].stats,flush=True)
    # A reference to a discarded CI value outside the cache must FAIL, not silently corrupt state.
    with open(args.output/'n40/legacy_depth1.pkl','rb') as stream:
        alias=pickle.load(stream)
    cache=alias['cg'].test.pvalue_cache
    key=next(k for k in cache if ';' in k)
    alias['external_ci_value']=cache[key]
    alias_source=args.output/'unsupported_alias.pkl'
    base.atomic_pickle(alias_source,alias)
    try:
        strip_checkpoint(alias_source,args.output/'unsupported_alias_output.pkl',args.output/'unsupported_alias.log')
        raise AssertionError('Expected unsupported cache-value external alias to fail closed')
    except RuntimeError:
        pass
    # Lean checkpoint ignores partial .tmp and reattaches verified expression on load.
    valid=args.output/'n40/migrated.pkl'
    valid.with_suffix('.tmp').write_bytes(b'partial interrupted pickle')
    cfg=batch.configuration(40,2000,20,4096,None,True)
    checked=batch.load_checkpoint(valid,inputs(40),batch.config_hash(cfg))
    assert checked['completed_depth']==12
    base.atomic_json(args.output/'summary.json',dict(results=results,unsupported_external_alias_rejected=True,
        partial_temporary_file_ignored=True,cuda_imported=('torch' in sys.modules),
        sources={path.name:base.sha(path) for path in (Path(__file__),Path(batch.__file__),Path(__file__).with_name('PcCacheStripper.cs'))}))
    assert 'torch' not in sys.modules
    print('PASS: native/batched/migrated/resumed graphs, separating sets, p-values and fail-closed cache references.')


if __name__=='__main__':
    main()
