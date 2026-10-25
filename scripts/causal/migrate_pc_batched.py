"""Read-only legacy PC migration: streaming cache removal, validation, NEW output only."""
import argparse
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import time

import discover as base
import pc_batched as batch
from discover_pc_batched import validate_inputs,ensure_idle
from causallearn.utils.cit import FisherZ

np=base.np
HERE=Path(__file__).resolve().parent


def strip_checkpoint(source,destination,log,ram_limit=768*2**20):
    if destination.exists() or destination.resolve()==source.resolve():
        raise ValueError('Strip output must be a NEW path distinct from source')
    command=['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(HERE/'strip_pc_cache.ps1'),
             '-InputPath',str(source.resolve()),'-OutputPath',str(destination.resolve())]
    peak=0;start=time.monotonic()
    with open(log,'w',encoding='utf-8') as stream:
        child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=stream,text=True,
                               creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            while child.poll() is None:
                try:
                    root=base.psutil.Process(child.pid)
                    memory=[p.memory_info() for p in [root]+root.children(recursive=True)]
                    peak=max(peak,sum(max(m.rss,getattr(m,'peak_wset',0)) for m in memory))
                    if peak>ram_limit:
                        raise MemoryError('Streaming converter exceeded configured RAM guard')
                except base.psutil.NoSuchProcess:
                    pass
                time.sleep(.1)
            stdout=child.stdout.read()
            if child.returncode:
                raise RuntimeError(f'Cache stripping failed; inspect {log}')
            result=json.loads(stdout)
        finally:
            if child.poll() is None:
                base.stop_worker_tree(child)
    result.update(process_peak_rss_bytes=peak,supervised_seconds=time.monotonic()-start,
                  memory_guard_bytes=ram_limit)
    return result


def validate_legacy_configuration(cfg):
    assert cfg['algorithm']=='pc' and cfg['stable'] is True and cfg['independence_test']=='fisherz'
    assert cfg['alpha']==.05 and cfg['uc_rule']==0 and cfg['uc_priority']==2
    assert cfg['max_depth'] is None and cfg['extra_standardization'] is False
    assert cfg['input_dtype']=='float64 cast from prepared float32'
    assert cfg['versions']==base.versions() and cfg['implementation_files']==base.package_fingerprints()
    for key,path in [('matrix_sha256',base.MATRIX),('gene_order_sha256',base.ORDER),('control_rows_sha256',base.CONTROL_ROWS)]:
        assert cfg[key]==base.sha(path),f'Input fingerprint changed: {key}'
    allowed=set(base.LEGACY_RECOVERY_CONTRACTS)|{base.sha(HERE/'discover.py')}
    assert cfg['wrapper_sha256'] in allowed,'Unreviewed legacy checkpoint producer'


def migrate(source,out,batch_size=4096,audit=False):
    if out.exists():
        raise SystemExit('Destination exists. Choose a NEW run ID; no overwrite allowed.')
    if source.resolve()==out.resolve():
        raise ValueError('Source/destination must differ')
    start=time.monotonic()
    oldcfg=json.loads((source/'config.json').read_text())
    oldstatus=json.loads((source/'status.json').read_text()) if (source/'status.json').exists() else {}
    if oldstatus.get('status')=='running':
        # Legacy supervisor and venv worker launcher may each have a live PID.
        for key in ('supervisor_pid','worker_pid'):
            pid=oldstatus.get(key,-1)
            if base.psutil.pid_exists(pid):
                raise SystemExit('Source run may still be active; refusing migration')
    validate_legacy_configuration(oldcfg)
    out.mkdir(parents=True)
    source_checkpoint=source/'pc_depth_checkpoint.pkl'
    before=dict(size=source_checkpoint.stat().st_size,mtime_ns=source_checkpoint.stat().st_mtime_ns,
                sha256=base.sha(source_checkpoint))
    base.atomic_json(out/'status.json',dict(status='migration_running',source=str(source.resolve())))
    base.event(out,'migration_started',source=str(source.resolve()),source_bytes=before['size'])
    manifest={}
    for name in ('selected_columns.npy','selected_rows.npy','selected_source_rows.npy','node_order.csv'):
        manifest[name]=base.sha(source/name)
        shutil.copyfile(source/name,out/name)
        assert base.sha(out/name)==manifest[name]
    x,names=validate_inputs(out)
    assert x.shape==(oldcfg['cells'],oldcfg['nodes'])
    cfg=batch.configuration(oldcfg['nodes'],oldcfg['cells'],oldcfg['seed'],batch_size,None,audit)
    base.atomic_json(out/'config.json',cfg)
    stripped=out/'migration_stripped.pkl'
    profile=strip_checkpoint(source_checkpoint,stripped,out/'migration_stream.log')
    base.atomic_json(out/'stream_profile.json',profile)
    if profile['OutputBytes']>256*2**20 or profile['OutputMemos']>1000000:
        raise MemoryError('Stripped checkpoint exceeds conservative deserialization guard; not loading it')
    # This load is safe only AFTER the cache and dense memo table have been stripped.
    with open(stripped,'rb') as stream:
        state=pickle.load(stream)
    cg=state['cg'];depth=state['completed_depth']
    assert isinstance(cg.test,FisherZ) and cg.test.method=='fisherz' and len(cg.test.pvalue_cache)==0
    assert isinstance(depth,int) and depth>=0
    assert cg.test.data.dtype==np.float64 and np.array_equal(cg.test.data,x)
    assert np.array_equal(cg.test.correlation_matrix,np.corrcoef(x.T))
    assert [n.get_name() for n in cg.G.nodes]==names.symbol.tolist()
    assert cg.G.graph.shape==(oldcfg['nodes'],oldcfg['nodes'])
    assert np.array_equal(cg.G.graph,cg.G.graph.T) and set(np.unique(cg.G.graph))<={0,-1}
    assert not np.any(np.diag(cg.G.graph)) and cg.sepset.shape==cg.G.graph.shape
    original_summary=batch.summary(cg)
    # Cross-check the latest completed-depth historical event, not an interrupted depth.
    events=[]
    with open(source/'progress.log',encoding='utf-8') as stream:
        for line in stream:
            event=json.loads(line)
            if event.get('event')=='pc_depth_complete' and event.get('depth')==depth:
                events.append(event)
    assert events,'No matching completed-depth provenance'
    expected=events[-1]
    assert expected['max_degree']==original_summary['max_degree'] and expected['skeleton_edges']==original_summary['edges']
    assert profile['CIKeys']==expected['ci_cache_entries']-3,'Unexpected legacy cache layout/count'
    replacement=batch.LeanFisherZ(x)
    assert np.array_equal(replacement.correlation_matrix,cg.test.correlation_matrix)
    cg.set_ind_test(replacement)
    assert batch.summary(cg)==original_summary
    checkpoint=out/'pc_depth_checkpoint.pkl'
    batch.save_checkpoint(checkpoint,cg,depth,{},batch.config_hash(cfg))
    restored=batch.load_checkpoint(checkpoint,x,batch.config_hash(cfg))
    assert batch.summary(restored['cg'])==original_summary and restored['completed_depth']==depth
    after=dict(size=source_checkpoint.stat().st_size,mtime_ns=source_checkpoint.stat().st_mtime_ns,
               sha256=base.sha(source_checkpoint))
    assert before==after,'Source checkpoint changed during migration; do not use destination'
    assert all(base.sha(source/name)==fingerprint for name,fingerprint in manifest.items())
    migration=dict(status='verified',source=str(source.resolve()),source_checkpoint=before,
        preserved_files_sha256=manifest,completed_depth=depth,next_depth=depth+1,
        state_summary=original_summary,stream_profile=profile,
        migrated_checkpoint_bytes=checkpoint.stat().st_size,migrated_checkpoint_sha256=base.sha(checkpoint),
        migrated_state_roundtrip_equal=True,source_unchanged=True,discovery_launched=False,
        elapsed_seconds=time.monotonic()-start,
        migration_tools_sha256={p.name:base.sha(p) for p in (HERE/'migrate_pc_batched.py',HERE/'PcCacheStripper.cs',HERE/'strip_pc_cache.ps1')})
    base.atomic_json(out/'migration_report.json',migration)
    base.atomic_json(out/'provenance.json',dict(migration=migration,legacy_configuration=oldcfg,
        git_commit=base.subprocess.check_output(['git','rev-parse','HEAD'],cwd=base.ROOT,text=True).strip(),
        limitations='Original whole-file normalize_total/log1p is transductive. Only training controls used. No graph discovery run by migration.'))
    base.atomic_json(out/'status.json',dict(status='migrated_ready',completed_depth=depth,next_depth=depth+1,attempt=0))
    base.event(out,'migration_verified',completed_depth=depth,next_depth=depth+1,checkpoint_bytes=checkpoint.stat().st_size)
    return migration


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',required=True,type=Path)
    p.add_argument('--run-id',required=True)
    p.add_argument('--batch-size',type=int,default=4096)
    p.add_argument('--audit-ci',action='store_true')
    args=p.parse_args()
    if not(args.run_id and all(c.isalnum() or c in '_-' for c in args.run_id)) or args.batch_size<=0:
        p.error('Invalid run ID/batch size')
    migrate(args.source_run,batch.BASE/args.run_id,args.batch_size,args.audit_ci)


if __name__=='__main__':
    main()
