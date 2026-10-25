"""Kill only a newly created 40-node pilot, then test real CLI checkpoint recovery."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import discover as base


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id',required=True)
    p.add_argument('--reference-run',required=True)
    args=p.parse_args()
    if not all(v and all(c.isalnum() or c in '_-' for c in v) for v in (args.run_id,args.reference_run)):
        p.error('Use simple run IDs')
    out=base.BASE/'pc_cuda_pilot'/args.run_id
    if out.exists():
        raise SystemExit('Test run exists; use a NEW ID.')
    command=[sys.executable,'-u',str(Path(__file__).with_name('discover_pc_cuda.py')),
             '--run-id',args.run_id,'--nodes','40','--cells','2000','--seed','20',
             '--backend','cuda','--batch-size','4096','--audit-ci']
    # This child is the ONLY process tree this test is authorized to stop.
    child=subprocess.Popen(command,cwd=base.ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    start=time.monotonic()
    try:
        while child.poll() is None:
            log=out/'progress.log'
            if log.exists() and 'pc_depth_complete' in log.read_text():
                base.stop_worker_tree(child)
                break
            if time.monotonic()-start>30:
                raise RuntimeError('No checkpoint within lightweight test budget')
            time.sleep(.02)
        if child.returncode==0:
            raise AssertionError('Pilot finished before interruption could be tested')
    finally:
        if child.poll() is None:
            base.stop_worker_tree(child)
    before=[json.loads(line) for line in (out/'progress.log').read_text().splitlines()]
    saved_depth=max(v['depth'] for v in before if v['event']=='pc_depth_complete')
    with open(out/'resume_test.log','w') as stream:
        resumed=subprocess.Popen([*command,'--resume'],cwd=base.ROOT,stdout=stream,stderr=subprocess.STDOUT)
        try:
            resumed.wait(timeout=30)
        except BaseException:
            base.stop_worker_tree(resumed)
            raise
    assert resumed.returncode==0
    stat=json.loads((out/'statistics.json').read_text())
    reference=base.BASE/'pc_cuda_pilot'/args.reference_run
    reference_stat=json.loads((reference/'statistics.json').read_text())
    assert base.np.array_equal(base.np.load(out/'endpoint_matrix.npy'),base.np.load(reference/'endpoint_matrix.npy'))
    assert stat['sepset_sha256']==reference_stat['sepset_sha256']
    events=[json.loads(line) for line in (out/'progress.log').read_text().splitlines()]
    restored=[v for v in events if v['event']=='pc_state_resumed']
    assert restored[-1]['depth']==saved_depth
    protected=subprocess.run([*command,'--resume'],capture_output=True,text=True,timeout=15)
    assert protected.returncode!=0 and 'Completed experiment protected' in protected.stderr+protected.stdout
    base.atomic_json(out/'resume_validation.json',dict(saved_depth=saved_depth,next_depth=saved_depth+1,
        final_depth=stat['last_completed_conditioning_depth'],endpoints_equal=True,sepsets_equal=True,
        audited_decisions=stat['ci'],completed_run_protected=True,
        note='Only this newly created test process was killed. Existing experiments untouched.'))
    print('PASS: real GPU CLI recovery from depth',saved_depth,'and completed-run overwrite protection.')


if __name__=='__main__':
    main()
