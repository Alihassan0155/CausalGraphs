"""Small real-process recovery/protection test for the production CPU runner."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import discover as base
import pc_batched as batch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id',required=True)
    args=p.parse_args()
    if not(args.run_id and all(c.isalnum() or c in '_-' for c in args.run_id)):
        p.error('Use a simple run ID')
    out=batch.BASE/args.run_id
    if out.exists():
        raise SystemExit('Use a NEW test run ID')
    command=[sys.executable,'-u',str(Path(__file__).with_name('discover_pc_batched.py')),
        '--run-id',args.run_id,'--nodes','40','--cells','2000','--seed','20','--batch-size','4096','--audit-ci']
    # Authorize interruption ONLY of this newly created validation process tree.
    child=subprocess.Popen(command,cwd=base.ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    start=time.monotonic()
    try:
        while child.poll() is None:
            path=out/'progress.log'
            if path.exists() and 'pc_depth_complete' in path.read_text():
                base.stop_worker_tree(child);break
            if time.monotonic()-start>20:
                raise RuntimeError('Checkpoint test budget exceeded')
            time.sleep(.01)
        assert child.returncode!=0,'Run completed before interruption test'
    finally:
        if child.poll() is None:base.stop_worker_tree(child)
    records=[json.loads(line) for line in (out/'progress.log').read_text().splitlines()]
    saved=max(v['depth'] for v in records if v['event']=='pc_depth_complete')
    with open(out/'resume_test.log','w') as stream:
        result=subprocess.run([*command,'--resume'],cwd=base.ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=30)
    assert result.returncode==0
    stats=json.loads((out/'statistics.json').read_text())
    reference=base.BASE/'memory_optimization/validation_v3/n40_original'
    assert base.np.array_equal(base.np.load(out/'endpoint_matrix.npy'),base.np.load(reference/'endpoint_matrix.npy'))
    assert stats['sepset_sha256']==json.loads((reference/'measurement.json').read_text())['sepset_sha256']
    assert stats['resumed_from_depth']==saved and stats['ci']['decision_mismatches']==0
    assert stats['ci']['nonidentical_pvalues']==0
    protected=subprocess.run([*command,'--resume'],capture_output=True,text=True,timeout=10)
    assert protected.returncode!=0 and 'Completed experiment protected' in protected.stdout+protected.stderr
    base.atomic_json(out/'cli_validation.json',dict(interrupted_completed_depth=saved,next_depth=saved+1,
        final_depth=stats['last_completed_conditioning_depth'],endpoints_equal=True,sepsets_equal=True,
        ci=stats['ci'],completed_run_protected=True,existing_processes_untouched=True))
    print('PASS actual CLI resume from depth',saved,'; native endpoints/sepsets/p-values; protection')


if __name__=='__main__':main()
