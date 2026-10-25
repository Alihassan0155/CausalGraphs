"""Small exact native vs forward/backward interrupted GES recovery checks."""
import argparse
import pickle
from pathlib import Path
import discover as b
from ges_recovery import build
from causallearn.search.ScoreBased.GES import ges


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise SystemExit('Choose NEW test output')
    args.output.mkdir(parents=True)
    rng=b.np.random.default_rng(20);x=rng.normal(size=(800,10))
    x[:,2]=x[:,0]+x[:,1]+.4*x[:,2];x[:,3]=x[:,2]+.6*x[:,3]
    kwargs=dict(score_func='local_score_BIC_from_cov',parameters={'lambda_value':2.})
    native=ges(x,**kwargs)
    results=[]
    class Stop(Exception):pass
    for phase in (1,2):
        path=args.output/f'phase{phase}.pkl'
        def interrupt(current,state):
            if current==phase and (phase==2 or len(state['update1'])>=2):raise Stop()
        try:
            build(lambda *a:None,path,'test-v1',on_saved=interrupt)(x,**kwargs)
            raise AssertionError('Expected interruption')
        except Stop:pass
        with open(path,'rb') as stream:s=pickle.load(stream)
        assert s['phase']==phase
        path.with_suffix('.tmp').write_bytes(b'incomplete checkpoint')
        restored=build(lambda *a:None,path,'test-v1',resume=True)(x,**kwargs)
        assert b.np.array_equal(native['G'].graph,restored['G'].graph)
        assert native['score']==restored['score']
        for key in ('update1','update2'):assert repr(native[key])==repr(restored[key])
        for key in ('G_step1','G_step2'):
            assert len(native[key])==len(restored[key])
            assert all(b.np.array_equal(a.graph,c.graph) for a,c in zip(native[key],restored[key]))
        results.append(dict(phase=phase,forward_moves_saved=len(s['state']['update1']),endpoints_equal=True,score_equal=True,updates_and_history_equal=True))
    try:
        build(lambda *a:None,path,'wrong-config',resume=True)
        raise AssertionError('Expected config rejection')
    except ValueError:pass
    b.atomic_json(args.output/'validation.json',dict(status='PASS',cases=results,partial_tmp_ignored=True,configuration_mismatch_rejected=True))
    print(results)


if __name__=='__main__':main()
