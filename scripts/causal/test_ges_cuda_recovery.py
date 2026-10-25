"""Small CUDA forward/backward recovery; does not touch existing experiments."""
import argparse
import pickle
from pathlib import Path
import discover as b
from ges_recovery import build
from ges_cuda_backend import CudaScore


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise SystemExit('Choose NEW output')
    args.output.mkdir(parents=True)
    rng=b.np.random.default_rng(20);x=rng.normal(size=(800,8));x[:,2]=x[:,0]+x[:,1]+.4*x[:,2]
    def runner(path,resume=False,saved=None):
        func=build(lambda *a:None,path,'cuda-test-v1',resume,saved)
        func.__globals__['LocalScoreClass']=CudaScore
        return func(x,score_func='local_score_BIC_from_cov',parameters={'lambda_value':2.})
    reference=runner(args.output/'reference.pkl');results=[]
    class Stop(Exception):pass
    for phase in (1,2):
        path=args.output/f'phase{phase}.pkl'
        def saved(current,state):
            if current==phase and (phase==2 or len(state['update1'])>=1):raise Stop()
        try:
            runner(path,saved=saved);raise AssertionError('Expected interruption')
        except Stop:pass
        with open(path,'rb') as stream:state=pickle.load(stream)
        assert state['state']['score_func'].device_cov is None
        actual=runner(path,resume=True)
        assert b.np.array_equal(reference['G'].graph,actual['G'].graph) and reference['score']==actual['score']
        assert repr(reference['update1'])==repr(actual['update1']) and repr(reference['update2'])==repr(actual['update2'])
        results.append(dict(phase=phase,endpoint_score_updates_equal=True,cuda_tensor_excluded_from_checkpoint=True))
    b.atomic_json(args.output/'validation.json',dict(status='PASS',cases=results,note='Equivalence to uninterrupted CUDA, not universal CPU equivalence.'))
    print(results)


if __name__=='__main__':main()
