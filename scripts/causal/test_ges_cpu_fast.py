"""Small exact helper, graph trajectory and checkpoint recovery validation."""
import argparse
import itertools
import json
import pickle
from pathlib import Path
import discover as b
import ges_cpu_fast as fast
from ges_recovery import build as reference
from causallearn.utils.GESUtils import insert_vc2_fast,find_subset_include


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise SystemExit('Use NEW output')
    args.output.mkdir(parents=True);rng=b.np.random.default_rng(20);count=0
    for trial in range(100):
        adjacency=rng.random((5,5))<.3;b.np.fill_diagonal(adjacency,False)
        successors={i:set(b.np.flatnonzero(adjacency[i])) for i in range(5)}
        for j,i in itertools.permutations(range(5),2):
            rest=[k for k in range(5) if k not in (j,i)]
            for mask in range(8):
                blocked={v for k,v in enumerate(rest) if mask&(1<<k)}
                assert insert_vc2_fast(j,i,blocked,successors)==fast.insert_vc2_fast(j,i,blocked,successors)
                count+=1
    for n in range(7):
        subsets=[list(c) for k in range(n+1) for c in itertools.combinations(range(n),k)]
        for subset in subsets:assert b.np.array_equal(find_subset_include(subset,subsets),fast.find_subset_include(subset,subsets))
    x=rng.normal(size=(800,10));x[:,2]=x[:,0]+x[:,1]+.4*x[:,2]
    kwargs=dict(score_func='local_score_BIC_from_cov',parameters={'lambda_value':2.})
    native=reference(lambda *a:None,args.output/'reference.pkl','test')(x,**kwargs)
    optimized=fast.build(lambda *a:None,args.output/'optimized.pkl','test')(x,**kwargs)
    def equal(a,c):
        assert b.np.array_equal(a['G'].graph,c['G'].graph) and a['score']==c['score']
        for key in ('update1','update2'):assert repr(a[key])==repr(c[key])
        for key in ('G_step1','G_step2'):
            assert len(a[key])==len(c[key]) and all(b.np.array_equal(g.graph,h.graph) for g,h in zip(a[key],c[key]))
    equal(native,optimized)
    class Stop(Exception):pass
    for phase in (1,2):
        path=args.output/f'phase{phase}.pkl'
        def saved(current,state):
            if current==phase and (phase==2 or len(state['update1'])>=1):raise Stop()
        try:fast.build(lambda *a:None,path,'test',on_saved=saved)(x,**kwargs)
        except Stop:pass
        with open(path,'rb') as stream:assert pickle.load(stream)['phase']==phase
        equal(native,fast.build(lambda *a:None,path,'test',resume=True)(x,**kwargs))
    b.atomic_json(args.output/'validation.json',dict(status='PASS',reachability_cases=count,subset_cases=True,
        native_moves_scores_endpoints_histories_equal=True,forward_backward_resume_equal=True))
    print('PASS',count,'reachability cases, subset helpers, exact search/history and resume')


if __name__=='__main__':main()
