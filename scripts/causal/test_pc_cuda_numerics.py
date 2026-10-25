"""Small numerical stress/threshold checks, no graph search or legacy-file loading."""
import argparse
from pathlib import Path
import discover as base
from pc_memory import MemoryFisherZ
from pc_cuda import BatchEngine
from scipy.stats import norm

np=base.np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise SystemExit('Output exists; choose a NEW path.')
    rng=np.random.default_rng(123)
    a=rng.normal(size=(1000,2)); a-=a.mean(axis=0)
    q=np.linalg.qr(a)[0]
    r=np.tanh(norm.ppf(.975)/np.sqrt(997))
    x=np.column_stack((q[:,0],r*q[:,0]+np.sqrt(1-r*r)*q[:,1]))
    reference=MemoryFisherZ(x,cache_entries=0)
    engine=BatchEngine(reference,'cuda',4096,.05,audit=True)
    returned=list(engine.batch([(0,1,())],0))[0][-1]
    assert returned==reference(0,1,()) and engine.stats['threshold_cpu_rechecks']==1
    results=[]
    for epsilon in (1.,1e-4,1e-6,1e-8):
        data=rng.normal(size=(1000,6))
        data[:,1]=data[:,0]+epsilon*data[:,1]
        test=MemoryFisherZ(data,cache_entries=0)
        item=dict(epsilon=epsilon,correlation_condition_number=float(np.linalg.cond(test.correlation_matrix)))
        try:
            gpu=BatchEngine(test,'cuda',4096,.05,audit=True)
            list(gpu.batch([(2,3,(0,1)),(4,5,(0,1))],2))
            item.update(status='completed',ci=gpu.stats)
        except (ValueError,RuntimeError) as exc:
            item.update(status='stopped',error=str(exc))
        results.append(item)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    base.atomic_json(args.output,dict(near_alpha_safeguard=engine.stats,stress_tests=results,
        note='Synthetic stress only; near-singular inputs are not guaranteed graph-equivalent. No regularization or pseudoinverse used.'))
    print(args.output)


if __name__=='__main__':
    main()
