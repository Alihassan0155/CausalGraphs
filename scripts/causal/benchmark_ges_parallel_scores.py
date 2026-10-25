"""Investigate ordered thread-pool scoring only; no GES search or production parallelism."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path
import discover as b
from causallearn.score.LocalScoreFunction import local_score_BIC_from_cov


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise SystemExit('Use NEW output')
    rng=b.np.random.default_rng(20);cols=b.np.sort(rng.choice(279,80,replace=False));rows=b.np.sort(rng.choice(10691,2000,replace=False))
    x=b.np.asarray(b.np.load(b.MATRIX,mmap_mode='r')[b.np.ix_(rows,cols)],dtype=b.np.float64)
    cov=b.np.cov(x.T,ddof=0);tasks=[]
    for i in range(80):
        for offset in range(1,21):tasks.append((i,[(i+offset)%80,(i+offset+21)%80]))
    def score(task):return local_score_BIC_from_cov((cov,2000),task[0],task[1],{'lambda_value':2.})
    start=time.perf_counter();reference=list(map(score,tasks));results=[dict(workers=1,seconds=time.perf_counter()-start,exact=True)]
    for workers in (2,4):
        start=time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:values=list(pool.map(score,tasks))
        results.append(dict(workers=workers,seconds=time.perf_counter()-start,exact=bool(b.np.array_equal(values,reference))))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    b.atomic_json(args.output,dict(tasks=len(tasks),results=results,
        note='Ordered independent score microbenchmark, NOT parallel full candidate evaluation. Shared caches/search not modified.'))
    print(results)


if __name__=='__main__':main()
