"""CPU-only batched stable-PC machinery and lean, data-fingerprinted checkpoints."""
import hashlib
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import pickle
import sys

import discover as base
from pc_memory import MemoryFisherZ
from causallearn.utils.cit import register_ci_test
from scipy.stats import norm

np = base.np
FORMAT = 'cpu-batched-pc-v1'
BASE = base.BASE/'pc_batched'


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def summary(cg):
    return dict(endpoint_sha256=array_hash(cg.G.graph),
                sepset_sha256=hashlib.sha256(repr(cg.sepset.tolist()).encode()).hexdigest(),
                node_names=[n.get_name() for n in cg.G.nodes],
                edges=int(np.count_nonzero(cg.G.graph)//2),max_degree=int(cg.max_degree()))


class LeanFisherZ(MemoryFisherZ):
    def __init__(self,data,**kwargs):
        kwargs.pop('cache_entries',None)
        super().__init__(data,cache_entries=0,**kwargs)
        self.input_sha256=array_hash(data)

    def __getstate__(self):
        state=self.__dict__.copy()
        state['data']=None  # reconstruct ONLY from verified prepared input, not every depth snapshot
        return state

    def bind(self,data):
        if data.dtype!=np.float64 or data.shape!=(self.sample_size,self.num_features) or array_hash(data)!=self.input_sha256:
            raise ValueError('Checkpoint expression matrix fingerprint differs')
        if not np.array_equal(np.corrcoef(data.T),self.correlation_matrix):
            raise ValueError('Checkpoint correlation differs')
        self.data=data


register_ci_test('lean_fisherz',LeanFisherZ)


class CpuBatch:
    def __init__(self,test,batch_size,audit=False,prior=None):
        self.test,self.batch_size,self.audit=test,batch_size,audit
        self.stats=prior.copy() if prior else dict(queries=0,unique_computations=0,batches=0,
            audited_queries=0,nonidentical_pvalues=0,max_abs_p_difference=0.,decision_mismatches=0)
        self.position=dict(depth=None,x=None,y=None)

    @staticmethod
    def queries(cg,depth):
        neighbors=[cg.neighbors(x) for x in range(cg.G.graph.shape[0])]
        for x,adjacent in enumerate(neighbors):
            if len(adjacent)<depth-1:
                continue
            for y in adjacent:
                rest=np.delete(adjacent,np.where(adjacent==y))
                for s in itertools.combinations(rest,depth):
                    yield x,int(y),s

    def batch(self,queries,depth,alpha):
        unique=[]; lookup={}; positions=[]
        for x,y,s in queries:
            key=(min(x,y),max(x,y),*sorted(set(map(int,s))))
            if key not in lookup:
                lookup[key]=len(unique);unique.append(key)
            positions.append(lookup[key])
        idx=np.asarray(unique,dtype=np.int64)
        sub=self.test.correlation_matrix[idx[:,:,None],idx[:,None,:]]
        # NumPy uses its original float64 inverse gufunc for both scalar and
        # stacked matrices. Keep the native scalar sqrt/log expression order.
        inverse=np.linalg.inv(sub)
        z=[]
        for mat in inverse:
            r=-mat[0,1]/math.sqrt(abs(mat[0,0]*mat[1,1]))
            if not np.isfinite(r):
                raise ValueError('Nonfinite partial correlation; no regularization substituted')
            if abs(r)>=1:
                r=(1.-np.finfo(float).eps)*np.sign(r)
            z.append(math.sqrt(self.test.sample_size-depth-3)*abs(.5*math.log((1+r)/(1-r))))
        pvalues=2*(1-norm.cdf(np.abs(z)))
        self.stats['batches']+=1;self.stats['unique_computations']+=len(unique)
        for (x,y,s),index in zip(queries,positions):
            p=pvalues[index]
            self.position.update(depth=depth,x=x,y=y)
            self.stats['queries']+=1
            if self.audit:
                reference=self.test(x,y,s)
                self.stats['audited_queries']+=1
                self.stats['nonidentical_pvalues']+=int(p!=reference)
                self.stats['max_abs_p_difference']=max(self.stats['max_abs_p_difference'],float(abs(p-reference)))
                if (p>alpha)!=(reference>alpha):
                    self.stats['decision_mismatches']+=1
                    raise RuntimeError('Native/batched CPU independence decision differs; stopping')
            yield x,y,s,p

    def stream(self,cg,depth,alpha):
        source=iter(self.queries(cg,depth))
        while True:
            batch=list(itertools.islice(source,self.batch_size))
            if not batch:
                break
            yield from self.batch(batch,depth,alpha)


def save_checkpoint(path,cg,depth,counters,configuration_sha256):
    value=dict(format=FORMAT,cg=cg,completed_depth=depth,ci=counters.copy(),
               configuration_sha256=configuration_sha256,state_summary=summary(cg))
    temporary=path.with_suffix('.tmp')
    with open(temporary,'wb') as stream:
        pickle.dump(value,stream,protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def load_checkpoint(path,data,configuration_sha256):
    with open(path,'rb') as stream:
        state=pickle.load(stream)
    if state.get('format')!=FORMAT or state['configuration_sha256']!=configuration_sha256:
        raise ValueError('Checkpoint format/configuration mismatch')
    if not isinstance(state['cg'].test,LeanFisherZ) or summary(state['cg'])!=state['state_summary']:
        raise ValueError('Checkpoint graph/separating-set integrity mismatch')
    state['cg'].test.bind(data)
    return state


def configuration(nodes,cells,seed,batch_size,max_depth=None,audit=False):
    folder=Path(__file__).parent
    return dict(nodes=nodes,cells=cells,seed=seed,alpha=.05,stable=True,uc_rule=0,uc_priority=2,
                dtype='float64',batch_size=batch_size,max_depth=max_depth,audit_ci=audit,
                input_sha256={p.name:base.sha(p) for p in (base.MATRIX,base.ORDER,base.CONTROL_ROWS)},
                versions=base.versions(),python=sys.version,
                source_sha256={name:base.sha(folder/name) for name in
                    ('discover_pc_batched.py','pc_batched.py','pc_memory.py','discover.py')},
                upstream=base.package_fingerprints(),
                ci_orientation={name:base.sha(base.ISOLATED/'causallearn'/path) for name,path in (
                    ('cit','utils/cit.py'),('uc','utils/PCUtils/UCSepset.py'),('meek','utils/PCUtils/Meek.py'))},
                threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()


def install_hooks(checkpoint,data,cfg,report,resume=False):
    from causallearn.utils.PCUtils import SkeletonDiscovery as sd
    original=sd.skeleton_discovery
    source=inspect.getsource(original)
    anchors=['    depth = -1','        edge_removal = []',
        '                for S in combinations(Neigh_x_noy, depth):\n                    p = cg.ci_test(x, y, S)',
        '    if show_progress:\n        pbar.close()']
    assert all(source.count(a)==1 for a in anchors), 'Upstream stable-PC anchors changed'
    fingerprint=config_hash(cfg)
    initial=load_checkpoint(checkpoint,data,fingerprint) if resume else None
    tracking=dict(last_completed_depth=-1,max_degree=None,initial_depth=-1)
    engine=[None];response=[None];prior=[{}]
    def restore(cg,depth):
        nonlocal initial
        if initial is not None:
            cg,depth=initial['cg'],initial['completed_depth'];prior[0]=initial['ci'];initial=None
            tracking.update(last_completed_depth=depth,initial_depth=depth,max_degree=int(cg.max_degree()))
            report('pc_state_resumed',depth=depth,next_depth=depth+1)
        return cg,depth
    def begin(cg,depth,stable,knowledge,alpha):
        assert stable and knowledge is None and alpha==.05
        if engine[0] is None:
            engine[0]=CpuBatch(cg.test,cfg['batch_size'],cfg['audit_ci'],prior[0])
        response[0]=iter(engine[0].stream(cg,depth,alpha))
        report('pc_depth_started',depth=depth,max_degree=int(cg.max_degree()))
    def next_value(x,y,s):
        a,b,condition,p=next(response[0])
        assert a==x and b==y and condition==s, 'Native/batched query ordering differs'
        return p
    def save(cg,depth):
        assert next(response[0],None) is None
        save_checkpoint(checkpoint,cg,depth,engine[0].stats,fingerprint)
        tracking.update(last_completed_depth=depth,max_degree=int(cg.max_degree()))
        report('pc_depth_complete',depth=depth,max_degree=tracking['max_degree'],ci=engine[0].stats.copy(),
               skeleton_edges=int(np.count_nonzero(cg.G.graph)//2),checkpoint_bytes=checkpoint.stat().st_size)
    source=source.replace(anchors[0],anchors[0]+'\n    cg, depth = _batched_restore(cg, depth)')
    source=source.replace(anchors[1],anchors[1]+'\n        _batched_begin(cg, depth, stable, background_knowledge, alpha)')
    source=source.replace(anchors[2],anchors[2].replace('cg.ci_test(x, y, S)','_batched_next(x, y, S)'))
    source=source.replace('edge_removal = []','edge_removal = set()').replace('edge_removal.append(','edge_removal.add(')
    source=source.replace(anchors[3],'        _batched_save(cg, depth)\n\n'+anchors[3])
    namespace=dict(sd.__dict__,_batched_restore=restore,_batched_begin=begin,_batched_next=next_value,_batched_save=save)
    exec(compile(source,inspect.getfile(original),'exec'),namespace)
    sd.skeleton_discovery=namespace['skeleton_discovery']
    return tracking,engine,original
