"""Experimental streamed/batched Fisher-Z; installed PC-stable search is unchanged."""
import inspect
import itertools
import math
import pickle
import time

import discover as base
from pc_memory import MemoryFisherZ
import torch
from scipy.stats import norm
from causallearn.utils.cit import FisherZ

np = base.np


class BatchEngine:
    def __init__(self, test, backend, batch_size, alpha, audit=False):
        self.test, self.backend, self.batch_size, self.alpha = test, backend, batch_size, alpha
        self.audit = audit
        self.reference = MemoryFisherZ(test.data, cache_entries=0) if audit else None
        self.stats = dict(queries=0, unique_computations=0, batches=0, audited_queries=0,
                          max_abs_p_difference=0., raw_decision_mismatches=0,
                          returned_decision_mismatches=0, threshold_cpu_rechecks=0,
                          minimum_raw_alpha_distance=1., mismatch_examples=[])
        if backend == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable. No CPU fallback permitted.')
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.cuda.reset_peak_memory_stats()
            self.correlation = torch.tensor(test.correlation_matrix, dtype=torch.float64, device='cuda')
            torch.linalg.inv(torch.eye(3, dtype=torch.float64, device='cuda'))
            torch.cuda.synchronize()
        else:
            self.correlation = test.correlation_matrix

    @staticmethod
    def queries(cg, depth):
        # Identical ordering/eligibility to the installed skeleton_discovery loop.
        for x in range(cg.G.graph.shape[0]):
            neighbors = cg.neighbors(x)
            if len(neighbors) < depth - 1:
                continue
            for y in neighbors:
                rest = np.delete(neighbors, np.where(neighbors == y))
                for s in itertools.combinations(rest, depth):
                    yield x, int(y), s

    def batch(self, queries, depth):
        # Deduplicate only within a bounded batch, not an unbounded history cache.
        unique, lookup, positions = [], {}, []
        for x,y,s in queries:
            key = (min(x,y),max(x,y),*sorted(set(map(int,s))))
            if key not in lookup:
                lookup[key] = len(unique); unique.append(key)
            positions.append(lookup[key])
        indices = np.asarray(unique, dtype=np.int64)
        if self.backend == 'cuda':
            idx = torch.as_tensor(indices, dtype=torch.int64, device='cuda')
            matrices = self.correlation[idx[:,:,None], idx[:,None,:]]
            inverse, info = torch.linalg.inv_ex(matrices, check_errors=False)
            if bool(torch.any(info != 0).item()):
                raise ValueError('Singular Fisher-Z correlation submatrix; no regularization substituted.')
            r = (-inverse[:,0,1] / torch.sqrt(torch.abs(inverse[:,0,0]*inverse[:,1,1]))).cpu().numpy()
        else:
            matrices = self.correlation[indices[:,:,None],indices[:,None,:]]
            inverse = np.linalg.inv(matrices)
            r = -inverse[:,0,1]/np.sqrt(np.abs(inverse[:,0,0]*inverse[:,1,1]))
        if not np.isfinite(r).all():
            raise ValueError('Nonfinite partial correlations; stopping rather than replacing values.')
        # Keep original clipping, scalar math.log/sqrt and SciPy normal CDF on CPU.
        r = np.where(np.abs(r)>=1, (1.-np.finfo(float).eps)*np.sign(r), r)
        z = np.asarray([math.sqrt(self.test.sample_size-depth-3)*abs(.5*math.log((1+v)/(1-v))) for v in r])
        pvalues = 2*(1-norm.cdf(np.abs(z)))
        self.stats['batches'] += 1
        self.stats['unique_computations'] += len(unique)
        for query, pos in zip(queries, positions):
            x,y,s = query
            raw = float(pvalues[pos]); p = raw
            self.stats['queries'] += 1
            self.stats['minimum_raw_alpha_distance'] = min(self.stats['minimum_raw_alpha_distance'],abs(raw-self.alpha))
            reference = None
            if abs(raw-self.alpha) <= 1e-8:
                # Explicit near-threshold exact-native safeguard, always reported.
                reference = float(self.test(x,y,s))
                p = reference
                self.stats['threshold_cpu_rechecks'] += 1
            if self.audit:
                if reference is None:
                    reference = float(self.reference(x,y,s))
                self.stats['audited_queries'] += 1
                self.stats['max_abs_p_difference'] = max(self.stats['max_abs_p_difference'],abs(raw-reference))
                raw_bad = (raw>self.alpha)!=(reference>self.alpha)
                bad = (p>self.alpha)!=(reference>self.alpha)
                self.stats['raw_decision_mismatches'] += int(raw_bad)
                self.stats['returned_decision_mismatches'] += int(bad)
                if raw_bad and len(self.stats['mismatch_examples'])<10:
                    self.stats['mismatch_examples'].append(dict(x=x,y=y,s=list(map(int,s)),gpu_p=raw,cpu_p=reference))
                if bad:
                    raise RuntimeError('GPU/CPU independence decision differs outside safeguard; pilot validation failed.')
            yield x,y,s,p

    def stream(self, cg, depth):
        iterator = iter(self.queries(cg,depth))
        while True:
            batch = list(itertools.islice(iterator,self.batch_size))
            if not batch:
                break
            yield from self.batch(batch,depth)


def migrate_state(state, data):
    """Trusted, already-loaded CPU state -> small CPU-portable pilot state.

    Loading the huge legacy pickle is NOT memory-safe automatically. This function
    is validated on small checkpoints only; it never loads a source path itself.
    """
    cg = state['cg']
    if not isinstance(cg.test,FisherZ) or cg.test.method != 'fisherz':
        raise ValueError('Only original Fisher-Z skeleton checkpoints are supported.')
    if not np.array_equal(cg.test.data,data):
        raise ValueError('Checkpoint input differs from supplied data.')
    replacement = MemoryFisherZ(data,cache_entries=0)
    if not np.array_equal(replacement.correlation_matrix,cg.test.correlation_matrix):
        raise ValueError('Correlation matrix differs; refusing migration.')
    cg.set_ind_test(replacement)
    return dict(cg=cg,completed_depth=state['completed_depth'],pilot_format=1)


def install_hooks(checkpoint, report, backend='cuda', batch_size=4096, audit=False, resume=False):
    from causallearn.utils.PCUtils import SkeletonDiscovery as sd
    original = sd.skeleton_discovery
    source = inspect.getsource(original)
    anchors = ['    depth = -1','        edge_removal = []',
               '                for S in combinations(Neigh_x_noy, depth):\n                    p = cg.ci_test(x, y, S)',
               '    if show_progress:\n        pbar.close()']
    assert all(source.count(a)==1 for a in anchors), 'Upstream PC instrumentation anchors changed'
    initial = None
    if resume:
        with open(checkpoint,'rb') as stream:
            initial = pickle.load(stream)
        assert initial.get('pilot_format')==1, 'Use only verified pilot-format checkpoints'
    track = dict(last_completed_depth=-1,max_degree=None)
    engine = [None]
    responses = [None]
    def restore(cg,depth):
        nonlocal initial
        if initial is not None:
            cg,depth = initial['cg'],initial['completed_depth']; initial = None
            track.update(last_completed_depth=depth,max_degree=int(cg.max_degree()))
            report('pc_state_resumed',depth=depth)
        return cg,depth
    def begin(cg,depth,stable,background_knowledge,alpha):
        assert stable and background_knowledge is None, 'Pilot supports audited PC-stable/no background knowledge only'
        if backend!='cpu':
            if engine[0] is None:
                engine[0] = BatchEngine(cg.test,backend,batch_size,alpha,audit)
            responses[0] = iter(engine[0].stream(cg,depth))
    def next_value(x,y,s):
        a,b,condition,p = next(responses[0])
        assert a==x and b==y and condition==s, 'Batched query order diverged from native PC'
        return p
    def save(cg,depth):
        if responses[0] is not None:
            assert next(responses[0],None) is None
        base.atomic_pickle(checkpoint,dict(cg=cg,completed_depth=depth,pilot_format=1))
        track.update(last_completed_depth=depth,max_degree=int(cg.max_degree()))
        report('pc_depth_complete',depth=depth,max_degree=track['max_degree'],
               skeleton_edges=int(np.count_nonzero(cg.G.graph)//2),
               checkpoint_bytes=checkpoint.stat().st_size,
               ci=engine[0].stats.copy() if engine[0] else {})
    source = source.replace(anchors[0],anchors[0]+'\n    cg, depth = _gpu_restore(cg, depth)')
    source = source.replace(anchors[1],anchors[1]+'\n        _gpu_begin(cg, depth, stable, background_knowledge, alpha)')
    if backend!='cpu':
        source = source.replace(anchors[2],anchors[2].replace('cg.ci_test(x, y, S)','_gpu_next(x, y, S)'))
        # Stable PC defers all removals; duplicates have no scientific meaning.
        # Bound removal storage to O(edges) while retaining the exact sepset loop.
        source = source.replace('edge_removal = []','edge_removal = set()')
        source = source.replace('edge_removal.append(', 'edge_removal.add(')
    source = source.replace(anchors[3],'        _gpu_save(cg, depth)\n\n'+anchors[3])
    namespace = dict(sd.__dict__,_gpu_restore=restore,_gpu_begin=begin,_gpu_next=next_value,_gpu_save=save)
    exec(compile(source,inspect.getfile(original),'exec'),namespace)
    sd.skeleton_discovery = namespace['skeleton_discovery']
    return track,engine,original
