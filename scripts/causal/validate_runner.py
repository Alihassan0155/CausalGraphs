"""Small synthetic checks for upstream equivalence, endpoint encoding and true PC recovery."""
import tempfile,pickle
from pathlib import Path
import numpy as np
import pandas as pd
import discover as runner
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.PCUtils import SkeletonDiscovery as sd
from causallearn.search.PermutationBased.BOSS import boss
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint

rng=np.random.default_rng(20)
x=rng.normal(size=(600,8))
x[:,2]=x[:,0]+x[:,1]+.3*x[:,2]
x[:,3]=.7*x[:,2]+x[:,3]
names=pd.DataFrame({'symbol':[f'G{i}' for i in range(8)]})
with tempfile.TemporaryDirectory() as temp:
    folder=Path(temp); path=folder/'checkpoint.pkl'
    native=pc(x,stable=True,indep_test='fisherz',show_progress=False,node_names=names.symbol.tolist())
    track,original=runner.install_pc_hooks(path,lambda *a,**k:None)
    instrumented=pc(x,stable=True,indep_test='fisherz',show_progress=False,node_names=names.symbol.tolist())
    assert np.array_equal(native.G.graph,instrumented.G.graph)
    sd.skeleton_discovery=original
    class StopDepth(Exception): pass
    def interrupt(event,**kw):
        if kw.get('depth')==0: raise StopDepth()
    _,original=runner.install_pc_hooks(path,interrupt)
    try:
        pc(x,stable=True,show_progress=False,node_names=names.symbol.tolist())
        raise AssertionError('Expected an interruption')
    except StopDepth: pass
    sd.skeleton_discovery=original
    with open(path,'rb') as f: assert pickle.load(f)['completed_depth']==0
    _,original=runner.install_pc_hooks(path,lambda *a,**k:None,resume=True)
    resumed=pc(x,stable=True,show_progress=False,node_names=names.symbol.tolist())
    sd.skeleton_discovery=original
    assert np.array_equal(native.G.graph,resumed.G.graph)
    stats=runner.graph_artifacts(folder,resumed.G,names)
    assert stats['nodes']==8 and stats['directed_part_acyclic']
    # Explicit endpoint fixtures catch the common tail/arrow matrix reversal bug.
    gn=[GraphNode(s) for s in ['A','B','C','D']]; graph=GeneralGraph(gn)
    graph.add_edge(Edge(gn[0],gn[1],Endpoint.TAIL,Endpoint.ARROW))
    graph.add_edge(Edge(gn[1],gn[2],Endpoint.TAIL,Endpoint.TAIL))
    graph.add_edge(Edge(gn[2],gn[3],Endpoint.CIRCLE,Endpoint.ARROW))
    stats=runner.graph_artifacts(folder,graph,pd.DataFrame({'symbol':['A','B','C','D']}),prefix='fixture_')
    assert stats['edge_types']=={'directed':1,'undirected':1,'partially_oriented':1}
    assert runner.sparse.load_npz(folder/'fixture_directed_adjacency.npz')[0,1]==1
print('PASS: instrumented PC equals upstream, interrupted PC resumes exactly, endpoint types preserved.')
