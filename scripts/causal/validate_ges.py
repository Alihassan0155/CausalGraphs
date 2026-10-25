"""Validate GES exports/CPDAG, PyTorch format and matched-node BOSS comparison."""
import argparse
import importlib.util
import json
import pickle
from pathlib import Path
import discover as b
from discover_pc_batched import validate_inputs
from validate_boss_final import validate_export
np,nx=b.np,b.nx


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',required=True);args=p.parse_args()
    if not args.run_id or not all(c.isalnum() or c in '_-' for c in args.run_id):p.error('Simple run ID required')
    folder=b.BASE/'ges'/args.run_id
    assert json.loads((folder/'status.json').read_text())['status']=='complete'
    x,names=validate_inputs(folder)
    a,d,u,s,stats=validate_export(folder,names.symbol.tolist())
    from causallearn.utils.PDAG2DAG import pdag2dag
    from causallearn.utils.DAG2CPDAG import dag2cpdag
    with open(folder/'ges_record.pkl','rb') as stream:record=pickle.load(stream)
    assert np.array_equal(record['G'].graph,a)
    assert np.array_equal(dag2cpdag(pdag2dag(record['G'])).graph,a)
    assert stats['directed_part_acyclic']
    undirected=nx.from_numpy_array(u);components=list(nx.connected_components(undirected));membership={n:i for i,c in enumerate(components) for n in c}
    quotient=nx.DiGraph();quotient.add_nodes_from(range(len(components)))
    for i,j in zip(*np.nonzero(d)):
        assert membership[i]!=membership[j];quotient.add_edge(membership[i],membership[j])
    assert nx.is_directed_acyclic_graph(quotient) and all(nx.is_chordal(undirected.subgraph(c)) for c in components)
    import torch
    edge_index=torch.tensor(np.concatenate([np.vstack(np.nonzero(d)),np.vstack(np.nonzero(u))],axis=1),dtype=torch.long)
    relation=torch.cat([torch.zeros(int(d.sum()),dtype=torch.long),torch.ones(int(u.sum()),dtype=torch.long)])
    rebuilt=np.zeros_like(a)
    for k in range(edge_index.shape[1]):
        i,j=edge_index[:,k].tolist();rebuilt[i,j]=-1;rebuilt[j,i]=1 if relation[k]==0 else -1
    assert np.array_equal(rebuilt,a)
    pyg=dict(edge_index_shape=list(edge_index.shape),num_nodes=len(names),endpoint_roundtrip_exact=True,
             installed=importlib.util.find_spec('torch_geometric') is not None,data_validate=None)
    if pyg['installed']:
        from torch_geometric.data import Data
        pyg['data_validate']=bool(Data(edge_index=edge_index,edge_type=relation,num_nodes=len(names)).validate(raise_on_error=True))
    bossfolder=b.BASE/'boss/full_boss_seed20';cols=np.load(folder/'selected_columns.npy')
    ba=np.load(bossfolder/'endpoint_matrix.npy')[np.ix_(cols,cols)]
    shared=(s!=0)&(ba!=0);overlap=int(np.triu(shared,1).sum());same=reversed_count=resolved=unresolved=0
    for i,j in zip(*np.nonzero(np.triu(shared,1))):
        pair=tuple(a[[i,j],[j,i]]);bp=tuple(ba[[i,j],[j,i]])
        if pair==bp:same+=1
        elif pair==(-1,-1):unresolved+=1
        elif bp==(-1,-1):resolved+=1
        else:reversed_count+=1
    bossedges=int(np.count_nonzero(ba)//2)
    comparison=dict(scope='Full BOSS vs full GES' if len(cols)==279 else 'GES subset vs induced full-BOSS subgraph; NOT matched discovery runs',
        boss_edges=bossedges,boss_density=bossedges/(len(cols)*(len(cols)-1)/2),boss_isolates=int(((ba!=0).sum(axis=1)==0).sum()),
        ges_edges=stats['edges'],ges_density=stats['density'],ges_isolates=stats['isolated_nodes'],shared_skeleton_edges=overlap,
        skeleton_jaccard=overlap/(stats['edges']+bossedges-overlap) if stats['edges']+bossedges else 1,
        shared_same_endpoints=same,shared_reversed_directions=reversed_count,
        ges_directed_boss_unresolved=resolved,ges_unresolved_boss_directed=unresolved)
    result=dict(status='PASS',statistics=stats,pyg=pyg,boss_comparison=comparison,cpdag_roundtrip_exact=True)
    path=folder/'validation.json'
    if path.exists():raise SystemExit('Validation exists; protected from overwrite')
    b.atomic_json(path,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
