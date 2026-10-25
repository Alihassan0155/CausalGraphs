"""Read-only graph validation; write a separate report, never discovery artifacts."""
import argparse
import importlib.util
import json
from pathlib import Path
import discover as b

np,pd,nx=b.np,b.pd,b.nx


def validate_export(folder,names,prefix=''):
    a=np.load(folder/f'{prefix}endpoint_matrix.npy')
    n=len(names);assert a.shape==(n,n) and not np.diag(a).any()
    assert set(np.unique(a))<={-1,0,1} and np.array_equal(a!=0,(a!=0).T)
    d=np.zeros((n,n),dtype=np.uint8);u=np.zeros_like(d)
    expected=[];labels={-1:'tail',1:'arrow'}
    for i in range(n):
        for j in range(i+1,n):
            if not a[i,j]:continue
            pair=(int(a[i,j]),int(a[j,i]))
            if pair==(-1,1):d[i,j]=1;kind='directed'
            elif pair==(1,-1):d[j,i]=1;kind='directed'
            elif pair==(-1,-1):u[i,j]=u[j,i]=1;kind='undirected'
            else:raise AssertionError(f'Unsupported endpoints {pair}')
            expected.append(dict(node_i=i,node_j=j,symbol_i=names[i],symbol_j=names[j],
                endpoint_at_i=labels[pair[0]],endpoint_at_j=labels[pair[1]],edge_type=kind))
    actual=pd.read_csv(folder/f'{prefix}edges.csv')
    assert actual.to_dict('records')==expected
    skeleton=(a!=0).astype(np.uint8)
    for name,array in [('directed',d),('undirected',u),('skeleton',skeleton)]:
        saved=b.sparse.load_npz(folder/f'{prefix}{name}_adjacency.npz').toarray()
        assert np.array_equal(saved,array)
    assert np.array_equal(skeleton,d+d.T+u)
    reconstructed=np.zeros_like(a)
    for i,j in zip(*np.nonzero(d)):reconstructed[i,j]=-1;reconstructed[j,i]=1
    reconstructed[u!=0]=-1
    assert np.array_equal(a,reconstructed)
    g=nx.from_numpy_array(skeleton);degree=np.array([g.degree(i) for i in range(n)])
    stats=dict(nodes=n,edges=len(expected),directed_edges=int(d.sum()),undirected_edges=int(u.sum()//2),
        density=float(nx.density(g)),components=nx.number_connected_components(g),
        component_sizes=sorted([len(c) for c in nx.connected_components(g)],reverse=True),
        isolated_genes=[names[i] for i in range(n) if degree[i]==0],
        isolated_nodes=int((degree==0).sum()),degree_min=int(degree.min()),degree_max=int(degree.max()),
        degree_mean=float(degree.mean()),directed_part_acyclic=nx.is_directed_acyclic_graph(nx.from_numpy_array(d,create_using=nx.DiGraph)))
    saved=json.loads((folder/f'{prefix}graph_statistics.json').read_text())
    for key in ('nodes','edges','density','components','isolated_nodes','degree_min','degree_max','degree_mean','directed_part_acyclic'):
        assert saved[key]==stats[key],key
    return a,d,u,skeleton,stats


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',default='full_boss_seed20')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not args.run_id or not all(c.isalnum() or c in '_-' for c in args.run_id):parser.error('Simple run ID required')
    if args.output.exists():raise SystemExit('Output exists; choose a NEW validation directory')
    folder=b.BASE/'boss'/args.run_id
    files=list(p for p in folder.iterdir() if p.is_file())
    before={p.name:b.sha(p) for p in files}
    status=json.loads((folder/'status.json').read_text());assert status['status']=='complete' and status['exit_code']==0
    cfg=json.loads((folder/'config.json').read_text());assert cfg['algorithm']=='boss' and cfg['nodes']==279 and cfg['cells']==10691
    for key,path in [('matrix_sha256',b.MATRIX),('gene_order_sha256',b.ORDER),('control_rows_sha256',b.CONTROL_ROWS)]:assert cfg[key]==b.sha(path)
    assert cfg['versions']==b.versions() and cfg['implementation_files']==b.package_fingerprints()
    assert cfg['wrapper_sha256']==b.sha(Path(b.__file__))
    cols=np.load(folder/'selected_columns.npy');rows=np.load(folder/'selected_rows.npy')
    assert np.array_equal(cols,np.arange(279)) and np.array_equal(rows,np.arange(10691))
    mapping=pd.read_csv(b.ORDER);names=pd.read_csv(folder/'node_order.csv')
    assert names.original_node_index.tolist()==list(range(279)) and names.discovery_node_index.tolist()==list(range(279))
    for key in ('symbol','ensembl_id','expression_column','sccvae_latent_index'):
        assert names[key].tolist()==mapping[key].tolist()
    assert mapping.equals(pd.read_csv(b.ROOT/'graphs/correlation_split0/gene_order.csv'))
    source=np.load(b.CONTROL_ROWS);assert np.array_equal(source,np.load(folder/'selected_source_rows.npy'))
    cells=pd.read_csv(b.ROOT/'processed/gnn_split0/cells.csv.gz',usecols=['source_row','split','training_control'])
    assert np.array_equal(cells.iloc[source].source_row.to_numpy(),source)
    assert cells.iloc[source].training_control.all() and (cells.iloc[source]['split']=='train').all()
    x=np.load(b.MATRIX,mmap_mode='r');assert x.shape==(10691,279) and np.isfinite(x).all() and (x.std(axis=0)>0).all()
    a,d,u,s,stats=validate_export(folder,names.symbol.tolist())
    da,dd,du,ds,dagstats=validate_export(folder,names.symbol.tolist(),'representative_dag_')
    assert dagstats['directed_part_acyclic'] and not du.any() and np.array_equal(ds,s)
    # Verify author-library DAG->CPDAG conversion reproduces every saved endpoint.
    from causallearn.graph.GeneralGraph import GeneralGraph
    from causallearn.graph.GraphNode import GraphNode
    from causallearn.graph.Edge import Edge
    from causallearn.graph.Endpoint import Endpoint
    from causallearn.utils.DAG2CPDAG import dag2cpdag
    nodes=[GraphNode(name) for name in names.symbol];graph=GeneralGraph(nodes)
    for i,j in zip(*np.nonzero(dd)):graph.add_edge(Edge(nodes[i],nodes[j],Endpoint.TAIL,Endpoint.ARROW))
    assert np.array_equal(dag2cpdag(graph).graph,a)
    # A CPDAG's chain components must contain no directed internal edges or
    # semi-directed cycles; this supplements ordinary directed DAG checks.
    ug=nx.from_numpy_array(u);chain=list(nx.connected_components(ug));membership={v:i for i,c in enumerate(chain) for v in c}
    quotient=nx.DiGraph();quotient.add_nodes_from(range(len(chain)))
    for i,j in zip(*np.nonzero(d)):
        assert membership[i]!=membership[j]
        quotient.add_edge(membership[i],membership[j])
    assert nx.is_directed_acyclic_graph(quotient)
    assert all(nx.is_chordal(ug.subgraph(c)) for c in chain)
    correlation=b.sparse.load_npz(b.ROOT/'graphs/correlation_split0/adjacency.npz').toarray()
    assert correlation.shape==(279,279) and np.array_equal(correlation,correlation.T) and not np.diag(correlation).any()
    cg=nx.from_numpy_array(correlation);overlap=int(np.count_nonzero(np.triu((s!=0)&(correlation!=0),1)))
    comparison=dict(correlation_edges=cg.number_of_edges(),correlation_isolates=len(list(nx.isolates(cg))),
        correlation_components=nx.number_connected_components(cg),correlation_density=nx.density(cg),
        shared_skeleton_edges=overlap,boss_only_edges=stats['edges']-overlap,
        correlation_only_edges=cg.number_of_edges()-overlap,
        skeleton_jaccard=overlap/(stats['edges']+cg.number_of_edges()-overlap))
    # Format compatibility only: no GNN, feature engineering or training.
    import torch
    directed_index=torch.tensor(np.vstack(np.nonzero(d)),dtype=torch.long)
    unresolved_index=torch.tensor(np.vstack(np.nonzero(u)),dtype=torch.long)
    edge_index=torch.cat((directed_index,unresolved_index),dim=1)
    edge_type=torch.cat((torch.zeros(directed_index.shape[1],dtype=torch.long),torch.ones(unresolved_index.shape[1],dtype=torch.long)))
    assert edge_index.shape==(2,stats['directed_edges']+2*stats['undirected_edges'])
    assert int(edge_index.max())<279 and edge_index.dtype==torch.long
    rebuilt=np.zeros_like(a)
    for k in range(edge_index.shape[1]):
        i,j=edge_index[:,k].tolist()
        if edge_type[k]==0:rebuilt[i,j]=-1;rebuilt[j,i]=1
        else:rebuilt[i,j]=rebuilt[j,i]=-1
    assert np.array_equal(a,rebuilt)
    pyg=dict(torch_version=torch.__version__,edge_index_shape=list(edge_index.shape),
        directed_relation_arcs=directed_index.shape[1],unresolved_relation_arcs=unresolved_index.shape[1],
        endpoint_roundtrip_exact=True,explicit_num_nodes=279,pyg_installed=importlib.util.find_spec('torch_geometric') is not None)
    if pyg['pyg_installed']:
        from torch_geometric.data import Data
        data=Data(edge_index=edge_index,edge_type=edge_type,num_nodes=279)
        assert data.validate(raise_on_error=True)
        pyg['data_validate_passed']=True
    else:pyg['data_validate_passed']=None
    recorded=json.loads((folder/'statistics.json').read_text())
    assert recorded['nodes']==stats['nodes'] and recorded['edges']==stats['edges']
    result=dict(status='PASS',run=str(folder.resolve()),statistics=stats,comparison=comparison,pyg_compatibility=pyg,
        runtime=dict(algorithm_seconds=recorded['algorithm_seconds'],supervised_seconds=recorded['supervised_seconds'],
                     peak_worker_rss_bytes=recorded['peak_worker_rss_bytes']),
        parameters=dict(seed=cfg['seed'],bic_penalty=cfg['bic_penalty'],score=recorded['score'],
                        source_dtype=str(x.dtype),discovery_dtype='float64',cells=10691),
        original_artifact_sha256=before,dag_to_cpdag_exact=True,chain_graph_checks_passed=True,
        scope='Exports checked losslessly against saved endpoints and representative DAG; no serialized live BOSS object exists for independent in-memory comparison.')
    assert before=={p.name:b.sha(p) for p in files}
    args.output.mkdir(parents=True)
    b.atomic_json(args.output/'validation.json',result)
    pd.DataFrame({'symbol':stats['isolated_genes']}).to_csv(args.output/'isolated_genes.csv',index=False)
    text=f'''# Module 4B — Final BOSS graph validation

Status: PASS. Ready as graph input for a future direction-aware GNN, conditional on retaining unresolved edges as a distinct relation; no GNN implemented or trained.

Run: `{folder.relative_to(b.ROOT)}`. Successful exit 0, seed {cfg['seed']}, all 10,691 training-control cells and all 279 nodes in verified original order. Input matrix, mapping, source rows, package and runner fingerprints match. No direct test perturbation outcomes enter discovery.

## Exact results

Algorithm runtime: {recorded['algorithm_seconds']!r} seconds; supervised runtime: {recorded['supervised_seconds']!r} seconds. Peak recorded worker memory: {recorded['peak_worker_rss_bytes']:,} bytes.

| Graph | Nodes | Edges | Density | Components | Isolates |
| --- | --- | --- | --- | --- | --- |
| BOSS CPDAG | 279 | {stats['edges']} | {stats['density']:.15f} | {stats['components']} | {stats['isolated_nodes']} |
| Correlation | 279 | {cg.number_of_edges()} | {nx.density(cg):.15f} | {nx.number_connected_components(cg)} | {len(list(nx.isolates(cg)))} |

BOSS: {stats['directed_edges']} directed and {stats['undirected_edges']} undirected edges. Component sizes: {stats['component_sizes']}. Degree min/mean/max: {stats['degree_min']} / {stats['degree_mean']:.9f} / {stats['degree_max']}. Directed part acyclic; chain-component quotient acyclic; undirected chain components chordal. Shared skeleton edges: {overlap}; BOSS-only: {stats['edges']-overlap}; correlation-only: {cg.number_of_edges()-overlap}; skeleton Jaccard: {comparison['skeleton_jaccard']:.9f}. Greater connectivity is not evidence of greater causal accuracy.

## Export and PyG validation

Endpoint matrix, every edge-list row and all sparse adjacencies reconstruct exactly in both directions. No endpoint marks are lost. The saved representative DAG has the same skeleton, is acyclic, and its official DAG-to-CPDAG conversion reproduces the saved CPDAG exactly. All 279 nodes, including isolates, are retained. No live serialized BOSS object was saved, so equivalence to an independently retained in-memory object cannot be tested; checks establish consistency across all saved representations.

In-memory PyTorch format roundtrip passes: `edge_index` int64 shape {list(edge_index.shape)}, {directed_index.shape[1]} directed arcs plus {unresolved_index.shape[1]} arcs encoding 34 unresolved edges in both message-passing directions. Edge relation 0=directed, 1=unresolved; reciprocal unresolved arcs are NOT two causal arrows. Set `num_nodes=279` explicitly to retain isolates. PyG installed: {pyg['pyg_installed']}; actual PyG Data.validate: {pyg['data_validate_passed']}. No new dependency was installed. Do not drop relation labels, arbitrarily orient unresolved edges, or treat the representative DAG as unique causal truth. Incoming/reverse message relations, if later used, must be explicit and must not be mistaken for inferred causal directions.

## Paper-ready methodology and limitations

Best Order Score Search (causal-learn {cfg['versions']['causal-learn']}) used Gaussian covariance SEM-BIC (`local_score_BIC_from_cov`, lambda_value={cfg['bic_penalty']}) with seed {cfg['seed']} and {recorded['boss_order_passes']} recorded order passes. Score/parameterization: {recorded['score']}. Prepared float32 expression was cast to float64 without additional standardization. PC/Fisher-Z/alpha fields in the shared configuration are inactive for BOSS and must NOT be reported as BOSS parameters.

Inherited SCCVAE preprocessing normalizes the full file across all 8,563 genes using normalize_total followed by log1p before selecting 279 mapped graph genes. The whole-file library-size statistic is transductive preprocessing, not strictly inductive; discovery itself uses training controls only. Correlation comparison uses its saved provisional |r|>=0.30 graph, not test-tuned selection. Observational Gaussian SEM, acyclicity, causal Markov/faithfulness and causal sufficiency assumptions may fail under latent confounding, batch/cell-state variation, compositional normalization and measurement error. Directed edges are equivalence-class estimates, not interventionally proven mechanisms. Gene membership includes mapped perturbation identities, not held-out response fitting.

Original artifact hashes were checked before/after validation and remained unchanged. No discovery was rerun. Machine-readable evidence: `validation.json`; complete isolate list: `isolated_genes.csv`.
'''
    (args.output/'REPORT.md').write_text(text,encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','runtime','comparison','pyg_compatibility')},indent=2))
    print('REPORT',args.output.resolve())


if __name__=='__main__':main()
