"""Plot saved BOSS CPDAG endpoints without changing discovery artifacts."""
import argparse
from pathlib import Path
import discover as base
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id',default='full_boss_seed20')
    args=p.parse_args()
    if not args.run_id or not all(c.isalnum() or c in '_-' for c in args.run_id):
        p.error('Use a simple run ID')
    folder=base.BASE/'boss'/args.run_id
    import json
    assert json.loads((folder/'status.json').read_text())['status']=='complete'
    a=base.np.load(folder/'endpoint_matrix.npy')
    nodes=base.pd.read_csv(folder/'node_order.csv')
    assert a.shape==(len(nodes),len(nodes))
    graph=base.nx.from_numpy_array((a!=0).astype(int))
    positions=base.nx.spring_layout(graph,seed=20,iterations=150)
    degree=dict(graph.degree())
    directed=[];undirected=[]
    for i,j in graph.edges():
        if (a[i,j],a[j,i])==(-1,1):directed.append((i,j))
        elif (a[i,j],a[j,i])==(1,-1):directed.append((j,i))
        elif (a[i,j],a[j,i])==(-1,-1):undirected.append((i,j))
        else:raise ValueError('Unexpected endpoint pair; refusing misleading visualization')
    out=folder/'visualizations'
    out.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(18,15))
    base.nx.draw_networkx_nodes(graph,positions,ax=ax,node_size=[22+5*degree[i] for i in graph],
        node_color=[degree[i] for i in graph],cmap='viridis',alpha=.9)
    base.nx.draw_networkx_edges(graph,positions,ax=ax,edgelist=directed,arrows=True,
        arrowstyle='-|>',arrowsize=7,width=.55,alpha=.35,edge_color='#466a91',node_size=30)
    base.nx.draw_networkx_edges(graph,positions,ax=ax,edgelist=undirected,arrows=False,
        width=1.1,alpha=.8,edge_color='#e18a27',style='dashed')
    hubs=sorted(degree,key=lambda n:(-degree[n],n))[:20]
    base.nx.draw_networkx_labels(graph,positions,labels={i:nodes.symbol.iloc[i] for i in hubs},font_size=8,ax=ax)
    ax.set_title(f'BOSS CPDAG — {len(nodes)} genes, {len(directed)} directed + {len(undirected)} undirected edges\n'
                 'Labels: top 20 degree hubs; directions are observational equivalence-class estimates')
    ax.legend(handles=[Line2D([0],[0],color='#466a91',label='Directed (arrowheads)'),
                       Line2D([0],[0],color='#e18a27',linestyle='--',label='Undirected / unresolved')],loc='lower left')
    ax.axis('off');fig.tight_layout()
    for extension in ('png','svg'):
        target=out/f'boss_cpdag.{extension}'
        if target.exists():raise FileExistsError(f'Visualization exists: {target}')
        fig.savefig(target,dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,7))
    hubs=sorted(degree,key=lambda n:(-degree[n],n))[:20][::-1]
    ax.barh([nodes.symbol.iloc[n] for n in hubs],[degree[n] for n in hubs],color='#466a91')
    ax.set_xlabel('Total CPDAG degree (directed + undirected)');ax.set_title('BOSS: top 20 connected genes')
    fig.tight_layout();fig.savefig(out/'top_degree_genes.png',dpi=160);plt.close(fig)
    print(out.resolve())


if __name__=='__main__':main()
