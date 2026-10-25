"""Publication figures from immutable saved BOSS CPDAG; no discovery or edge filtering."""
import argparse
import json
from pathlib import Path
import discover as b
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as effects
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from validate_boss_final import validate_export

np,nx,pd=b.np,b.nx,b.pd
BLUE='#47789a';ORANGE='#c98136';GRAY='#b4c6cf';HUB='#284a60'


def save(fig,out,name):
    for ext in ('png','pdf','svg'):
        fig.savefig(out/f'{name}.{ext}',dpi=400,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise SystemExit('Choose a NEW output directory')
    folder=b.BASE/'boss/full_boss_seed20'
    before={p.name:b.sha(p) for p in folder.iterdir() if p.is_file()}
    assert json.loads((folder/'status.json').read_text())['status']=='complete'
    names=pd.read_csv(folder/'node_order.csv').symbol.tolist()
    a,d,u,s,stats=validate_export(folder,names)
    graph=nx.from_numpy_array(s)
    components=sorted(nx.connected_components(graph),key=lambda c:(-len(c),min(c)))
    largest=graph.subgraph(components[0]).copy()
    degree=dict(graph.degree())
    hubs=sorted(largest,key=lambda n:(-degree[n],names[n]))[:10]
    # Kamada-Kawai spreads only the connected component; a seeded spring
    # initialization makes the layout reproducible without external installations.
    initial=nx.spring_layout(largest,seed=20,iterations=300,k=2/np.sqrt(len(largest)))
    pos=nx.kamada_kawai_layout(largest,pos=initial,scale=1)
    directed=[(int(i),int(j)) for i,j in zip(*np.nonzero(d)) if i in largest and j in largest]
    unresolved=[(int(i),int(j)) for i,j in zip(*np.nonzero(np.triu(u,1))) if i in largest and j in largest]
    assert len(directed)+len(unresolved)==stats['edges']
    args.output.mkdir(parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(18,16))
    node_order=list(largest)
    sizes=[42+2.6*degree[n] if n not in hubs else 120+3*degree[n] for n in node_order]
    # Draw unresolved marks without arrowheads; directed arrows stop outside
    # the ACTUAL target-node radius, avoiding arrowheads hidden under node disks.
    nx.draw_networkx_edges(largest,pos,edgelist=directed,ax=ax,arrows=True,
        arrowstyle='-|>',arrowsize=12,width=.8,alpha=.60,edge_color=BLUE,
        node_size=sizes,nodelist=node_order,min_source_margin=1,min_target_margin=2,
        connectionstyle='arc3,rad=0.035')
    nx.draw_networkx_edges(largest,pos,edgelist=unresolved,ax=ax,arrows=False,
        width=1.5,alpha=.95,edge_color=ORANGE,style=(0,(4,3)))
    nx.draw_networkx_nodes(largest,pos,ax=ax,nodelist=node_order,node_size=sizes,
        node_color=[HUB if n in hubs else GRAY for n in node_order],edgecolors='white',linewidths=.65)
    # Place only ten labels outside the network, with non-overlapping callouts
    # on each side. Leader lines associate each label with its hub unambiguously.
    for sign in (-1,1):
        group=sorted([n for n in hubs if (pos[n][0]>=0)==(sign==1)],key=lambda n:pos[n][1])
        yvalues=np.linspace(-.83,.83,len(group)) if len(group)>1 else [0]
        for n,y in zip(group,yvalues):
            ax.annotate(names[n],xy=pos[n],xytext=(sign*1.25,y),fontsize=11,fontweight='bold',
                color=HUB,ha='left' if sign==1 else 'right',va='center',
                bbox=dict(boxstyle='round,pad=.22',fc='white',ec='none',alpha=.95),
                arrowprops=dict(arrowstyle='-',color='#83949e',lw=.7,connectionstyle='arc3,rad=0.05'))
    ax.set_xlim(-1.57,1.57);ax.set_ylim(-1.14,1.14);ax.set_aspect('equal');ax.axis('off')
    ax.set_title('BOSS CPDAG | largest connected component',loc='left',fontsize=19,fontweight='bold',pad=24)
    ax.text(0,1.015,f'{len(largest)} genes · {len(directed)} directed edges · {len(unresolved)} unresolved edges',
            transform=ax.transAxes,color='#52616b',fontsize=12)
    legend=[Line2D([0],[0],color=BLUE,lw=1.5,marker='>',markevery=[1],label='Directed (solid blue, arrowhead)'),
            Line2D([0],[0],color=ORANGE,lw=1.5,linestyle='--',label='Unresolved (dashed orange; no direction)'),
            Line2D([0],[0],marker='o',color='none',markerfacecolor=HUB,markersize=9,label='Top 10 hubs (total degree)')]
    ax.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,-.025),frameon=False,ncol=1)
    fig.text(.12,.035,f'{stats["isolated_nodes"]} isolated genes excluded from this panel only. All 766 edges retained. '
             'Node area reflects degree; layout has no causal meaning.',fontsize=10,color='#52616b')
    save(fig,args.output,'boss_largest_component')

    fig,axes=plt.subplots(2,2,figsize=(13,9),gridspec_kw={'hspace':.48,'wspace':.35})
    ax=axes[0,0]
    ax.barh(['Isolated genes','Largest component'],[stats['isolated_nodes'],len(largest)],color=[GRAY,BLUE])
    ax.set_xlim(0,279);ax.set_xlabel('Genes (279 total)');ax.set_title('A  Connectivity',loc='left',fontweight='bold')
    for i,v in enumerate([stats['isolated_nodes'],len(largest)]):ax.text(v+3,i,str(v),va='center')
    ax=axes[0,1]
    counts={size:sum(len(c)==size for c in components) for size in sorted(set(map(len,components)))}
    ax.bar([str(v) for v in counts],list(counts.values()),color=[GRAY,BLUE])
    ax.set_xlabel('Component size (genes)');ax.set_ylabel('Number of components')
    ax.set_title('B  Component-size distribution',loc='left',fontweight='bold')
    for i,v in enumerate(counts.values()):ax.text(i,v+.7,str(v),ha='center')
    ax.set_ylim(0,max(counts.values())*1.15)
    ax=axes[1,0]
    values=np.array(list(degree.values()))
    ax.hist(values,bins=np.arange(-.5,values.max()+1.5,2),color=BLUE,edgecolor='white')
    ax.set_xlabel('Total degree');ax.set_ylabel('Genes');ax.set_title('C  Degree distribution (all genes)',loc='left',fontweight='bold')
    ax=axes[1,1]
    ax.barh(['Unresolved','Directed'],[len(unresolved),len(directed)],color=[ORANGE,BLUE])
    ax.set_xlabel('Edges (766 total)');ax.set_xlim(0,800)
    ax.set_title('D  Endpoint types',loc='left',fontweight='bold')
    for i,v in enumerate([len(unresolved),len(directed)]):ax.text(v+7,i,str(v),va='center')
    fig.suptitle('BOSS graph overview | 279 genes, 69 components',fontsize=17,fontweight='bold')
    fig.text(.10,.025,'Undirected edges denote unresolved orientations, not reciprocal causal arrows. Observational estimates; not proven mechanisms.',fontsize=9,color='#52616b')
    save(fig,args.output,'boss_connectivity_overview')
    pd.DataFrame([dict(node_index=n,symbol=names[n],degree=degree[n],x=float(pos[n][0]),y=float(pos[n][1]),
                           highlighted=n in hubs) for n in node_order]).to_csv(args.output/'layout_and_hubs.csv',index=False)
    assert before=={p.name:b.sha(p) for p in folder.iterdir() if p.is_file()}
    b.atomic_json(args.output/'figure_metadata.json',dict(statistics=stats,largest_component_nodes=len(largest),
        directed_edges_shown=len(directed),unresolved_edges_shown=len(unresolved),
        hubs=[dict(symbol=names[n],degree=degree[n]) for n in hubs],
        layout='Kamada-Kawai with seeded spring initialization, seed 20; Graphviz unavailable on PATH',
        source_sha256=before,original_files_unchanged=True,script_sha256=b.sha(Path(__file__))))
    print(json.dumps(dict(output=str(args.output.resolve()),nodes=len(largest),directed=len(directed),
        unresolved=len(unresolved),isolates=stats['isolated_nodes'],hubs=[names[n] for n in hubs]),indent=2))


if __name__=='__main__':main()
