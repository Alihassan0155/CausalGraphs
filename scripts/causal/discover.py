"""Manual PC/BOSS runner: unlimited by default, optional time budget, recoverable stages."""
from pathlib import Path
import argparse, os, sys, json, time, hashlib, pickle, random, inspect
import subprocess, platform, traceback, ast, math

ROOT=Path(__file__).resolve().parents[2]
ISOLATED=ROOT/'.venv/module4a_packages'
if not ISOLATED.exists():
    raise SystemExit('Install configs/module4a/requirements.txt into .venv/module4a_packages first.')
sys.path.insert(0,str(ISOLATED))
# Limit BLAS oversubscription before importing numeric libraries.
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np
import pandas as pd
import psutil
import networkx as nx
from scipy import sparse
import importlib.metadata

MATRIX=ROOT/'processed/gnn_split0/training_control_graph_expression.npy'
ORDER=ROOT/'processed/gnn_split0/graph_gene_mapping.csv'
CONTROL_ROWS=ROOT/'processed/gnn_split0/training_control_source_rows.npy'
BASE=ROOT/'graphs/causal_split0'

# Reviewed pre-update runner: algorithm, checkpoint and graph-export code is unchanged.
# This whitelist cannot authorize future algorithm changes: the contract must also match.
LEGACY_RECOVERY_CONTRACTS={
    'bb01f0ac83dcd53486d7310d004dd157abac32fc8d841bb1e0bb2f7b58b94370':
    '081c0aef8fa690bdc93da6576a8369bd92ff3eb29d08f1e6e5bdcf7c1921efbd'
}

def recovery_contract():
    source=Path(__file__).read_text()
    tree=ast.parse(source)
    selected='\n'.join(ast.get_source_segment(source,n) for n in tree.body
                      if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))
                      and n.name in ['install_pc_hooks','worker','graph_artifacts'])
    return hashlib.sha256(selected.encode()).hexdigest()

def compatible_configuration(saved,current):
    """Permit supervisor-only edits; require identical discovery/checkpoint semantics."""
    ignored={'wrapper_sha256','algorithm_state_contract_sha256'}
    if {k:v for k,v in saved.items() if k not in ignored}!={k:v for k,v in current.items() if k not in ignored}:
        return False
    old_contract=saved.get('algorithm_state_contract_sha256')
    if old_contract is None:
        old_contract=LEGACY_RECOVERY_CONTRACTS.get(saved.get('wrapper_sha256'))
    return old_contract==current['algorithm_state_contract_sha256']

def time_budget_exhausted(limit,elapsed):
    return limit is not None and limit>0 and elapsed>=limit

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(2**20),b''): h.update(block)
    return h.hexdigest()

def atomic_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2),encoding='utf-8'); os.replace(tmp,path)

def atomic_pickle(path,value):
    tmp=path.with_suffix('.tmp')
    with open(tmp,'wb') as f: pickle.dump(value,f,protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp,path)

def stop_worker_tree(child):
    """Windows venv launchers spawn a real interpreter: terminate both, not only the shim."""
    try:
        root=psutil.Process(child.pid)
        processes=root.children(recursive=True)+[root]
    except psutil.NoSuchProcess:
        processes=[]
    for process in reversed(processes):
        try: process.terminate()
        except psutil.NoSuchProcess: pass
    _,alive=psutil.wait_procs(processes,timeout=5)
    for process in alive:
        try: process.kill()
        except psutil.NoSuchProcess: pass
    child.wait()

def event(out,kind,**fields):
    value={'time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'event':kind,**fields}
    line=json.dumps(value)
    with open(out/'progress.log','a',encoding='utf-8') as f: f.write(line+'\n')
    print(line,flush=True)

def versions():
    return {p:importlib.metadata.version(p) for p in ['causal-learn','momentchi2','numpy','scipy','pandas','psutil','networkx']}

def package_fingerprints():
    return {str(p.relative_to(ISOLATED)):sha(p) for p in [
      ISOLATED/'causallearn/search/ConstraintBased/PC.py',
      ISOLATED/'causallearn/utils/PCUtils/SkeletonDiscovery.py',
      ISOLATED/'causallearn/search/PermutationBased/BOSS.py',
      ISOLATED/'causallearn/search/PermutationBased/gst.py',
      ISOLATED/'causallearn/score/LocalScoreFunction.py']}

def install_pc_hooks(checkpoint,report,resume=False):
    """Add I/O at complete stable-PC depth boundaries, preserving upstream search/orientation."""
    from causallearn.utils.PCUtils import SkeletonDiscovery as sd
    original=sd.skeleton_discovery
    source=inspect.getsource(original)
    anchors=['    depth = -1','    if show_progress:\n        pbar.close()']
    assert all(source.count(s)==1 for s in anchors), 'Upstream instrumentation anchors changed'
    initial=None
    if resume:
        # Only load our own local checkpoint after configuration/fingerprint verification.
        with open(checkpoint,'rb') as f: initial=pickle.load(f)
    tracking={'last_completed_depth':-1}
    def restore(cg,depth):
        if initial is None: return cg,depth
        tracking['last_completed_depth']=initial['completed_depth']
        report('pc_state_resumed',depth=initial['completed_depth'])
        return initial['cg'],initial['completed_depth']
    def save(cg,depth):
        atomic_pickle(checkpoint,{'cg':cg,'completed_depth':depth})
        tracking['last_completed_depth']=depth
        report('pc_depth_complete',depth=depth,max_degree=int(cg.max_degree()),
               ci_cache_entries=len(cg.test.pvalue_cache),skeleton_edges=int(np.count_nonzero(cg.G.graph)//2))
    source=source.replace(anchors[0],anchors[0]+'\n    cg, depth = _module4a_restore(cg, depth)')
    source=source.replace(anchors[1],'        _module4a_save(cg, depth)\n\n'+anchors[1])
    namespace=dict(sd.__dict__)
    namespace.update(_module4a_restore=restore,_module4a_save=save)
    exec(compile(source,inspect.getfile(original),'exec'),namespace)
    sd.skeleton_discovery=namespace['skeleton_discovery']
    return tracking,original

def graph_artifacts(out,graph,nodes,prefix=''):
    """Endpoint matrix retains equivalence-class ambiguity; never orient undirected edges arbitrarily."""
    a=np.asarray(graph.graph)
    p=len(nodes)
    assert a.shape==(p,p) and not np.any(np.diag(a))
    assert np.array_equal(a!=0,(a!=0).T), 'Asymmetric edge existence'
    assert [n.get_name() for n in graph.get_nodes()]==nodes.symbol.tolist()
    endpoint_names={-1:'tail',0:'null',1:'arrow',2:'circle',3:'star',4:'tail_and_arrow',5:'arrow_and_arrow'}
    assert set(np.unique(a))<=set(endpoint_names)
    directed=np.zeros((p,p),dtype=np.uint8); undirected=np.zeros_like(directed)
    edges=[]; kinds={}
    for i in range(p):
        for j in range(i+1,p):
            if not a[i,j]: continue
            x,y=int(a[i,j]),int(a[j,i])
            if (x,y)==(-1,1): kind='directed'; directed[i,j]=1
            elif (x,y)==(1,-1): kind='directed'; directed[j,i]=1
            elif (x,y)==(-1,-1): kind='undirected'; undirected[i,j]=undirected[j,i]=1
            elif (x,y)==(1,1): kind='bidirected_conflict'
            elif 2 in (x,y): kind='partially_oriented'
            else: kind='other_endpoint_pair'
            kinds[kind]=kinds.get(kind,0)+1
            edges.append({'node_i':i,'node_j':j,'symbol_i':nodes.symbol.iloc[i],'symbol_j':nodes.symbol.iloc[j],
              'endpoint_at_i':endpoint_names[x],'endpoint_at_j':endpoint_names[y],'edge_type':kind})
    skeleton=(a!=0).astype(np.uint8)
    np.save(out/f'{prefix}endpoint_matrix.npy',a)
    sparse.save_npz(out/f'{prefix}skeleton_adjacency.npz',sparse.csr_matrix(skeleton))
    sparse.save_npz(out/f'{prefix}directed_adjacency.npz',sparse.csr_matrix(directed))
    sparse.save_npz(out/f'{prefix}undirected_adjacency.npz',sparse.csr_matrix(undirected))
    pd.DataFrame(edges,columns=['node_i','node_j','symbol_i','symbol_j','endpoint_at_i','endpoint_at_j','edge_type']).to_csv(out/f'{prefix}edges.csv',index=False)
    d=skeleton.sum(axis=1)
    components=list(nx.connected_components(nx.from_numpy_array(skeleton)))
    stats={'nodes':p,'edges':len(edges),'density':float(2*len(edges)/(p*(p-1))) if p>1 else 0.,
      'edge_types':kinds,'isolated_nodes':int((d==0).sum()),'components':len(components),
      'degree_min':int(d.min()),'degree_max':int(d.max()),'degree_mean':float(d.mean()),
      'directed_part_acyclic':nx.is_directed_acyclic_graph(nx.from_numpy_array(directed,create_using=nx.DiGraph)),
      'endpoint_encoding':'matrix[i,j] is endpoint at i; -1 tail, +1 arrow, 2 circle. Directed adjacency[i,j]=1 means i -> j.',
      'orientation_conflicts':kinds.get('bidirected_conflict',0)+kinds.get('other_endpoint_pair',0)}
    atomic_json(out/f'{prefix}graph_statistics.json',stats)
    return stats

def worker(out,mode):
    cfg=json.loads((out/'config.json').read_text())
    seed=cfg['seed']; np.random.seed(seed); random.seed(seed)
    x=np.asarray(np.load(MATRIX,mmap_mode='r')[np.ix_(np.load(out/'selected_rows.npy'),np.load(out/'selected_columns.npy'))],dtype=np.float64)
    assert np.isfinite(x).all() and (x.std(axis=0)>0).all()
    names=pd.read_csv(out/'node_order.csv')
    begin=time.monotonic()
    report=lambda kind,**kw:event(out,kind,algorithm_seconds=time.monotonic()-begin,**kw)
    try:
        if cfg['algorithm']=='pc':
            from causallearn.search.ConstraintBased.PC import pc
            track,_=install_pc_hooks(out/'pc_depth_checkpoint.pkl',report,resume=(mode=='resume'))
            result=pc(x,alpha=cfg['alpha'],indep_test='fisherz',stable=True,uc_rule=0,uc_priority=2,
                      show_progress=True,node_names=names.symbol.tolist(),max_k=cfg['max_depth'])
            graph=result.G
            depth=track['last_completed_depth']
            # A further skeleton depth remains eligible only under the original loop condition.
            with open(out/'pc_depth_checkpoint.pkl','rb') as f: skeleton_state=pickle.load(f)
            remaining=skeleton_state['cg'].max_degree()-1>depth
            truncated=bool(cfg['max_depth'] is not None and remaining and depth>=cfg['max_depth'])
            details={'last_completed_conditioning_depth':depth,'conditioning_truncated':truncated,
                     'graph_kind':'depth-truncated PC PDAG' if truncated else 'PC CPDAG candidate',
                     'resume_semantics':'True stable-PC skeleton recovery from last completed conditioning depth; interrupted depth is repeated; orientation is restarted.'}
        else:
            import importlib
            boss_module=importlib.import_module('causallearn.search.PermutationBased.BOSS')
            original_mutation=boss_module.better_mutation
            calls=[0]
            def mutation(v,order,gsts):
                improved=original_mutation(v,order,gsts); calls[0]+=1
                if calls[0]%x.shape[1]==0: report('boss_pass_complete',passes=calls[0]//x.shape[1])
                return improved
            boss_module.better_mutation=mutation
            # Retain upstream's representative DAG before its built-in CPDAG conversion.
            source=inspect.getsource(boss_module.boss)
            anchor='    G = dag2cpdag(G)'
            assert source.count(anchor)==1
            captured=[]
            def capture(g): captured.append(g)
            namespace=dict(boss_module.__dict__); namespace['_module4a_capture']=capture
            exec(compile(source.replace(anchor,'    _module4a_capture(G)\n'+anchor),inspect.getfile(boss_module.boss),'exec'),namespace)
            graph=namespace['boss'](x,score_func='local_score_BIC_from_cov',parameters={'lambda_value':cfg['bic_penalty']},verbose=True,node_names=names.symbol.tolist())
            dagstats=graph_artifacts(out,captured[0],names,prefix='representative_dag_')
            assert dagstats['directed_part_acyclic'] and dagstats['edge_types'].get('undirected',0)==0
            details={'graph_kind':'BOSS CPDAG','boss_order_passes':calls[0]//x.shape[1],
              'conditioning_truncated':False,'resume_semantics':'BOSS has no serialized GST/order-search recovery here. Restart algorithm from saved input subset and seed.',
              'score':'Gaussian SEM penalized log likelihood: -n/2*(1+log(residual_variance)) - lambda*(parent_count+1)*log(n), maximize; ddof=0 covariance',
              'bic_penalty':cfg['bic_penalty'],'representative_dag_is_unique_causal_truth':False}
        stats=graph_artifacts(out,graph,names)
        details.update(stats); details['algorithm_seconds']=time.monotonic()-begin
        mem=psutil.Process().memory_info()
        details['worker_peak_rss_bytes']=max(mem.rss,getattr(mem,'peak_wset',0))
        atomic_json(out/'algorithm_result.json',details)
        report('algorithm_complete',**{'result_seconds':details['algorithm_seconds']})
    except BaseException:
        (out/'worker_error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithm',choices=['pc','boss'],required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--nodes',type=int,default=279)
    parser.add_argument('--cells',type=int,default=10691)
    parser.add_argument('--seed',type=int,default=20)
    parser.add_argument('--alpha',type=float,default=.05)
    parser.add_argument('--max-depth',type=int,default=-1,help='PC: -1 unrestricted; >=0 caps conditioning-set size and reports truncation')
    parser.add_argument('--bic-penalty',type=float,default=2.,help='BOSS covariance SEM-BIC lambda_value (upstream BOSS default 2; conventional BIC in this parameterization is 0.5)')
    parser.add_argument('--time-limit-seconds',type=float,default=None,
                        help='Optional wall-time limit. Omit for unlimited execution; 0 also means unlimited.')
    group=parser.add_mutually_exclusive_group(); group.add_argument('--resume',action='store_true'); group.add_argument('--restart-stage',action='store_true')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.time_limit_seconds is not None and (not math.isfinite(args.time_limit_seconds) or args.time_limit_seconds<0):
        parser.error('--time-limit-seconds must be finite and nonnegative; omit it for no timeout')
    assert args.run_id and all(c.isalnum() or c in '_-' for c in args.run_id), 'Use a simple run ID'
    assert 2<=args.nodes<=279 and 4<=args.cells<=10691 and 0<args.alpha<1 and args.max_depth>=-1 and args.bic_penalty>0
    if args.algorithm=='boss' and args.max_depth!=-1: parser.error('--max-depth is PC-only')
    if args.algorithm=='boss' and args.resume: parser.error('BOSS --resume unsupported; use --restart-stage to explicitly restart the search')
    out=BASE/args.algorithm/args.run_id
    mode='resume' if args.resume else 'restart' if args.restart_stage else 'new'
    if args.worker:
        worker(out,mode); return
    cfg={'algorithm':args.algorithm,'nodes':args.nodes,'cells':args.cells,'seed':args.seed,
      'alpha':args.alpha,'stable':True,'independence_test':'fisherz','uc_rule':0,'uc_priority':2,
      'max_depth':None if args.max_depth==-1 else args.max_depth,'bic_penalty':args.bic_penalty,
      'matrix_sha256':sha(MATRIX),'gene_order_sha256':sha(ORDER),'control_rows_sha256':sha(CONTROL_ROWS),
      'wrapper_sha256':sha(Path(__file__)),'algorithm_state_contract_sha256':recovery_contract(),
      'versions':versions(),'implementation_files':package_fingerprints(),
      'input_dtype':'float64 cast from prepared float32','extra_standardization':False,
      'subset_policy':'Seeded uniform subsets of rows/columns without replacement; sort selected indices to preserve original order.'}
    assert cfg['versions']['causal-learn']=='0.1.4.8'
    prior={}
    if out.exists():
        if (out/'status.json').exists(): prior=json.loads((out/'status.json').read_text())
        if prior.get('status')=='complete': raise SystemExit('Completed experiment protected. Choose a NEW run ID; no overwrite permitted.')
        if not (args.resume or args.restart_stage): raise SystemExit('Run exists. Use --resume for PC state recovery or --restart-stage explicitly.')
        if prior.get('status')=='running' and psutil.pid_exists(prior.get('supervisor_pid',-1)):
            proc=psutil.Process(prior['supervisor_pid'])
            if abs(proc.create_time()-prior.get('supervisor_created',0))<1: raise SystemExit('Run is still active; refusing a concurrent writer.')
        assert compatible_configuration(json.loads((out/'config.json').read_text()),cfg), 'Algorithm settings, recovery semantics, versions or input fingerprints changed; use a NEW run ID'
        if args.resume and not (out/'pc_depth_checkpoint.pkl').exists(): raise SystemExit('No completed-depth PC checkpoint; use --restart-stage')
    else:
        if args.resume or args.restart_stage: raise SystemExit('Run does not exist; omit recovery flags')
        out.mkdir(parents=True)
        x=np.load(MATRIX,mmap_mode='r'); nodes=pd.read_csv(ORDER)
        assert x.shape==(10691,279) and len(nodes)==279 and np.array_equal(nodes.node_index,np.arange(279))
        assert nodes.equals(pd.read_csv(ROOT/'graphs/correlation_split0/gene_order.csv'))
        rowids=np.load(CONTROL_ROWS)
        cells=pd.read_csv(ROOT/'processed/gnn_split0/cells.csv.gz',usecols=['source_row','split','training_control'])
        assert (cells.iloc[rowids]['split']=='train').all() and cells.iloc[rowids].training_control.all()
        rng=np.random.default_rng(args.seed)
        columns=np.sort(rng.choice(279,args.nodes,replace=False)) if args.nodes<279 else np.arange(279)
        rows=np.sort(rng.choice(10691,args.cells,replace=False)) if args.cells<10691 else np.arange(10691)
        np.save(out/'selected_columns.npy',columns); np.save(out/'selected_rows.npy',rows)
        np.save(out/'selected_source_rows.npy',rowids[rows])
        chosen=nodes.iloc[columns].copy(); chosen=chosen.rename(columns={'node_index':'original_node_index'})
        chosen.insert(0,'discovery_node_index',np.arange(len(chosen))); chosen.to_csv(out/'node_order.csv',index=False)
        atomic_json(out/'config.json',cfg)
        atomic_json(out/'provenance.json',{'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'python':sys.version,'platform':platform.platform(),'logical_cpus':psutil.cpu_count(),
          'ram_bytes':psutil.virtual_memory().total,'command':sys.argv,'software_versions':cfg['versions'],
          'normalization_limitation':'Full-file median normalization includes held-out library-size statistics. Only training-control rows enter this discovery; no direct test-response fitting. Not strictly inductive.',
          'assumptions':'Observational linear-Gaussian/partial-correlation or Gaussian-SEM scoring assumptions, causal Markov/faithfulness, acyclicity and causal sufficiency; hidden confounding, compositional effects, cell state/batch dependence and measurement error may violate them.',
          'orientation_note':'Undirected CPDAG edges are unresolved directions; bidirected PC conflict edges are not automatic evidence of latent confounding.'})
    attempt=int(prior.get('attempt',0))+1
    status={'status':'running','attempt':attempt,'recovery_mode':mode,'supervisor_pid':os.getpid(),
      'supervisor_created':psutil.Process().create_time(),'time_limit_seconds':args.time_limit_seconds,
      'timeout_enabled':args.time_limit_seconds is not None and args.time_limit_seconds>0,
      'runner_sha256':cfg['wrapper_sha256'],'algorithm_state_contract_sha256':cfg['algorithm_state_contract_sha256']}
    atomic_json(out/'status.json',status)
    begin=time.monotonic(); peak=0; child=None; state='failed'; returncode=None
    old_elapsed=float(prior.get('cumulative_supervised_seconds',0.))
    event(out,'attempt_started',attempt=attempt,mode=mode)
    if mode=='restart': event(out,'algorithm_restarting',checkpoint_reused=False)
    command=[sys.executable,'-u',str(Path(__file__).resolve()),*sys.argv[1:],'--worker']
    try:
        with open(out/'worker.log','a',encoding='utf-8') as stream:
            stream.write(f'\nATTEMPT {attempt} {mode}\n'); stream.flush()
            child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            status['worker_pid']=child.pid; atomic_json(out/'status.json',status)
            while child.poll() is None:
                elapsed=time.monotonic()-begin
                try:
                    root=psutil.Process(child.pid)
                    memories=[p.memory_info() for p in [root]+root.children(recursive=True)]
                    peak=max(peak,sum(m.rss for m in memories),sum(getattr(m,'peak_wset',m.rss) for m in memories))
                except psutil.NoSuchProcess: pass
                event(out,'heartbeat',attempt=attempt,elapsed_seconds=elapsed,peak_worker_rss_bytes=peak)
                if time_budget_exhausted(args.time_limit_seconds,elapsed):
                    state='timed_out'; stop_worker_tree(child); break
                time.sleep(2)
            child.wait(); returncode=child.returncode
            if state!='timed_out': state='complete' if returncode==0 else 'failed'
    except KeyboardInterrupt:
        state='interrupted'
        if child and child.poll() is None: stop_worker_tree(child)
    finally:
        elapsed=time.monotonic()-begin
        status.update(status=state,exit_code=returncode,attempt_seconds=elapsed,
                      cumulative_supervised_seconds=old_elapsed+elapsed,peak_worker_rss_bytes=peak)
        atomic_json(out/'status.json',status)
        if state=='complete':
            details=json.loads((out/'algorithm_result.json').read_text())
            details.update(supervised_seconds=elapsed,peak_worker_rss_bytes=max(peak,details['worker_peak_rss_bytes']))
            atomic_json(out/'statistics.json',details)
        event(out,'attempt_finished',**status)
    if state!='complete': raise SystemExit(2)

if __name__=='__main__': main()
