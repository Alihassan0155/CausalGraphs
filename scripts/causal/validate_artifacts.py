"""Validate saved small graphs and compare the BOSS wrapper to native upstream BOSS."""
from pathlib import Path
import json,random
import numpy as np
import pandas as pd
import discover as runner
from causallearn.search.PermutationBased.BOSS import boss

ROOT=runner.ROOT; BASE=runner.BASE
checks=[]
for folder in BASE.glob('*/*'):
    if not (folder/'statistics.json').exists(): continue
    cfg=json.loads((folder/'config.json').read_text())
    stats=json.loads((folder/'statistics.json').read_text())
    order=pd.read_csv(folder/'node_order.csv')
    original=pd.read_csv(runner.ORDER).iloc[np.load(folder/'selected_columns.npy')]
    assert order.symbol.tolist()==original.symbol.tolist()
    assert order.expression_column.tolist()==original.expression_column.tolist()
    matrix=np.load(folder/'endpoint_matrix.npy')
    assert matrix.shape==(cfg['nodes'],cfg['nodes']) and not np.any(matrix.diagonal())
    assert np.array_equal(matrix!=0,(matrix!=0).T)
    assert (np.count_nonzero(matrix)//2)==stats['edges']
    assert stats['directed_part_acyclic'] and stats['orientation_conflicts']==0
    assert np.isclose(stats['density'],2*stats['edges']/(cfg['nodes']*(cfg['nodes']-1)))
    checks.append({'path':str(folder.relative_to(ROOT)),'status':'PASS','nodes':cfg['nodes'],'edges':stats['edges']})
# Tiny native search on the identical saved benchmark subset, not a new full experiment.
folder=BASE/'boss/bench_n20_cells2000_seed20'
rows=np.load(folder/'selected_rows.npy'); cols=np.load(folder/'selected_columns.npy')
x=np.asarray(np.load(runner.MATRIX,mmap_mode='r')[np.ix_(rows,cols)],dtype=np.float64)
random.seed(20); np.random.seed(20)
native=boss(x,parameters={'lambda_value':2.},verbose=False,node_names=pd.read_csv(folder/'node_order.csv').symbol.tolist())
assert np.array_equal(native.graph,np.load(folder/'endpoint_matrix.npy'))
pcbase=BASE/'pc'
assert np.array_equal(np.load(pcbase/'safety_pc_resume/endpoint_matrix.npy'),np.load(pcbase/'bench_n40_cells2000_seed20/endpoint_matrix.npy'))
assert json.loads((pcbase/'depth_cap_check/statistics.json').read_text())['conditioning_truncated']
result={'status':'PASS','completed_graph_checks':checks,'boss_native_endpoint_equality':True,
 'pc_real_data_resume_endpoint_equality':True,'depth_cap_truncation_reported':True,
 'full_279_node_experiments_launched':False}
runner.atomic_json(BASE/'validation.json',result)
print(json.dumps(result,indent=2))
