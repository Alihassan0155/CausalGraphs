"""Summarize saved small Module 4A runs; never performs causal discovery."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'graphs/causal_split0'
rows=[]
for algo in ['pc','boss']:
    for folder in sorted((BASE/algo).glob('bench_*')):
        config=json.loads((folder/'config.json').read_text())
        status=json.loads((folder/'status.json').read_text())
        row={'algorithm':algo,'nodes':config['nodes'],'cells':config['cells'],'status':status['status'],
             'supervised_seconds':status.get('attempt_seconds')}
        if (folder/'statistics.json').exists():
            stats=json.loads((folder/'statistics.json').read_text())
            for k in ['algorithm_seconds','edges','density','isolated_nodes','edge_types','conditioning_truncated','last_completed_conditioning_depth','directed_part_acyclic','orientation_conflicts']:
                row[k]=stats.get(k)
            row['peak_MiB']=stats['peak_worker_rss_bytes']/2**20
            row['memory_measurement']='worker OS peak / process-tree maximum'
        else:
            row['peak_MiB']=status.get('peak_worker_rss_bytes',0)/2**20
            row['memory_measurement']='sampled process-tree maximum'
            if (folder/'memory_observation.json').exists():
                observation=json.loads((folder/'memory_observation.json').read_text())
                row['peak_MiB']=observation['observed_peak_working_set_bytes']/2**20
                row['memory_measurement']='LOWER BOUND from independent real-worker OS observation; initial shim-only monitor excluded'
            if (folder/'pc_depth_checkpoint.pkl').exists():
                # Use logged depth instead of deserializing potentially large checkpoints.
                for line in (folder/'progress.log').read_text().splitlines():
                    event=json.loads(line)
                    if event['event']=='pc_depth_complete':
                        row['last_completed_conditioning_depth']=event['depth']
                        row['cached_ci_entries']=event['ci_cache_entries']
        rows.append(row)
df=pd.DataFrame(rows); df.to_csv(BASE/'benchmark_summary.csv',index=False)
lines=['# Module 4A feasibility report','',
 'Only reproducible training-control subsets were used; no full 279-node search was launched. All runs use seed 20. PC is stable/Fisher-Z alpha 0.05, unrestricted conditioning for benchmarks; BOSS is genuine causal-learn 0.1.4.8 covariance SEM-BIC lambda=2, one seeded search, returned CPDAG. CPU execution; Windows, 24 logical CPUs and approximately 31.7 GiB RAM. BLAS thread limits default to 1.',
 '', '| Algorithm | Nodes | Cells | Status | Algorithm seconds | Wall seconds | Peak MiB | Edges | Density | Isolates |',
 '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|']
for r in rows:
    mem=('>=' if r['status']!='complete' or r['memory_measurement'].startswith('LOWER') else '')+f"{r['peak_MiB']:.1f}"
    seconds=f"{r['algorithm_seconds']:.3f}" if 'algorithm_seconds' in r else 'not completed'
    density=f"{r['density']:.5f}" if 'density' in r else 'not completed'
    lines.append(f"| {r['algorithm']} | {r['nodes']} | {r['cells']} | {r['status']} | {seconds} | {r['supervised_seconds']:.3f} | {mem} | {r.get('edges','not completed')} | {density} | {r.get('isolated_nodes','not completed')} |")
lines+=['', '## Feasibility and interpretation','',
 'Unrestricted PC is highly sensitive to neighborhood size and conditioning depth: a polynomial extrapolation in node count is not defensible. Timeouts are lower bounds on completion time and do not yield final graphs. Full PC may take hours or longer and consume substantial CI-cache memory; there is no reliable upper bound from these pilots. Start only manually with a wall-time budget. Native conditioning caps offer an explicitly truncated scientific pilot, not an equivalent full-PC result. No correlation pre-screen or algorithm substitution was implemented.',
 '', 'BOSS pilot timing is reported above. A 279-node search may be feasible but is not validated here: GST growth, order-search difficulty and memory are data-dependent. Simple pair-count/quadratic and cubic multipliers from 80 to 279 are 12.16 and 42.42, respectively; these are illustrative workload factors, not confidence intervals or a promised runtime. BOSS uses one start and returns a CPDAG plus a saved representative DAG; no search-state checkpoint is available.',
 '', '## Recovery and validation','',
 'Synthetic PC instrumentation equals native PC output. A real 40-node/2000-cell run was deliberately timed out, resumed from completed depth 6, and produced exactly the uninterrupted endpoint matrix. A max-depth=1 run reported conditioning_truncated=true. Attempting to resume a completed experiment was rejected. Endpoint fixtures distinguish directed, undirected and circle-arrow edges. Windows process-tree termination was tested. The first 80-node PC benchmark began before the monitor fix: its shim-only memory value is excluded; a separate real-worker OS observation supplies only a lower bound. Completed small runs contain real-worker OS peak values.',
 '', 'Subsets are uniformly sampled without replacement using a seed and sorted to preserve source ordering. The selected genes/rows are saved, and PC/BOSS use identical selections at matching dimensions. They are not guaranteed to represent the hardest full-data subgraph. Node vocabulary, correlation and causal matrices share the original 279-gene ordering for full manual runs.',
 '', 'Only training-control rows enter algorithms. Inherited full-file normalization still uses a held-out library-size statistic, so these graphs are not strictly inductive. Observational sufficiency/Markov/faithfulness, acyclicity, Fisher-Z Gaussian partial-correlation and Gaussian-SEM score assumptions are not guaranteed in single-cell data. No edges are established intervention effects. No test responses select parameters.',
 '', 'Complete settings, source/package fingerprints, memory scope, endpoint files and logs are in each run folder. See experiments/future/gnn_split0/MODULE4A.md for manual commands and graph encoding. No baseline files or original datasets were modified.']
(BASE/'feasibility_report.md').write_text('\n'.join(lines),encoding='utf-8')
print(df[['algorithm','nodes','cells','status','supervised_seconds','peak_MiB']].to_string(index=False))
