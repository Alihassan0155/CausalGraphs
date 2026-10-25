"""Validate fast streaming against generic snapshots and a one-item final batch."""
import argparse
from pathlib import Path
import pickle
import discover as base
import pc_batched as batch
from migrate_pc_batched import strip_checkpoint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixtures',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    if args.output.exists():
        raise SystemExit('Use a NEW output directory')
    args.output.mkdir(parents=True)
    results=[]
    for nodes in (20,40,80):
        source=args.fixtures/f'n{nodes}/legacy_depth1.pkl'
        output=args.output/f'n{nodes}.pkl'
        profile=strip_checkpoint(source,output,args.output/f'n{nodes}.log')
        assert base.sha(output)==base.sha(args.fixtures/f'n{nodes}/stripped.pkl')
        with open(source,'rb') as stream:a=pickle.load(stream)
        with open(output,'rb') as stream:b=pickle.load(stream)
        assert batch.summary(a['cg'])==batch.summary(b['cg']) and b['cg'].test.pvalue_cache=={}
        results.append(dict(nodes=nodes,byte_identical_to_generic=True,profile=profile))
    with open(args.fixtures/'n20/legacy_depth1.pkl','rb') as stream:state=pickle.load(stream)
    metadata={k:v for k,v in state['cg'].test.pvalue_cache.items() if ';' not in k}
    state['cg'].test.pvalue_cache={**metadata,**{f'{i};{i+1}':base.np.float64(i/1000) for i in range(998)}}
    assert len(state['cg'].test.pvalue_cache)==1001
    source=args.output/'final_single_item_source.pkl';base.atomic_pickle(source,state)
    output=args.output/'final_single_item_stripped.pkl'
    single=strip_checkpoint(source,output,args.output/'final_single_item.log')
    with open(output,'rb') as stream:restored=pickle.load(stream)
    assert batch.summary(state['cg'])==batch.summary(restored['cg']) and single['CIKeys']==998
    try:
        strip_checkpoint(args.fixtures/'unsupported_alias.pkl',args.output/'alias_output.pkl',args.output/'alias.log')
        raise AssertionError('Unexpected acceptance of unsupported cross-reference')
    except RuntimeError:
        pass
    base.atomic_json(args.output/'summary.json',dict(cases=results,final_single_item_batch_preserved=True,
                    unsupported_cross_reference_rejected=True,stripper_sha256=base.sha(Path(__file__).with_name('PcCacheStripper.cs'))))
    print('PASS: fast/generic bytes identical, final single item, external-alias rejection')


if __name__=='__main__':main()
