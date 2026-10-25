"""Exact GES loop-boundary recovery; repeat only an unfinished candidate scan."""
import inspect
import os
import pickle
import random
import discover as b

FIELDS=('G','score','score_new','record_local_score','score_func',
        'count1','update1','G_step1','score_record1','graph_record1',
        'count2','update2','G_step2','score_record2','graph_record2')


def build(report,checkpoint,fingerprint,resume=False,on_saved=None):
    import importlib
    module=importlib.import_module('causallearn.search.ScoreBased.GES')
    initial=None
    if resume:
        with open(checkpoint,'rb') as stream:initial=pickle.load(stream)
        if initial.get('format')!='ges-loop-v1' or initial['configuration_sha256']!=fingerprint:
            raise ValueError('GES checkpoint/configuration mismatch')
        b.np.random.set_state(initial['numpy_rng']);random.setstate(initial['python_rng'])
    phase=initial['phase'] if initial else 1
    state=initial['state'] if initial else {}
    def restored(name,default):return state.get(name,default)
    def save(current,values):
        payload=dict(format='ges-loop-v1',configuration_sha256=fingerprint,phase=current,
                     state={k:values[k] for k in FIELDS if k in values},
                     numpy_rng=b.np.random.get_state(),python_rng=random.getstate())
        temporary=checkpoint.with_suffix('.tmp')
        with open(temporary,'wb') as stream:
            pickle.dump(payload,stream,protocol=pickle.HIGHEST_PROTOCOL);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,checkpoint)
        if on_saved:on_saved(current,payload['state'])
    source=inspect.getsource(module.ges)
    anchor='    graph_record1 = []\n    while True:'
    assert source.count(anchor)==1
    restore1='\n'.join(f'    {k} = _restore("{k}", {k})' for k in FIELDS[:10])
    source=source.replace(anchor,'    graph_record1 = []\n'+restore1+'\n    while _resume_phase != 2:\n        _save(1, locals())')
    anchor='    graph_record2 = []\n    while True:'
    assert source.count(anchor)==1
    restore2='\n'.join(f'    {k} = _restore("{k}", {k})' for k in FIELDS)
    source=source.replace(anchor,'    graph_record2 = []\n    if _resume_phase == 2:\n'+
                          '\n'.join('    '+line for line in restore2.splitlines())+'\n    while True:\n        _save(2, locals())')
    for current in (1,2):
        anchor=f'            G_step{current}.append(G)';assert source.count(anchor)==1
        source=source.replace(anchor,anchor+f'\n            _report({current}, G, score_new, len(update{current}))')
    namespace=dict(module.__dict__,_restore=restored,_resume_phase=phase,_save=save,_report=report)
    exec(compile(source,inspect.getfile(module.ges),'exec'),namespace)
    return namespace['ges']
