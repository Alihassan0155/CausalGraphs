"""Small native/instrumented GES and Gaussian-score checks; no full discovery."""
import json
import numpy as np
import discover as b
from discover_ges import instrument
from causallearn.search.ScoreBased.GES import ges
from causallearn.score.LocalScoreFunction import local_score_BIC_from_cov

rng=np.random.default_rng(20);x=rng.normal(size=(600,8));x[:,2]=x[:,0]+x[:,1]+.3*x[:,2]
events=[]
native=ges(x,parameters={'lambda_value':2.},score_func='local_score_BIC_from_cov')
hooked=instrument(lambda phase,g,score,moves:events.append((phase,moves)))(x,parameters={'lambda_value':2.},score_func='local_score_BIC_from_cov')
assert np.array_equal(native['G'].graph,hooked['G'].graph) and native['score']==hooked['score']
assert repr(native['update1'])==repr(hooked['update1']) and repr(native['update2'])==repr(hooked['update2'])
cov=np.cov(x.T,ddof=0);local=local_score_BIC_from_cov((cov,600),2,[0,1],{'lambda_value':2.})
res=x[:,2]-x[:,2].mean()-(x[:,:2]-x[:,:2].mean(axis=0))@np.linalg.lstsq(x[:,:2]-x[:,:2].mean(axis=0),x[:,2]-x[:,2].mean(),rcond=None)[0]
expected=-300*(1+np.log(np.mean(res**2)))-2*3*np.log(600)
assert np.isclose(local,expected,rtol=0,atol=1e-9)
out=b.BASE/'ges/instrumentation_validation.json'
if out.exists():raise SystemExit('Existing validation protected')
b.atomic_json(out,dict(native_endpoints_equal=True,native_score_equal=True,native_updates_equal=True,
                      score_regression_check=True,accepted_moves_logged=len(events)))
print('PASS native/instrumented GES endpoints, score, updates; explicit Gaussian score')
