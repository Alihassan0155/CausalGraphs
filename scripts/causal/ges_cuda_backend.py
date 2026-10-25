"""Experimental CUDA Gaussian score; unchanged GES operators and loop recovery."""
import discover as b
import torch
from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
from causallearn.score.LocalScoreFunction import local_score_BIC_from_cov


class CudaScore(LocalScoreClass):
    def __init__(self,*args,**kwargs):
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no fallback')
        super().__init__(*args,**kwargs)
        assert self.local_score_fun.__name__=='local_score_BIC_from_cov'
        self.device_cov=None
        self.metrics=dict(misses=0,max_score_difference=0.,score_checks=0)
        self._device()
        # Empty and singleton parent scores are requested repeatedly early in GES.
        # Compute all in one device batch; preserve CPU np.log and score arithmetic.
        cov=self.device_cov;diagonal=torch.diagonal(cov)
        variance=(diagonal[:,None]-cov.square()/diagonal[None,:]).cpu().numpy()
        for i in range(len(self.cov)):
            self.score_cache[i]={():self._formula(float(self.cov[i,i]),0)}
            for j in range(len(self.cov)):
                if i!=j:self.score_cache[i][(j,)]=self._formula(float(variance[i,j]),1)

    def _device(self):
        if self.device_cov is None:self.device_cov=torch.tensor(self.cov,dtype=torch.float64,device='cuda')

    def _formula(self,sigma,parents):
        if sigma<=0:sigma=b.np.finfo(float).eps
        return -.5*self.n*(1+b.np.log(sigma))-self.parameters['lambda_value']*(parents+1)*b.np.log(self.n)

    def score(self,i,parents):
        key=tuple(sorted(parents));cache=self.score_cache.setdefault(i,{})
        if key not in cache:
            self._device();cov=self.device_cov
            idx=torch.tensor(parents,dtype=torch.long,device='cuda')
            xx=cov[idx[:,None],idx[None,:]];yx=cov[i,idx].reshape(1,-1)
            inverse,info=torch.linalg.inv_ex(xx)
            if info.item()!=0:inverse=torch.linalg.pinv(xx) # same fallback policy, not guaranteed identical numerics
            sigma=float((cov[i,i]-yx@inverse@yx.T).item())
            if not b.np.isfinite(sigma):raise ValueError('Nonfinite CUDA variance')
            cache[key]=self._formula(sigma,len(parents));self.metrics['misses']+=1
        return cache[key]

    def __getstate__(self):
        state=self.__dict__.copy();state['device_cov']=None;return state

    def audit_cache(self):
        for i,cache in self.score_cache.items():
            for parents,value in cache.items():
                reference=local_score_BIC_from_cov((self.cov,self.n),i,list(parents),self.parameters)
                self.metrics['max_score_difference']=max(self.metrics['max_score_difference'],float(abs(reference-value)))
                self.metrics['score_checks']+=1
        return self.metrics
