"""Exactly-three-subset conditional Bernoulli research primitives.

No production use: requires ALL actual starters and exactly three top3
binary outcomes per race. Not a Brier improvement claim.
"""
from __future__ import annotations

import numpy as np

K = 3


def _normalizer_and_prefix(log_weights: np.ndarray
                           ) -> tuple[float,np.ndarray,np.ndarray]:
    lw=np.asarray(log_weights,dtype=float)
    if lw.ndim!=1 or len(lw)<K or not np.isfinite(lw).all():
        raise ValueError("Require >=3 finite all-starter log weights")
    n=len(lw)
    prefix=np.full((n+1,K+1),-np.inf)
    suffix=np.full((n+1,K+1),-np.inf)
    prefix[0,0]=0
    suffix[n,0]=0
    for i in range(n):
        prefix[i+1,0]=0
        for t in range(1,min(K,i+1)+1):
            prefix[i+1,t]=np.logaddexp(
                prefix[i,t],lw[i]+prefix[i,t-1])
    for i in range(n-1,-1,-1):
        suffix[i,0]=0
        for t in range(1,min(K,n-i)+1):
            suffix[i,t]=np.logaddexp(
                suffix[i+1,t],lw[i]+suffix[i+1,t-1])
    return float(prefix[n,K]),prefix,suffix


def top3_subset_marginals(log_weights: np.ndarray) -> np.ndarray:
    """P(i in S | |S|=3) where P(S) is proportional to exp(sum(log_w[S])).

    Log-space elementary symmetric polynomial dynamic programming is
    O(3*n) rather than enumerating all n-choose-3 combinations.
    """
    lw=np.asarray(log_weights,dtype=float)
    logz,prefix,suffix=_normalizer_and_prefix(lw)
    p=np.empty(len(lw),dtype=float)
    for i in range(len(lw)):
        # e2(excluding horse i) = sum_{k=0}^{2} e_k(prefix)*e_{2-k}(suffix)
        e2=np.logaddexp.reduce(np.array([
            prefix[i,t]+suffix[i+1,2-t] for t in range(3)
        ]))
        p[i]=np.exp(lw[i]+e2-logz)
    if not np.all((p>=-1e-12)&(p<=1+1e-12)):
        raise ValueError("Invalid subset marginal")
    if not np.isclose(p.sum(),K,atol=1e-10):
        raise ValueError("Three-subset probability sum is not three")
    return np.clip(p,0,1)


def subset_log_probability(log_weights: np.ndarray,
                           selected_indices: list[int]|tuple[int,...]) -> float:
    """Log probability of exactly one specific 3-horse set."""
    lw=np.asarray(log_weights,dtype=float)
    logz,_,_=_normalizer_and_prefix(lw)
    indices=list(selected_indices)
    if len(indices)!=K or len(set(indices))!=K or (
        any((not isinstance(i,(int,np.integer)) or i<0 or i>=len(lw))
            for i in indices)
    ):
        raise ValueError("The selection must contain exactly three distinct starter indices")
    return float(lw[indices].sum()-logz)
