"""Covariate Plackett-Luce and stage-dependent generalized PL for unordered trios.

Meta-rank model uses only prior-year OUT-OF-FOLD F3 component predictions and
market-derived probabilities; learning target is actual 1-2-3 ORDER, not top3 BCE.
"""
from __future__ import annotations

from itertools import combinations, permutations

import numpy as np
from scipy.optimize import minimize

BASE_FEATURES=("xgb800_logit","cat400_logit","market_top3_logit")


def preknown_covariates(xgb: np.ndarray, cat: np.ndarray, market: np.ndarray) -> np.ndarray:
    a=np.column_stack([xgb,cat,market]).astype(float)
    if a.ndim!=2 or a.shape[1]!=3 or len(a)<4 or not np.isfinite(a).all():
        raise ValueError("Incomplete same-time meta features")
    if np.any((a<=0)|(a>=1)):
        raise ValueError("Meta features must be probability forecasts")
    p=np.log(a)-np.log1p(-a)
    p=np.clip((p-p.mean(axis=0))/3,-5,5)
    return p


def rank_stage_alphas(theta: np.ndarray, kind: str) -> np.ndarray:
    if kind=="pl":
        return np.ones(3)
    if kind=="fixed_gpl":
        return np.array([1,.8,.6])
    if kind=="learned_gpl":
        if len(theta)!=5:
            raise ValueError("Generalized PL requires 3 strengths + 2 stage params")
        return np.r_[1.,np.exp(theta[3:])]
    raise ValueError("Unknown preregistered order family")


def ordered_top3_probability(matrix: np.ndarray, ranks: tuple[int,int,int],
                             theta: np.ndarray, kind: str) -> float:
    x=np.asarray(matrix,dtype=float)
    a=rank_stage_alphas(theta,kind)
    s=x@np.asarray(theta[:3],float)
    remain=np.ones(len(s),dtype=bool)
    result=1.
    for rank,idx in enumerate(ranks):
        if not remain[idx]:
            raise ValueError("Repeated top-three finishing horse")
        z=a[rank]*s
        z-=np.max(z[remain])
        w=np.exp(z[remain])
        result*=np.exp(z[idx])/w.sum()
        remain[idx]=False
    return float(result)


def unordered_top3_distribution(matrix:np.ndarray, theta:np.ndarray,
                                kind:str) -> tuple[np.ndarray,np.ndarray]:
    """Sum the six rank permutations, not three independent binary tickets."""
    x=np.asarray(matrix,float)
    n=len(x)
    if n<4 or x.shape!=(n,3):
        raise ValueError("Rank model requires all horses of one race")
    s=x@np.asarray(theta[:3],float)
    alphas=rank_stage_alphas(theta,kind)
    weights=[np.exp(alpha*s-np.max(alpha*s)) for alpha in alphas]
    triplets=np.asarray(list(combinations(range(n),3)),dtype=int)
    prob=np.zeros(len(triplets))
    for permutation in permutations((0,1,2)):
        i,j,k=triplets[:,permutation].T
        w1,w2,w3=weights
        prob+=(
            w1[i]/w1.sum() *
            w2[j]/(w2.sum()-w2[i]) *
            w3[k]/(w3.sum()-w3[i]-w3[j])
        )
    if not np.isfinite(prob).all() or not np.isclose(prob.sum(),1,atol=1e-9):
        raise ValueError("Generalized PL unordered tickets do not sum to one")
    return triplets,prob


def fit_rank_model(races:list[dict],kind:str,lam:float,
                   *,maxiter:int=150)->dict:
    if len(races)<100 or lam not in (.03,.10):
        raise ValueError("Rank model must use pre-2024 source chronological races")
    if kind not in ("pl","fixed_gpl","learned_gpl"):
        raise ValueError("Unexpected rank family")
    initial=np.array([1.,.25,.25])
    bounds=[(-3.,3.)]*3
    if kind=="learned_gpl":
        initial=np.r_[initial,np.log([.8,.6])]
        bounds += [(np.log(.5),np.log(1.5))]*2

    def objective(theta:np.ndarray)->float:
        errors=[]
        for race in races:
            p=ordered_top3_probability(race["x"],race["order"],theta,kind)
            errors.append(-np.log(max(p,1e-35)))
        return float(np.mean(errors)+.5*lam*np.dot(theta[:3],theta[:3]))

    result=minimize(objective,initial,method="L-BFGS-B",bounds=bounds,
                    options={"maxiter":maxiter,"ftol":1e-9})
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise ValueError("Conditional ordered-top3 learner did not converge")
    return {"kind":kind,"lambda":lam,"theta":result.x.tolist(),
            "train_races":len(races),"train_ordered_nll":float(result.fun),
            "optim_success":bool(result.success),"iterations":int(result.nit)}
