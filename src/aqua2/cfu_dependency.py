"""CFU dependency saddlepoint p-values, signed delays, and ordered grouping."""
import numpy as np
from math import erfc
from numba import njit
from scipy.stats import binom


@njit(cache=True)
def solve_k1(p,m):
    if m==len(p):return np.inf
    lo=-1000.;hi=1000.
    for _ in range(256):
        mid=(lo+hi)/2;ex=np.exp(mid);value=0.
        for prob in p:value+=1-(1-prob)/(prob*ex+1-prob)
        value-=m
        if abs(value)<=1e-10:return mid
        if value>0:hi=mid
        else:lo=mid
    raise RuntimeError('CFU saddlepoint root did not converge')


@njit(cache=True)
def binomial_tail(p,m):
    n=len(p)
    if m==0:return 1.
    if m==n:
        total=0.
        for value in p:total+=np.log(value)
        return np.exp(total)
    total=0.
    for value in p:total+=value
    if abs(m-total)<1e-3:
        k3=0.;k2=0.
        for value in p:k3+=value-2*value**2+2*value**3;k2+=(1-value)*value
        return .5-1/np.sqrt(2*np.pi)*(k3/6/k2**1.5-.5/np.sqrt(k2))
    root=solve_k1(p,m);ex=np.exp(root);k2=0.;k=0.
    for value in p:
        denominator=value*ex+1-value;k2+=value*(1-value)*ex/denominator**2;k+=np.log(denominator)
    w=np.sign(root)*np.sqrt(2*(root*m-k));u=(1-np.exp(-root))*np.sqrt(k2)
    # erfc is the direct upper-tail formula, avoiding cancellation at large w.
    from_tail=0.5*erfc(w/np.sqrt(2.))
    return from_tail-np.exp(-w*w/2)/np.sqrt(2*np.pi)*(1/w-1/u)


@njit(cache=True)
def _dependency(first,second,shift,distances):
    length=len(second);centers=np.flatnonzero(first)
    centers=centers[centers+shift<length]+shift
    positions=np.flatnonzero(second);rate=len(positions)/length
    best=1.;best_distance=int(distances[0]);best_delays=np.empty(0,np.int64)
    for k,d in enumerate(distances):
        n=len(centers);prob=np.empty(n);delays=np.empty(n,np.int64);count=0
        for j,center in enumerate(centers):
            start=max(0,center-d);end=min(length-1,center+d)
            if j and centers[j-1]+d+1>=center-d:start=int(np.floor((centers[j-1]+center)/2+.5))
            if j+1<n and center+d+1>=centers[j+1]-d:end=int(np.floor((center+centers[j+1])/2+.5))-1
            prob[j]=1-np.exp(-rate*(end-start+1));nearest=length+1;chosen=0
            for pos in positions:
                if pos<start:continue
                if pos>end:break
                delta=pos-center
                if abs(delta)<nearest:nearest=abs(delta);chosen=delta
            if nearest<=length:delays[count]=chosen+shift;count+=1
        p=binomial_tail(prob,count)
        if k==0 or p<best:best=p;best_distance=int(d);best_delays=delays[:count].copy()
    return best,best_distance,best_delays


def dependency(first,second,shift=0,distances=None):
    first=np.asarray(first,bool).ravel();second=np.asarray(second,bool).ravel()
    if len(first)!=len(second) or not len(first):raise ValueError('Occurrence sequences must have equal positive length')
    shift=int(shift);distances=np.arange(11,dtype=np.int64) if distances is None else np.asarray(distances,np.int64)
    if shift<0 or len(distances)==0 or np.any(distances<0):raise ValueError('Shift/distances must be nonnegative and distances nonempty')
    p,d,delays=_dependency(first,second,shift,distances);values,counts=np.unique(delays,return_counts=True)
    return float(p),int(d),np.column_stack((values,counts)).astype(float)


def dependency_simple(first,second,distances=None):
    a=np.flatnonzero(first);b=np.flatnonzero(second)
    if not len(a) or not len(b):return 1.,1.
    distances=np.arange(11) if distances is None else np.asarray(distances)
    column=np.asarray(first).ndim==2 and np.asarray(first).shape[1]==1
    delta=abs(a[:,None]-b) if column else abs(b[:,None]-a);result=[]
    for n1,n2,nearest in ((len(a),len(b),delta.min(axis=0)),(len(b),len(a),delta.min(axis=1))):
        result.append(min(binom.sf(np.count_nonzero(nearest<=d)-1,n1,1-np.exp(-n2/np.asarray(first).size*(2*d+1))) for d in distances))
    return tuple(result)


def all_dependencies(info1,info2=(),max_distance=10,shift=0,progress=None):
    sequences=[i['occurrence'] for i in [*info1,*info2]];n=len(sequences);output=[];distances=np.arange(int(max_distance)+1)
    for i in range(n):
        for j in range(i+1,n):
            p1,_,d1=dependency(sequences[i],sequences[j],abs(int(shift)),distances)
            p2,_,d2=dependency(sequences[j],sequences[i],abs(int(shift)),distances)
            if p1<p2:p,direction,sign=p1,d1,1
            else:p,direction,sign=p2,d2,-1
            delay=float(sign*(direction[:,0]@direction[:,1])/direction[:,1].sum()) if len(direction) else np.nan
            if p<.1:output.append([i,j,p,delay])
        if progress:progress('dependency',i+1,n)
    return np.asarray(output,float).reshape(-1,4)


def group_cfus(info1,info2,relation,p_threshold=1e-5,min_cfus=3):
    n=len(info1)+len(info2);relation=np.asarray(relation,float).reshape(-1,4).copy()
    if not len(relation):return []
    relation=relation[relation[:,2]<p_threshold];negative=relation[:,3]<0
    relation[negative,:2]=relation[negative,1::-1];relation[negative,3]*=-1
    relation=relation[np.argsort(relation[:,2],kind='stable')];parent=np.arange(n)
    def root(i):
        while parent[i]!=i:i=parent[i]
        return i
    for a,b,_,_ in relation:
        a,b=root(int(a)),root(int(b));parent[a]=parent[b]=min(a,b)
    groups=[np.array([i for i in range(n) if root(i)==r],np.int64) for r in range(n)]
    groups=sorted([g for g in groups if len(g)>=min_cfus],key=lambda g:-len(g))
    visited=np.zeros(n,bool);delay=np.full(n,np.nan);added=delay.copy();output=[]
    for labels in groups:
        values=np.array([relation[(relation[:,0]==c)|(relation[:,1]==c),2].mean() for c in labels]);start=labels[np.argmin(values)]
        visited[start]=True;added[start]=values.min();delay[start]=0;queue=[start]
        for current in queue:
            for forward in (True,False):
                selected=relation[relation[:,0 if forward else 1]==current]
                for a,b,p,d in selected:
                    neighbor=int(b if forward else a)
                    if not visited[neighbor]:
                        delay[neighbor]=delay[current]+(d if forward else -d);added[neighbor]=p;queue.append(neighbor);visited[neighbor]=True
        output.append(dict(id=len(output),cfuIds=labels,delays=delay[labels].copy(),pvalues=added[labels].copy()))
    return output
