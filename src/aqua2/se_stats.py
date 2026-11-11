"""Seed statistics using the original finite-sample order-statistics table."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy import special,stats

from .io import read_mat
from .pre import truncated_kept_var
from .se_common import my_resize,matlab_round


@lru_cache(maxsize=1)
def order_table():
    table=read_mat(Path(__file__).with_name("data")/"Order_mus_sigmas.mat")
    return table["mus"].ravel(),table["covMatrixs"].ravel()


def order_statistics(fg,bg,omit=()):
    fg,bg,omit=(np.asarray(v).ravel() for v in (fg,bg,omit))
    m,n=len(fg),len(bg); count=m+n+len(omit)
    if not m or not n: return np.nan,np.nan
    means,covariances=order_table()
    if count>len(means):
        raise ValueError(f"Original order-statistics table supports n <= {len(means)}, got {count}")
    labels=np.r_[-np.ones(n),np.ones(m),np.zeros(len(omit))]
    labels=labels[np.argsort(np.r_[bg,fg,omit],kind="stable")]
    mu=means[count-1].ravel()
    expectation=mu[labels==1].mean()-mu[labels==-1].mean()
    covariance=covariances[count-1].copy()
    covariance[labels==0,:]=0;covariance[:,labels==0]=0
    covariance[labels==1,:]/=m;covariance[:,labels==1]/=m
    covariance[labels==-1,:]=-covariance[labels==-1,:]/n
    covariance[:,labels==-1]=-covariance[:,labels==-1]/n
    return expectation,np.sqrt(covariance.ravel(order="F").sum())


def approximate_order_statistics(fg,bg,omit=()):
    fg,bg,omit=(np.asarray(v,dtype=np.float64).ravel() for v in (fg,bg,omit))
    m,n=len(fg),len(bg); size=m+n+len(omit)
    if not m or not n: return np.nan,np.nan
    labels=np.r_[-np.ones(n),np.ones(m),np.zeros(len(omit))]
    labels=labels[np.argsort(np.r_[bg,fg,omit],kind="stable")]
    breaks=np.flatnonzero(np.diff(labels))+1
    a=np.r_[labels[breaks-1],labels[-1]]
    a[a>0]*=size/m;a[a<0]*=size/n
    low=special.ndtri(np.r_[0,breaks/size]);low[0]=-40
    high=special.ndtri(np.r_[breaks/size,1]);high[-1]=40
    pdf=lambda x:np.exp(-x*x/2)/np.sqrt(2*np.pi)
    f1=lambda x:x*special.ndtr(x)+pdf(x)
    f2=lambda x:.5*(special.ndtr(x)*x*x-special.ndtr(x)+pdf(x)*x)
    mean=np.sum(a*(pdf(low)-pdf(high)))
    term1=sum(a[i]*np.sum(a[i+1:]*(high[i+1:]-low[i+1:]))*(f1(high[i])-f1(low[i])) for i in range(len(a)-1))
    term2=np.sum(a*a*high*(f1(high)-f1(low)))
    term3=np.sum(a*a*(f2(high)-f2(low)))
    variance=2*(term1+term2-term3)-np.sum(a*(f1(high)-f1(low)))**2
    return mean,np.sqrt(variance)/np.sqrt(size)


def get_seed_score(pixels,time_scale,data,shape):
    h,w,z,t=np.unravel_index(pixels,shape,order="F")
    total=shape[3]; spatial=int(np.prod(shape[:3])); reduced=total//time_scale
    t=t//time_scale; good=t<reduced
    positions=np.ravel_multi_index((h[good],w[good],z[good],t[good]),shape[:3]+(reduced,),order="F")
    unique,counts=np.unique(positions,return_counts=True)
    foreground=unique[counts>time_scale/2]
    if not len(foreground): return (0.,)*4
    times=foreground//spatial; spatial_ids=foreground%spatial
    footprints=np.unique(spatial_ids)
    foregrounds=[];lefts=[];rights=[];holes=[];noise=[];degrees=[];sizes=[]
    zero_count=0;sample_count=0
    for pixel in footprints:
        current=times[spatial_ids==pixel]; duration=len(current); sizes.append(duration)
        original=data[pixel,:]
        curve=my_resize(original,1/time_scale)*np.float32(np.sqrt(time_scale))
        lo,hi=int(current.min())+1,int(current.max())+1
        left=np.arange(max(1,lo-duration),lo)
        right=np.arange(hi+1,min(reduced,hi+duration)+1)
        foregrounds.append(curve[current]);lefts.append(curve[left-1]);rights.append(curve[right-1])
        holes.append(curve[np.setdiff1d(np.arange(lo-1,hi),current)])
        for side,neighbor in ((-1,left),(1,right)):
            if not len(neighbor):
                degrees.append(0);continue
            shifted=neighbor+side
            if side==-1:
                start=max(2,int(shifted.min())*time_scale);end=int(shifted.max())*time_scale
                differences=(original[start-1:end]-original[start-2:end-1])**2 if end>=start else np.empty(0,original.dtype)
                sample=original[start-2:end] if end>=start-1 else np.empty(0,original.dtype)
            else:
                start=int(shifted.min())*time_scale;end=min(int(shifted.max())*time_scale,total-1)
                differences=(original[start-1:end]-original[start:end+1])**2 if end>=start else np.empty(0,original.dtype)
                sample=original[start-1:end+1] if end>=start-1 else np.empty(0,original.dtype)
            zero_count+=int(np.sum(sample==0));sample_count+=end-start+2
            noise.append(differences);degrees.append(len(differences))
    fg=np.concatenate(foregrounds);fg=fg[~np.isnan(fg)]
    left=np.concatenate(lefts);left=left[~np.isnan(left)]
    right=np.concatenate(rights);right=right[~np.isnan(right)]
    noise=np.concatenate(noise) if noise else np.empty(0)
    if len(left)<=1 or len(right)<=1 or len(noise)<=2: return (0.,)*4
    correction=np.float32(truncated_kept_var(zero_count/sample_count))
    sigma=np.float32(np.sqrt(noise.mean()/correction)/np.float32(np.sqrt(time_scale)))
    dof=2*np.sum(degrees,dtype=float)**2/(3*np.sum(degrees)-len(degrees))
    dof=int(matlab_round(dof))
    z_values=[]
    for backgrounds,opposite in ((lefts,rights),(rights,lefts)):
        means=np.zeros(len(footprints));values=np.full(len(footprints),-np.inf)
        for i in range(len(footprints)):
            if len(backgrounds[i]):
                f,b=foregrounds[i]/sigma,backgrounds[i]/sigma
                other=np.r_[opposite[i],holes[i]]/sigma
                mu,sd=order_statistics(f,b,other)
                means[i]=mu/sd
                values[i]=np.float32((f.mean()-b.mean())/np.float32(sd))
        z=stats.norm.isf(stats.nct.sf(values,dof,means))
        z_values.append(np.sqrt(sizes)@z/np.sqrt(np.sum(sizes)))
    fg,left,right=fg/sigma,left/sigma,right/sigma
    scores=[]
    for background in (left,right):
        score=np.float32((fg.mean()-background.mean())/np.float32(np.sqrt(1/len(fg)+1/len(background))))
        if len(noise)<100:
            score=np.float32(stats.norm.isf(np.float32(stats.t.sf(score,len(noise)-1))))
        scores.append(score)
    return tuple(z_values+scores)


def curve_significance(curve,start,end,threshold,event_start=None,event_end=None):
    """Zero-based inclusive bounds; event bounds select majorCurveFilter2 rule."""
    curve=np.asarray(curve).ravel()
    sigma=np.float32(np.sqrt(np.median(np.diff(curve)**2)/np.float32(.9099)))
    with np.errstate(divide="ignore",invalid="ignore"): curve=curve/sigma
    total=len(curve);left_sig=False;right_sig=False
    if event_start is None:
        maximum=np.max(curve[start:end+1]);minimum=np.min(curve[start:end+1]);steps=5
    else:
        peak=start+int(np.argmax(curve[start:end+1]));maximum=curve[peak]
        start=max(0,event_start-1);end=min(total-1,event_end+1)
        start+=int(np.argmin(curve[start:peak+1]));end=peak+int(np.argmin(curve[peak:end+1]))
        if curve[start]<curve[end]:
            start=start+np.flatnonzero(curve[start:peak+1]<curve[end])[-1]
        elif curve[end]<curve[start]:
            end=peak+np.flatnonzero(curve[peak:end+1]<curve[start])[0]
        minimum=np.min(curve[max(0,start-5):min(total,end+6)]);steps=20
    thresholds=[maximum] if maximum==minimum else np.linspace(maximum,minimum,steps+1,dtype=curve.dtype)
    for value in thresholds:
        if event_start is None:
            selected=np.flatnonzero(curve[start:end+1]>=value)
            if not len(selected): continue
            lo=start+selected[0];hi=start+selected[-1]
            previous=np.flatnonzero(curve[:lo]>=value)
            next_values=np.flatnonzero(curve[hi+1:]>=value)
            left_start=int(previous[-1]+1) if len(previous) else 0
            right_end=hi+int(next_values[0]) if len(next_values) else total-1
        else:
            before=np.flatnonzero(curve[start:peak+1]<value)
            after=np.flatnonzero(curve[peak:end+1]<value)
            lo=start+int(before[-1])+1 if len(before) else start
            hi=peak+int(after[0])-1 if len(after) else end
            last=lo-1
            while last>=0 and curve[last]>=value:last-=1
            previous=np.flatnonzero(curve[:last+1]>=value)
            left_start=int(previous[-1]+1) if len(previous) else 0
            first=hi+1
            while first<total and curve[first]>=value:first+=1
            after=np.flatnonzero(curve[first:]>=value)
            right_end=first+int(after[0])-1 if len(after) else total-1
        duration=hi-lo+1
        foreground=curve[lo:hi+1]
        left=curve[max(left_start,lo-duration):lo]
        right=curve[hi+1:min(right_end,hi+duration)+1]
        if not len(foreground):continue
        for direction,background,other in ((0,left,right),(1,right,left)):
            if not len(background):continue
            score=np.float32((foreground.mean()-background.mean())/np.float32(np.sqrt(1/len(foreground)+1/len(background))))
            if score>=threshold:
                mu,sd=order_statistics(foreground,background,other)
                z=np.float32((foreground.mean()-background.mean()-np.float32(mu))/np.float32(sd))
                if z>=threshold:
                    if direction==0:left_sig=True
                    else:right_sig=True
        if left_sig and right_sig:return True
    return left_sig and right_sig
