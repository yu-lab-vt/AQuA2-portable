"""Seed majority windows, propagation delays, and super-event merging."""
import numpy as np
from numba import njit

from .act import _components
from .pre import gaussian_xy
from .se_common import my_resize,matlab_round,directions,neighbor_indices,correlate_curves,mean_spatial


def majority_time(curve,seed_start,seed_end,start,end):
    peak=seed_start+int(np.argmax(curve[seed_start:seed_end+1]))
    left=seed_start+int(np.argmin(curve[seed_start:peak+1]));minimum=curve[left];lo=left
    for t in range(left,start-1,-1):
        if curve[t]<minimum:minimum=curve[t];lo=t
        elif curve[t]-minimum>=3:break
    right=peak+int(np.argmin(curve[peak:seed_end+1]));minimum=curve[right];hi=right
    for t in range(right,end+1):
        if curve[t]<minimum:minimum=curve[t];hi=t
        elif curve[t]-minimum>=3:break
    return np.arange(lo,hi+1,dtype=np.int64),peak


def seed_to_majority(seed_pixels,vectors,shape,event,seed_start,seed_end,options):
    spatial=int(np.prod(shape[:3]));time=shape[3]
    reference=mean_spatial(vectors[seed_pixels,:])
    noise=np.sqrt(np.mean(np.diff(reference)**2)/np.float32(2))
    with np.errstate(invalid="ignore",divide="ignore"):reference=reference/noise
    event_times=event//spatial
    window,peak=majority_time(gaussian_xy(reference[None,:],2).ravel(),seed_start,seed_end,int(event_times.min()),int(event_times.max()))
    delays=np.full(shape[:3],np.nan,np.float32,order="F");delay=delays.ravel(order="F")
    delay[seed_pixels]=0
    duration=len(window);scale=max(1,int(matlab_round(duration/options["TPatch"])))
    resized=my_resize(reference[window],1/scale)
    n=len(resized);start=int(window[0]);footprint=np.unique(event%spatial)
    # Connectivity is unchanged when cropping away an all-false exterior.
    # Test contact in the original coordinates to retain clipped borders and
    # seed delays even when a seed lies outside this event's footprint.
    if len(footprint):
        coords=np.unravel_index(footprint,shape[:3],order="F")
        origin=tuple(int(c.min()) for c in coords)
        local_shape=tuple(int(c.max())-lo+1 for c,lo in zip(coords,origin))
        local_pixels=np.ravel_multi_index(tuple(c-lo for c,lo in zip(coords,origin)),local_shape,order="F")
    else:
        origin=(0,0,0);local_shape=(1,1,1);local_pixels=np.empty(0,np.int64)

    best=np.full(spatial,-1,np.float32)
    checked=False;previous=0
    for shift in list(range(n+1))+list(range(-1,-n-1,-1)):
        lo=start+shift*scale;hi=lo+duration
        if lo<0 or hi>time:continue
        if shift>0 and checked:continue
        if shift==-1:previous=0
        curves=my_resize(vectors[footprint,lo:hi],(1,1/scale))
        correlation=correlate_curves(resized,curves)
        chosen=(correlation>.7)&(correlation>best[footprint])
        pixels=footprint[chosen];correlation=correlation[chosen]
        mask=np.zeros(local_shape,bool,order="F");mask.ravel(order="F")[local_pixels[chosen]]=True
        connected=np.zeros(local_shape,bool,order="F")
        for group in _components(mask):
            coords=np.unravel_index(group,local_shape,order="F")
            global_group=np.ravel_multi_index(tuple(c+lo for c,lo in zip(coords,origin)),shape[:3],order="F")
            for offset in directions(3):
                if np.any(delay[neighbor_indices(global_group,shape[:3],offset)]==previous):
                    connected.ravel(order="F")[group]=True;break
        keep=connected.ravel(order="F")[local_pixels[chosen]]
        pixels,correlation=pixels[keep],correlation[keep]
        best[pixels]=correlation
        keep=np.isin(pixels+spatial*(peak+shift*scale),event)
        pixels=pixels[keep]
        if not len(pixels):
            if not checked:checked=True
            else:break
        delay[pixels]=shift;previous=shift
    pixels=np.flatnonzero(~np.isnan(delay))
    return pixels,window,delay[pixels]*np.float32(scale)


def majority_for_event(seed,event,delta,options):
    shape=delta.shape;spatial=int(np.prod(shape[:3]));time=shape[3]
    seed_times=seed//spatial;seed_pixels=np.unique(seed%spatial)
    vectors=delta.reshape((spatial,time),order="F")
    pixels,window,delays=seed_to_majority(seed_pixels,vectors,shape,event,int(seed_times.min()),int(seed_times.max()),options)
    if len(window)!=1:
        selected=np.union1d(seed_pixels,pixels[delays==0])
        curve=mean_spatial(vectors[np.ix_(selected,window)])
        curve=curve-curve.min()
        with np.errstate(divide="ignore",invalid="ignore"):curve=curve/curve.max()
    else:curve=np.array([1.],dtype=np.float64)
    return {"ihw":pixels,"TW":window,"delays":delays,"curve":curve}


def get_majority(seeds,events,delta,options):
    return [majority_for_event(seed,event,delta,options) for seed,event in zip(seeds,events)]


@njit(cache=True)
def dtw_path(x,y):
    n,m=len(x),len(y)
    cost=np.full((n+1,m+1),np.inf,dtype=x.dtype);cost[0,0]=0
    path=np.zeros((n,m),np.uint8)
    for i in range(n):
        for j in range(m):
            a,b,c=cost[i,j],cost[i,j+1],cost[i+1,j]
            if a<=b and a<=c:value=a;direction=0
            elif b<=c:value=b;direction=1
            else:value=c;direction=2
            cost[i+1,j+1]=value+abs(x[i]-y[j]);path[i,j]=direction
    px=np.empty(n+m,np.int64);py=np.empty(n+m,np.int64)
    i=n-1;j=m-1;k=0
    while i>=0 and j>=0:
        px[k]=i;py[k]=j;k+=1
        direction=path[i,j]
        if direction==0:i-=1;j-=1
        elif direction==1:i-=1
        else:j-=1
    return px[:k][::-1],py[:k][::-1]


def avg_dist(curve1,curve2,window1,window2):
    c1,c2=np.asarray(curve1).ravel(),np.asarray(curve2).ravel()
    p1,p2=int(np.argmax(c1)),int(np.argmax(c2))
    left=max(float(c1[:p1+1].min()),float(c2[:p2+1].min()),.1)
    right=max(float(c1[p1:].min()),float(c2[p2:].min()),.1)
    selected=[]
    for curve,peak,window in ((c1,p1,window1),(c2,p2,window2)):
        lo=np.flatnonzero(curve[:peak+1]<=left)[-1]+1
        hi=peak+np.flatnonzero(curve[peak:]<=right)[0]-1
        if lo>hi:lo=hi=peak
        values=curve[lo:hi+1].copy()
        if hi>lo:values-=np.array(min(left,right),dtype=values.dtype)
        values/=values.max()
        selected.append((values,np.asarray(window)[lo:hi+1]))
    (c1,w1),(c2,w2)=selected
    start=min(w1.min(),w2.min());end=max(w1.max(),w2.max())
    c1=np.pad(c1,(int(w1.min()-start)+1,int(end-w1.max())+1))
    c2=np.pad(c2,(int(w2.min()-start)+1,int(end-w2.max())+1))
    ix,iy=dtw_path(c1,c2)
    mask1,mask2=c1[ix]>0,c2[iy]>0
    total=np.sum(abs(iy[mask1]-ix[mask1]))+np.sum(abs(iy[mask2]-ix[mask2]))
    count=mask1.sum()+mask2.sum()
    duration=min(np.sum(c1>0),np.sum(c2>0))+2
    return float((float(total)/float(count))/float(duration))


def create_merging_info(events,majority,regions,shape,options):
    count=len(events)
    labels=np.zeros(shape,np.uint16,order="F");flat=labels.ravel(order="F")
    for i,event in enumerate(events):flat[event]=i+1
    neighbors=[]
    for i,event in enumerate(events):
        growing=event;new=event
        for _ in range(int(2*options.get("spaMergeDist",0))+1):
            all_neighbors=np.concatenate([neighbor_indices(new,shape,offset+(0,)) for offset in directions(3)]) if len(new) else np.empty(0,np.int64)
            new=np.setdiff1d(all_neighbors,growing);growing=np.r_[growing,new]
        neighbors.append(np.setdiff1d(flat[growing],[0,i+1]).astype(np.int64)-1)
    component_labels=np.full(count,-1,np.int64);groups=[]
    for i,region in enumerate(regions):
        values=np.unique(flat[region]);values=values[values>0].astype(np.int64)-1
        groups.append(values);component_labels[values]=i
    return {"neibLst":neighbors,"delayDif":[{} for _ in events],"evtCCLabel":component_labels,"labelsInActRegs":groups}


def relative_delays(one,two,shape):
    coords1=np.column_stack(np.unravel_index(one["ihw"],shape[:3],order="F")).astype(float)+1
    coords2=np.column_stack(np.unravel_index(two["ihw"],shape[:3],order="F")).astype(float)+1
    c1=coords1[one["delays"]==0].mean(axis=0) if np.any(one["delays"]==0) else coords1.mean(axis=0)
    c2=coords2[two["delays"]==0].mean(axis=0) if np.any(two["delays"]==0) else coords2.mean(axis=0)
    dist1=(coords1-c1)@(c2-c1);dist2=(coords2-c2)@(c1-c2)
    return int(matlab_round(one["delays"][dist1>=dist1.max()*.8].mean())),int(matlab_round(two["delays"][dist2>=dist2.max()*.8].mean()))


def merge_super_events(events,majority,info,shape,regions,options):
    n=len(events);footprints=[m["ihw"].copy() for m in majority]
    labels=np.arange(n,dtype=np.int64)
    def find(parent,index):
        while parent[index]!=index:index=int(parent[index])
        return index
    for group in info["labelsInActRegs"]:
        pairs=[]
        for current in group:
            for neighbor in info["neibLst"][current]:
                if neighbor<=current:continue
                if neighbor not in info["delayDif"][current]:
                    a,b=relative_delays(majority[current],majority[neighbor],shape)
                    info["delayDif"][current][int(neighbor)]=avg_dist(majority[current]["curve"],majority[neighbor]["curve"],majority[current]["TW"]+a,majority[neighbor]["TW"]+b)
                distance=info["delayDif"][current][int(neighbor)]
                if distance<=options["maxDelay"]:pairs.append((int(current),int(neighbor),distance))
        parent=np.arange(n,dtype=np.int64)
        for first,second,_ in sorted(pairs,key=lambda p:p[2]):
            first,second=find(parent,first),find(parent,second)
            shared=len(np.intersect1d(footprints[first],footprints[second]))
            if shared/len(footprints[first])<=options["overlap"] and shared/len(footprints[second])<=options["overlap"]:
                low,high=min(first,second),max(first,second)
                parent[high]=low;footprints[low]=np.union1d(footprints[first],footprints[second])
        for event in group:labels[event]=find(parent,int(event))
    _,inverse=np.unique(labels,return_inverse=True)
    super_events=[np.concatenate([events[i] for i in range(n) if inverse[i]==group]) for group in range(len(np.unique(labels)))]
    return super_events,inverse.astype(np.int64),info
