"""Optional temporal peak splitting, preserving the current MATLAB rules."""
import numpy as np

from .act import _components
from .pre import avg_dat_noise_est,gaussian_xy
from .se_common import directions,neighbor_indices,region_crop,to_global,label_lists,my_resize,correlate_curves,mean_spatial
from .se_majority import majority_time,seed_to_majority
from .se_stats import approximate_order_statistics


def _normalized_curve(vectors,pixels,window):
    if not len(pixels):return np.full(len(window),np.nan,np.float32)
    curve=mean_spatial(vectors[np.ix_(pixels,window)])
    curve=curve-curve.min()
    with np.errstate(invalid="ignore",divide="ignore"):return curve/curve.max()


def _new_seed(reference,vectors,shape,event,seed_start,seed_end,options):
    spatial=int(np.prod(shape[:3]));times=event//spatial
    footprint=np.unique(event%spatial)
    window,peak=majority_time(reference,seed_start,seed_end,int(times.min()),int(times.max()))
    scale=max(1,int(np.floor(len(window)/options["TPatch"]+.5)))
    curve=my_resize(reference[window],1/scale)
    values=my_resize(vectors[np.ix_(footprint,window)],(1,1/scale))
    footprint=footprint[correlate_curves(curve,values)>.6]
    # Preserve getInitIhwInSplit's historical H*W and (tPeak+1) formula.
    footprint=footprint[np.isin(footprint+shape[0]*shape[1]*(peak+2),event)]
    if not len(footprint):return np.empty(0,np.int64)
    mask=np.zeros(shape[:3],bool,order="F");mask.ravel(order="F")[footprint]=True
    return max(_components(mask),key=len)


def split_events(delta,original,events,seeds,majority,options):
    shape=delta.shape;spatial=int(np.prod(shape[:3]));time=shape[3];initial=len(events)
    labels=np.zeros(shape,np.uint16,order="F");flat=labels.ravel(order="F")
    for i,event in enumerate(events):flat[event]=i+1
    events=[e.copy() for e in events];seeds=[s.copy() for s in seeds];majority=[dict(m) for m in majority]
    raw=original.reshape((spatial,time),order="F");vectors=delta.reshape((spatial,time),order="F")
    trivial=[False]*initial
    for i in range(initial):
        pixels=events[i];times=pixels//spatial
        footprint=majority[i]["ihw"]
        curve=mean_spatial(raw[footprint,:])
        curve=curve/avg_dat_noise_est(curve,footprint,options)
        smooth=gaussian_xy(curve[None,:],2).ravel()
        start,end=int(times.min()),int(times.max())
        maximum=float(smooth[start:end+1].max());minimum=float(smooth[start:end+1].min())
        thresholds=maximum-.05*np.arange(int(np.floor((maximum-minimum)/.05))+1)
        used=np.ones(time,bool);used[start:end+1]=False
        candidates=[]
        for threshold in thresholds:
            groups=_components(smooth[start:end+1]>threshold)
            for group in groups:
                if len(group)<options["minDur"]:continue
                window=group+start;lo,hi=int(window[0]),int(window[-1])
                if lo==start and start>0 and smooth[start-1]<threshold:continue
                if hi==end and end<time-1 and smooth[end+1]<threshold:continue
                if used[window].any():continue
                duration=len(window)
                left=np.arange(max(0,lo-duration),lo);right=np.arange(hi+1,min(time,hi+duration+1))
                foreground=curve[window]
                backgrounds=(curve[left[smooth[left]<=threshold]],curve[right[smooth[right]<=threshold]])
                missing=(curve[left[smooth[left]>threshold]],curve[right[smooth[right]>threshold]])
                t_scores=[np.inf if not len(bg) else float((foreground.mean()-bg.mean())/np.float32(np.sqrt(1/len(foreground)+1/len(bg)))) for bg in backgrounds]
                if min(t_scores)<=options["sigThr"]:continue
                scores=[]
                for bg,omitted in zip(backgrounds,missing):
                    if not len(bg):scores.append(np.inf);continue
                    mu,sigma=approximate_order_statistics(foreground,bg,omitted)
                    scores.append((foreground.astype(float).mean()-bg.astype(float).mean()-mu)/sigma)
                if min(scores)>options["sigThr"]:
                    candidates.append(window);used[window]=True
        if len(candidates)<=1:continue
        candidates.sort(key=lambda w:w[0])
        gaps=[start-1]+[int(a[-1]+1+np.argmin(smooth[a[-1]+1:b[0]])) for a,b in zip(candidates[:-1],candidates[1:])]+[end]
        valid_gap=np.ones(len(gaps),bool);valid_peak=np.ones(len(candidates),bool)
        for k in range(len(candidates)):
            lo,hi=gaps[k]+1,gaps[k+1]
            peak=lo+int(np.argmax(smooth[lo:hi+1]))
            minimum=max(smooth[lo:peak+1].min(),smooth[peak:hi+1].min())
            if smooth[peak]-minimum<3:
                valid_peak[k]=False
                if k+1==len(gaps)-1:
                    position=k
                    while position>=0 and not valid_gap[position]:position-=1
                    if position<0:raise RuntimeError("No valid splitting margin")
                    valid_gap[position]=False
                else:valid_gap[k+1]=False
        candidates=[w for w,good in zip(candidates,valid_peak) if good]
        gaps=np.asarray(gaps)[valid_gap]
        if len(candidates)<=1:continue
        winner=int(np.argmax([len(np.intersect1d(w,majority[i]["TW"])) for w in candidates]))
        main=True
        for k,candidate in enumerate(candidates):
            lo,hi=int(gaps[k]+1),int(gaps[k+1])
            part=pixels[(times>=lo)&(times<=hi)]
            if not len(part):raise RuntimeError("Temporal split produced an empty component interval")
            _,origin,local_shape,local_pixels=region_crop(part,shape)
            mask=np.zeros(local_shape,bool,order="F");mask.ravel(order="F")[local_pixels]=True
            components=_components(mask)
            if k==winner:components=sorted(components,key=len,reverse=True)
            for component in components:
                event=to_global(component,local_shape,origin,shape)
                if k==winner and main:
                    majority[i]["TW"]=majority_time(curve,int(candidate[0]),int(candidate[-1]),lo,hi)[0]
                    flat[event]=i+1
                    majority[i]["curve"]=_normalized_curve(raw,majority[i]["ihw"],majority[i]["TW"])
                    main=False
                else:
                    seed=_new_seed(curve,vectors,shape,event,int(candidate[0]),int(candidate[-1]),options)
                    if len(seed):
                        current,window,delays=seed_to_majority(seed,vectors,shape,event,int(candidate[0]),int(candidate[-1]),options)
                    else:current=np.empty(0,np.int64);window=np.arange(lo,hi+1);delays=np.empty(0,np.float32)
                    shared=len(np.intersect1d(footprint,current))
                    trivial.append(not (len(current)>0 and shared/len(footprint)>=options["overlap"] and shared/len(current)>=options["overlap"]))
                    majority.append({"ihw":current,"TW":window,"delays":delays,"curve":_normalized_curve(raw,current,window)})
                    seeds.append(seeds[i].copy())
                    flat[event]=min(len(majority),65535)
    events=label_lists(labels)
    for i in range(len(events)):
        intersection=np.intersect1d(seeds[i],events[i])
        seeds[i]=intersection if len(intersection) else events[i].copy()
    for i in range(initial,len(events)):
        if not trivial[i]:continue
        pixels=events[i];window=majority[i]["TW"]
        neighbors=[]
        for offset in directions(3):
            # Source uses unshifted il in this optional split-neighbor call.
            shifted=neighbor_indices(pixels,shape,(offset[0],offset[1],0,0))
            neighbors.extend(np.setdiff1d(flat[shifted],[0,i+1]).tolist())
        neighbors=np.unique(neighbors).astype(np.int64)-1
        neighbors=neighbors[neighbors<initial]
        merged=None
        if len(neighbors):
            overlaps=[len(np.intersect1d(window,majority[n]["TW"]))/len(np.union1d(window,majority[n]["TW"])) for n in neighbors]
            winner=int(np.argmax(overlaps))
            if overlaps[winner]>.8:merged=int(neighbors[winner])
        if merged is not None:
            flat[pixels]=merged+1;events[i]=np.empty(0,np.int64);events[merged]=np.r_[events[merged],pixels]
            old=majority[merged]
            current,new_window,delays=seed_to_majority(old["ihw"],vectors,shape,events[merged],int(old["TW"][0]),int(old["TW"][-1]),options)
            majority[merged]={"ihw":current,"TW":new_window,"delays":delays,"curve":_normalized_curve(raw,current,new_window)}
        elif len(majority[i]["ihw"])<options["minSize"]:
            for offset in directions(4):
                candidate=np.setdiff1d(flat[neighbor_indices(pixels,shape,offset)],[0,i+1])
                if len(candidate):
                    target=int(candidate[0])-1;events[target]=np.r_[events[target],pixels];flat[pixels]=target+1;break
        else:trivial[i]=False
    keep=[i for i in range(len(events)) if not trivial[i]]
    return [events[i] for i in keep],[majority[i] for i in keep],[seeds[i] for i in keep]
