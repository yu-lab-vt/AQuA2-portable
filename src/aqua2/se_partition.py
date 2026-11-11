"""Marker splitting and growth. No MATLAB runtime is used."""
import numpy as np
from scipy import ndimage

from .act import _components
from .pre import gaussian_xy
from .se_common import label_lists,region_crop,to_global,directions,neighbor_indices,marker_watershed,fill_gaps,mean_spatial


def watershed_partition(score,labels,region_pixels,shape,background=None):
    if shape[2]==1:
        view_shape=(shape[0],shape[1],shape[3])
    else:view_shape=shape
    local_labels=labels.reshape(view_shape,order="F")
    surface=score.reshape(view_shape,order="F").copy(order="F")
    if background is None:
        background=np.ones(view_shape,bool,order="F")
        background.ravel(order="F")[region_pixels]=False
    else:background=np.array(background.reshape(view_shape,order="F"),order="F")
    eroded=ndimage.binary_erosion(background,structure=np.ones((8,)*len(view_shape),bool),origin=-1,border_value=1)
    surface[background]=np.max(surface.ravel(order="F")[region_pixels])+np.array(1,dtype=surface.dtype)
    surface[eroded]=-100
    water=marker_watershed(surface,(local_labels>0)|eroded)
    water=np.array(water.reshape(shape,order="F"),order="F")
    return water,background.reshape(shape,order="F")


def marker_controlled_splitting(seed_map,active_regions,delta,options,progress=None):
    shape=delta.shape
    labels=seed_map.copy(order="F");flat=labels.ravel(order="F")
    seeds=label_lists(seed_map)
    score=-gaussian_xy(delta,3)
    region_labels=[]
    for region in active_regions:
        selected=np.unique(flat[region]);selected=selected[selected>0]
        if len(selected)==1:flat[region]=selected[0]
        region_labels.append(selected)
    splits={}
    for i,(region,selected) in enumerate(zip(active_regions,region_labels)):
        if progress:progress("watershed",i/max(1,len(active_regions)))
        if len(selected)<=1:continue
        slices,start,local_shape,local_pixels=region_crop(region,shape)
        local_labels=np.zeros(local_shape,np.uint16,order="F")
        local_labels.ravel(order="F")[local_pixels]=flat[region]
        water,background=watershed_partition(score[slices],local_labels,local_pixels,local_shape)
        water[background]=0
        locations={int(label):[] for label in selected}
        for group in label_lists(water):
            if not len(group):continue
            ids=np.unique(local_labels.ravel(order="F")[group]);ids=ids[ids>0]
            if not len(ids):continue
            if len(ids)!=1:raise RuntimeError("Watershed basin includes more than one seed label")
            label=int(ids[0])
            mask=np.zeros(local_shape,bool,order="F");mask.ravel(order="F")[group]=True
            components=[c for c in _components(mask) if len(c)>=len(seeds[label-1])]
            if not components:raise RuntimeError("No seed-containing watershed component")
            kept=next((c for c in components if local_labels.ravel(order="F")[c].sum()>0),components[0])
            locations[label].append(to_global(kept,local_shape,start,shape))
        splits[i]=locations
    valid=[]
    for i,(region,selected) in enumerate(zip(active_regions,region_labels)):
        if len(selected):valid.append(region)
        if len(selected)>1:
            for label in selected:
                groups=splits[i][int(label)]
                if groups:flat[np.concatenate(groups)]=label
            pending=region[flat[region]==0]
            fill_gaps(labels,pending)
    return label_lists(labels),seeds,valid,score


def major_curve_filter(original,delta,seeds,events,majority,options,score=None):
    from .se_common import my_resize,matlab_round
    from .se_stats import curve_significance
    from .se_majority import majority_for_event
    shape=delta.shape;spatial=int(np.prod(shape[:3]));time=shape[3]
    score=np.array(-gaussian_xy(delta,3) if score is None else score,dtype=np.float64,order="F")
    labels=np.zeros(shape,np.uint16,order="F");flat=labels.ravel(order="F")
    events=[event.copy() for event in events];seeds=[seed.copy() for seed in seeds];majority=[dict(m) for m in majority]
    for i,event in enumerate(events):flat[event]=i+1
    score[labels==0]=-np.inf
    vectors=original.reshape((spatial,time),order="F")
    updates=set()
    for i in range(len(events)):
        seed_times=seeds[i]//spatial
        seed_start,seed_end=int(seed_times.min()),int(seed_times.max())
        pixels=events[i]
        if not len(pixels):continue
        times=pixels//spatial
        footprint=np.unique(pixels[(times>=seed_start)&(times<=seed_end)]%spatial)
        start,end=int(times.min()),int(times.max())
        curve=mean_spatial(vectors[footprint,:])
        scale=max(1,int(matlab_round((end-start+1)/options["TPatch"])))
        resized=my_resize(curve,1/scale)
        lo=max(1,int(np.floor((seed_start+1)/scale)))-1
        hi=int(np.ceil((seed_end+1)/scale))-1
        if curve_significance(resized,lo,hi,options["sigThr"],
            event_start=max(1,int(np.floor((start+1)/scale)))-1,event_end=int(np.ceil((end+1)/scale))-1):continue
        events[i]=np.empty(0,np.int64)
        slices,origin,local_shape,local_pixels=region_crop(pixels,shape,1)
        local_labels=np.array(labels[slices],order="F")
        local_flat=local_labels.ravel(order="F")
        nearby=np.unique(np.concatenate([neighbor_indices(local_pixels,local_shape,o) for o in directions(4)]))
        neighbors=np.setdiff1d(local_flat[nearby],[0,i+1]).astype(np.int64)
        if not len(neighbors):flat[pixels]=0;continue
        if len(neighbors)==1:
            target=int(neighbors[0])-1
            flat[pixels]=target+1;events[target]=np.r_[events[target],pixels];updates.add(target);continue
        possible=np.setdiff1d(local_labels,[0,i+1])
        for other in possible:
            if other not in neighbors:local_labels[local_labels==other]=0
        background=local_labels==0
        local_flat[local_pixels]=0;flat[pixels]=0
        # Separate touching seed labels in the original direction order.
        pending=np.flatnonzero(local_flat>0)
        for offset in directions(4):
            nearby=neighbor_indices(pending,local_shape,offset)
            remove=(local_flat[nearby]!=local_flat[pending])&(local_flat[nearby]>0)
            local_flat[pending[remove]]=0;pending=pending[~remove]
        water,background=watershed_partition(score[slices],local_labels,local_pixels,local_shape,background)
        fill_gaps(water,local_pixels[water.ravel(order="F")[local_pixels]==0])
        water[background]=0
        for group in label_lists(water):
            if not len(group):continue
            targets=np.unique(local_flat[group]);targets=targets[targets>0]
            if not len(targets):continue
            if len(targets)!=1:raise RuntimeError("Refined watershed basin has multiple event seeds")
            group=np.intersect1d(group,local_pixels)
            if not len(group):continue
            target=int(targets[0])-1
            transferred=to_global(group,local_shape,origin,shape)
            events[target]=np.r_[events[target],transferred]
            flat[transferred]=target+1;updates.add(target)
    for i in sorted(updates):
        if len(events[i]):majority[i]=majority_for_event(seeds[i],events[i],delta,options)
    keep=[i for i,event in enumerate(events) if len(event)]
    return [seeds[i] for i in keep],[events[i] for i in keep],[majority[i] for i in keep]


def grow_watershed_result(events,majority,delta,options):
    shape=delta.shape;spatial=int(np.prod(shape[:3]));time=shape[3]
    labels=np.zeros(shape,np.uint16,order="F")
    labels[delta<options["thrARScl"]-1]=65535
    flat=labels.ravel(order="F")
    footprints=[];new=[];forbidden=[]
    for i,event in enumerate(events):
        flat[event]=i+1
        pixels=np.unique(event%spatial);footprints.append(pixels);new.append(pixels);forbidden.append(np.empty(0,np.int64))
    matrix=labels.reshape((spatial,time),order="F")
    vectors=delta.reshape((spatial,time),order="F")
    from .se_common import correlate_curves
    for _ in range(40):
        for i in range(len(events)):
            # Source assigns [dw,dh,dl] then uses [dh,dw,dl].
            neighbors=np.unique(np.concatenate([neighbor_indices(new[i],shape[:3],(o[1],o[0],o[2])) for o in directions(3)])) if len(new[i]) else np.empty(0,np.int64)
            candidate=np.setdiff1d(neighbors,np.r_[footprints[i],forbidden[i]])
            if not len(candidate):continue
            window=majority[i]["TW"]
            reference=mean_spatial(vectors[majority[i]["ihw"],:])[window]
            peak=int(np.argmax(reference))
            curves=vectors[np.ix_(candidate,window)]
            good=correlate_curves(reference,curves)>.7
            forbidden[i]=np.r_[forbidden[i],candidate[~good]]
            candidate=candidate[good];curves=curves[good]
            added=[]
            for j,pixel in enumerate(candidate):
                existing=matrix[pixel,window].copy()
                left=np.flatnonzero(existing[:peak+1]>0)
                lo=int(left[-1]+1) if len(left) else 0
                if lo>peak:continue
                right=np.flatnonzero(existing[peak:]>0)
                hi=peak+int(right[0])-1 if len(right) else len(window)-1
                if hi<peak:continue
                lo+=int(np.argmin(curves[j,lo:peak+1]));hi=peak+int(np.argmin(curves[j,peak:hi+1]))
                existing[lo:hi+1]=i+1
                if np.any(existing==i+1):matrix[pixel,window]=existing;added.append(pixel)
            added=np.asarray(added,dtype=np.int64)
            footprints[i]=np.r_[footprints[i],added]
            forbidden[i]=np.r_[forbidden[i],np.setdiff1d(candidate,added)]
            new[i]=added
    labels[labels==65535]=0
    output=[];final=np.zeros(shape,np.uint16,order="F")
    for i,event in enumerate(label_lists(labels)):
        if not len(event):output.append(event);continue
        _,start,local_shape,local=region_crop(event,shape)
        mask=np.zeros(local_shape,bool,order="F");mask.ravel(order="F")[local]=True
        components=_components(mask);largest=max(components,key=len)
        kept=to_global(largest,local_shape,start,shape)
        output.append(kept);final.ravel(order="F")[kept]=i+1
    return output,_components(final>0)
