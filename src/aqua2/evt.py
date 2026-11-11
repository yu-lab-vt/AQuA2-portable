"""Spatial event segmentation from the current +evt implementation.

Public voxels, paths and rising-map frame coordinates are zero based.
Internal spgtw delay levels remain one based until se2evt corrects coordinates.
"""
import numpy as np
from scipy import ndimage

from .act import bw2reg,spatial_footprint
from .pre import gaussian_xy
from .se_common import (mean_spatial,matlab_round,label_lists,directions,
    neighbor_indices,marker_watershed,fill_gaps)
from .native import dtw_edge,bilco,midpoint_to_path
from .pipeline import scalar_options


def warp_reference(paths,reference,time):
    output=np.empty((len(paths),time),np.float32)
    for i,path in enumerate(paths):
        total=np.full(time,np.nan);count=np.zeros(time)
        for r,t in path:
            if 0<=r<time and 0<=t<time and not np.isnan(reference[r]):
                total[t]=reference[r] if np.isnan(total[t]) else total[t]+reference[r]
                count[t]+=1
        count[count==0]=1;output[i]=total/count
    return output


def spgtw(delta,se_map,selected,super_voxels,major,options,*,checkpoints=False):
    h,w,l,time=delta.shape;spatial=h*w*l
    coords=np.unravel_index(np.flatnonzero(se_map.ravel(order="F")==selected),delta.shape,order="F")
    footprint=np.unique(np.ravel_multi_index(coords[:3],(h,w,l),order="F"))
    ratio=max(2,int(matlab_round(np.sqrt(len(footprint)/options.get("nRoughPixel",2000)))))
    shape=(int(np.ceil(h/ratio)),int(np.ceil(w/ratio)),l)
    reduced_coords=(coords[0]//ratio,coords[1]//ratio,coords[2],coords[3])
    mask=np.zeros(shape+(time,),bool,order="F");mask[reduced_coords]=True
    super_pixels=np.flatnonzero(mask.any(axis=3).ravel(order="F"))
    major_mask=np.zeros((int(np.prod(shape)),time),bool,order="F")
    for voxels,m in zip(super_voxels,major):
        c=np.unravel_index(voxels,delta.shape,order="F")
        p=np.unique(np.ravel_multi_index((c[0]//ratio,c[1]//ratio,c[2]),shape,order="F"))
        major_mask[np.ix_(p,m["TW"])]=True
    mask=mask.reshape((-1,time),order="F");major_mask &= mask
    mapping=np.zeros(shape,np.int64,order="F");mapping.ravel(order="F")[super_pixels]=np.arange(len(super_pixels))+1
    c=np.unravel_index(footprint,(h,w,l),order="F")
    block_ids=mapping[c[0]//ratio,c[1]//ratio,c[2]]
    pixel_map=np.zeros((h,w,l),np.int64,order="F");pixel_map.ravel(order="F")[footprint]=block_ids
    pixel_lists=label_lists(pixel_map);n=len(pixel_lists)
    vectors=delta.reshape((spatial,time),order="F")
    original=np.zeros((n,time));test=np.zeros_like(original);peaks=[];minimum=time-1
    for k,pixels in enumerate(pixel_lists):
        # MATLAB mtimes(single vector,double scalar) multiplies in double
        # and rounds the product back to single (unlike elementwise .*).
        curve=(mean_spatial(vectors[pixels,:]).astype(float)*np.sqrt(len(pixels))).astype(np.float32)
        curve=gaussian_xy(curve[None,:],2).ravel()
        window=np.flatnonzero(major_mask[super_pixels[k]])
        if not len(window):window=np.flatnonzero(mask[super_pixels[k]])
        peak=int(window[np.argmax(curve[window])]);peaks.append(peak)
        lo=max(0,int(window[0])-5);start=lo+int(np.argmin(curve[lo:peak+1]));minimum=min(minimum,start)
        low,high=curve[start],curve[peak]
        curve[:start+1]=low;original[k]=curve
        curve[peak:]=high;test[k]=curve-low
    maximum=max(peaks);duration=maximum-minimum+1
    temporal_ratio=max(1,int(matlab_round(duration/options["TPatch"])))
    if duration==1:
        maps=[np.full((h,w,l),np.nan,order="F") for _ in range(3)]
        for m in maps:m.ravel(order="F")[footprint]=1
        cx=np.zeros((n,time));cx[:,maximum]=1
        return maps,temporal_ratio,minimum,pixel_lists,cx
    count=int(np.ceil(duration/temporal_ratio))
    test=np.column_stack([test[:,minimum+t*temporal_ratio:min(maximum+1,minimum+(t+1)*temporal_ratio)].mean(axis=1) for t in range(count)])*np.sqrt(temporal_ratio)
    reference_base=np.arange(count,dtype=float)/(count-1)
    reference=test.max(axis=1)[:,None]*reference_base
    half_neighbors=[(dh,dw,1) for dw in (-1,0,1) for dh in (-1,0,1)]+[(1,0,0),(-1,1,0),(0,1,0),(1,1,0)]
    pairs=[]
    for offset in half_neighbors:
        shifted=neighbor_indices(super_pixels,shape,offset)
        neighbors=mapping.ravel(order="F")[shifted]-1
        pairs.extend((i,int(j)) for i,j in enumerate(neighbors) if j>=0 and i!=j)
    pairs=np.sort(np.asarray(pairs,np.int64).reshape(-1,2),axis=1)
    pairs=np.unique(pairs,axis=0) # source orders unique by first, then second
    distances=(reference[:,None,:]-test[:,:,None])**2
    smooth=options["gtwSmo"]*ratio
    if smooth==0 or not len(pairs):
        midpoints=np.vstack([dtw_edge(d) for d in distances])
    else:
        initial=np.repeat((dtw_edge(distances.mean(axis=0))+1)[None,:],n,axis=0)
        midpoints,_=bilco(reference,test,pairs,smooth,initial)
    paths=[midpoint_to_path(cut,count,count) for cut in midpoints]
    aligned=warp_reference(paths,reference_base,count)
    # Relational comparison with the double threshold retains its precision.
    thresholds=np.arange(1,20)*.05;achievement=np.zeros((n,19))
    for k,curve in enumerate(aligned):
        peak=int(np.argmax(curve))
        for j,threshold in enumerate(thresholds):
            hit=np.flatnonzero(curve[:peak+1]>=threshold)
            achievement[k,j]=(hit[0] if len(hit) else peak)+1
    delays=[achievement[:,:7].mean(axis=1),achievement.mean(axis=1),achievement[:,12:19].mean(axis=1)]
    maps=[]
    for values in delays:
        m=np.full((h,w,l),np.nan,order="F");m.ravel(order="F")[footprint]=values[block_ids-1];maps.append(m)
    expanded=np.zeros((n,duration));previous_time=(temporal_ratio+1)/2
    with np.errstate(divide="ignore",invalid="ignore"):
        expanded[:,:int(np.floor(previous_time))]=aligned[:,0,None]/np.float32(previous_time-1)*np.arange(int(np.floor(previous_time)),dtype=np.float32)
    previous_value=aligned[:,0]
    for t in range(1,count):
        lo=int(np.ceil(previous_time));left=t*temporal_ratio+1;right=min((t+1)*temporal_ratio,duration)
        current_time=(left+right)/2;hi=int(np.floor(current_time))
        expanded[:,lo-1:hi]=previous_value[:,None]+(aligned[:,t]-previous_value)[:,None]/np.float32(current_time-previous_time)*(np.arange(lo,hi+1,dtype=np.float32)-np.float32(previous_time))
        previous_value=aligned[:,t];previous_time=current_time
    if duration!=previous_time:
        lo=int(np.ceil(previous_time))
        expanded[:,lo-1:]=previous_value[:,None]+(np.float32(1)-previous_value)[:,None]/np.float32(duration-previous_time)*(np.arange(lo,duration+1,dtype=np.float32)-np.float32(previous_time))
    cx=np.zeros((n,time))
    for k,curve in enumerate(expanded):
        peak=int(np.argmax(curve));global_peak=minimum+peak;cx[k,minimum:global_peak+1]=curve[:peak+1]
        decay=original[k,global_peak:].copy();end=int(np.argmin(decay));low=decay[end];high=original[k,global_peak]
        with np.errstate(divide="ignore",invalid="ignore"):decay=(decay-low)/(high-low)
        decay[decay>1]=1;decay[end:]=0;cx[k,global_peak:]=decay
    result=(maps,temporal_ratio,minimum,pixel_lists,cx)
    if checkpoints:return result+({"pairs":pairs,"test":test,"reference":reference,"midpoints":midpoints,"aligned":aligned,"paths":paths},)
    return result


def delay_map_to_events(delay,major,options):
    shape=delay.shape;finite=np.flatnonzero(np.isfinite(delay).ravel(order="F"))
    low,high=np.nanmin(delay),np.nanmax(delay)
    threshold=1 if options["sourceSensitivity"]==10 else options["TPatch"]*(10-options["sourceSensitivity"])/10
    if high-low<threshold:return np.zeros(len(major),np.int64)
    seeds=np.zeros(shape,np.int64,order="F");flat=seeds.ravel(order="F");values=delay.ravel(order="F");n=0
    for level in np.linspace(low,high,int(options["TPatch"])+1):
        # bw2reg requires HWLT; use a singleton time dimension.
        for pixels in bw2reg((delay<level)[...,None],options):
            if len(pixels)<=max(len(finite)*options["sourceSzRatio"],options["minSize"]):continue
            labels=np.unique(flat[pixels]);labels=labels[labels>0]
            if len(labels)==1:flat[pixels]=labels[0]
            elif len(labels)==0:
                new=pixels;grown=pixels;neighbors=np.empty(0,np.int64)
                for _ in range(100):
                    if not len(new) or len(neighbors)>=len(pixels):break
                    new=np.setdiff1d(np.concatenate([neighbor_indices(new,shape,o) for o in directions(3)]),grown)
                    new=new[~np.isnan(values[new])];neighbors=np.r_[neighbors,new];grown=np.r_[grown,new]
                if len(neighbors) and values[neighbors].mean()-np.nanmean(values[pixels])>threshold:n+=1;flat[pixels]=n
    if n<=1:return np.zeros(len(major),np.int64)
    dimension=2 if shape[2]==1 else 3
    footprint=spatial_footprint(options.get("spaMergeDist",0),dimension)
    erosion=spatial_footprint(2,dimension)
    if dimension==2:footprint=footprint[:,:,None];erosion=erosion[:,:,None]
    outside=~ndimage.binary_dilation(~np.isnan(delay),structure=footprint)
    eroded=ndimage.binary_erosion(outside,structure=erosion,border_value=1)
    surface=np.nan_to_num(delay,nan=high);surface[outside]=high+100;surface[eroded]=0
    # MATLAB removes the singleton depth when selecting watershed connectivity.
    water=marker_watershed(surface[:,:,0],((seeds>0)|eroded)[:,:,0])[:,:,None] if dimension==2 else marker_watershed(surface,(seeds>0)|eroded)
    water[outside]=0
    for pixels in label_lists(water):
        labels=np.setdiff1d(flat[pixels],0)
        if len(labels):flat[pixels]=labels[0]
    fill_gaps(seeds,finite[flat[finite]==0],directions(3))
    labels=[]
    for m in major:
        ids,counts=np.unique(flat[m["ihw"]],return_counts=True)
        if not len(ids):raise ValueError("Source clustering has an empty major footprint")
        labels.append(ids[np.argmax(counts)])
    return np.unique(labels,return_inverse=True)[1].astype(np.int64)


def event_reconstruction(labels,se_map,delta,selected,pixel_lists,curves,options):
    shape=delta.shape;spatial=int(np.prod(shape[:3]));time=shape[3]
    reconstruction=np.zeros((spatial,time))
    for pixels,curve in zip(pixel_lists,curves):reconstruction[pixels,:]=curve
    labels=np.array(labels,order="F",copy=True);se_map=se_map.copy(order="F")
    if options.get("whetherExtend",False):
        sp_map=np.zeros(shape[:3],np.int64,order="F")
        for i,pixels in enumerate(pixel_lists):sp_map.ravel(order="F")[pixels]=i+1
        vectors=gaussian_xy(delta,2).reshape((spatial,time),order="F")
        matrix=labels.reshape((spatial,time),order="F");se_matrix=se_map.reshape((spatial,time),order="F")
        for i,event in enumerate(label_lists(labels)):
            footprint=np.unique(event%spatial);current=matrix==i+1
            for block in np.setdiff1d(sp_map.ravel(order="F")[footprint],0):
                pixels=np.intersect1d(pixel_lists[block-1],footprint,assume_unique=True);curve=mean_spatial(vectors[pixels,:])
                times=np.flatnonzero(current[pixels,:].sum(axis=0)>0);peak=int(times[0])+int(np.argmax(curve[times[0]:times[-1]+1]));high=curve[peak]
                left=np.flatnonzero(curve[:peak+1]<high*options["minShow1"]);right=np.flatnonzero(curve[peak:]<high*options["minShow1"])
                start=int(left[-1])+1 if len(left) else 0;end=peak+int(right[0])-1 if len(right) else time-1
                ix=np.ix_(pixels,np.arange(start,end+1));available=(se_matrix[ix]==0)&(matrix[ix]==0)
                local=matrix[ix];local[available]=i+1;matrix[ix]=local
                local=se_matrix[ix];local[available]=selected;se_matrix[ix]=local
    reconstruction=reconstruction.reshape(shape,order="F");reconstruction[se_map!=selected]=0
    return labels,reconstruction


def se_to_events(delta,se_map,selected,footprint,frames,super_voxels,major,options):
    shape=delta.shape;time=shape[3];minimum=0
    if len(footprint)>30 and time>1:
        maps,ratio,minimum,pixel_lists,cx=spgtw(delta,se_map,selected,super_voxels,major,{**options,"nRoughPixel":2000})
        sv_labels=delay_map_to_events(maps[1],major,options) if len(super_voxels)>1 else np.zeros(len(super_voxels),np.int64)
    else:
        ratio=1;sv_labels=np.zeros(len(super_voxels),np.int64)
        maps=[np.ones(shape[:3]) for _ in range(3)];pixel_lists=[footprint]
        vectors=delta.reshape((-1,time),order="F");curve=np.nanmean(vectors[footprint,:],axis=0)
        curve=gaussian_xy(curve[None,:],2).ravel();curve-=curve.min()
        with np.errstate(divide="ignore",invalid="ignore"):cx=(curve/curve.max())[None,:]
    labels=np.zeros(shape,np.uint16,order="F")
    for i,pixels in enumerate(super_voxels):labels.ravel(order="F")[pixels]=sv_labels[i]+1
    labels,reconstruction=event_reconstruction(labels,se_map,delta,selected,pixel_lists,cx,options)
    maps=[m*ratio+.5-ratio/2+minimum+int(frames[0])-1 for m in maps]
    return reconstruction,labels,maps,int(sv_labels.max())+1,sv_labels


def add_rising_maps(output,labels,maps,origin):
    shape=labels.shape;spatial=int(np.prod(shape[:3]))
    for event in label_lists(labels):
        coords=np.unravel_index(event,shape,order="F");footprint=np.unique(event%spatial)
        lo=tuple(int(c.min()) for c in coords[:3]);hi=tuple(int(c.max())+1 for c in coords[:3]);slices=tuple(slice(a,b) for a,b in zip(lo,hi))
        mask=np.zeros(shape[:3],bool,order="F");mask.ravel(order="F")[footprint]=True
        record={name:np.arange(a,b)+offset for name,a,b,offset in zip(("rgh","rgw","rgl"),lo,hi,origin)}
        for name,m in zip(("dlyMap20","dlyMap50","dlyMap80"),maps):record[name]=np.where(mask,m,np.nan)[slices]
        # MATLAB quantile uses midpoint empirical interpolation and ignores NaN.
        record["maxDly"]=np.ceil(np.nanquantile(record["dlyMap80"],.98,method="hazen"))
        record["minDly"]=np.floor(np.nanquantile(record["dlyMap20"],.02,method="hazen"))
        output.append(record)
    return output


def se_to_events_top(delta,temporal,options=None,*,progress=None):
    options=scalar_options(options if options is not None else temporal["opts"])
    delta=np.asarray(delta,np.float32,order="F");shape=delta.shape
    se_map=np.zeros(shape,np.uint16,order="F")
    for i,pixels in enumerate(temporal["seLst"]):se_map.ravel(order="F")[pixels]=min(i+1,65535)
    reconstructed=np.zeros(shape,np.uint8,order="F");labels=np.zeros(shape,np.uint16,order="F");rising=[];count=0
    for i,event in enumerate(temporal["seLst"]):
        if progress:progress("spatial",i/max(1,len(temporal["seLst"])))
        c=np.unravel_index(event,shape,order="F");gap=min(int(c[3].max()-c[3].min()),int(options["gapExt"]))
        origin=tuple(int(axis.min()) for axis in c[:3])+(max(0,int(c[3].min())-gap),)
        end=tuple(int(axis.max())+1 for axis in c[:3])+(min(shape[3],int(c[3].max())+gap+1),)
        slices=tuple(slice(a,b) for a,b in zip(origin,end));local_shape=tuple(b-a for a,b in zip(origin,end))
        footprint=np.unique(np.ravel_multi_index(tuple(axis-start for axis,start in zip(c[:3],origin[:3])),local_shape[:3],order="F"))
        selected=np.flatnonzero(temporal["seLabel"]==i);super_voxels=[];major=[]
        for j in selected:
            coords=np.unravel_index(temporal["subEvtLst"][j],shape,order="F")
            super_voxels.append(np.ravel_multi_index(tuple(a-b for a,b in zip(coords,origin)),local_shape,order="F"))
            m=dict(temporal["majorInfo"][j]);m["TW"]=m["TW"]-origin[3];pixels=m["ihw"] if len(m["ihw"])>=options["minSize"] else np.empty(0,np.int64)
            coords=np.unravel_index(pixels,shape[:3],order="F")
            m["ihw"]=np.ravel_multi_index(tuple(a-b for a,b in zip(coords,origin)),local_shape[:3],order="F");major.append(m)
        frames=np.arange(origin[3],end[3]);recon,local_labels,maps,n,_=se_to_events(delta[slices],se_map[slices],i+1,footprint,frames,super_voxels,major,options)
        add_rising_maps(rising,local_labels,maps,origin[:3])
        local_map=se_map[slices].copy();local_map[local_labels>0]=i+1;se_map[slices]=local_map
        brightness=np.floor(np.clip(np.nan_to_num(recon*255,nan=0),0,255)+.5).astype(np.uint8)
        reconstructed[slices]=np.maximum(reconstructed[slices],brightness)
        local_labels=local_labels.astype(np.uint32);local_labels[local_labels>0]+=count
        labels[slices]=np.maximum(labels[slices],np.minimum(local_labels,65535).astype(np.uint16));count+=n
    return {"riseLst":rising,"datR":reconstructed,"evt":label_lists(labels),"seLst":temporal["seLst"],"opts":options}


def spatial_mat_payload(result):
    events=np.empty((1,len(result["evt"])),object)
    for i,event in enumerate(result["evt"]):events[0,i]=(event+1).astype(float).reshape(-1,1)
    rising=np.empty((1,len(result["riseLst"])),object)
    for i,record in enumerate(result["riseLst"]):
        converted={}
        for key,value in record.items():
            value=np.asarray(value)
            if key in ("rgh","rgw","rgl"):value=(value+1).astype(float).reshape(1,-1)
            elif key.startswith("dlyMap"):value=(value+1)[:,:,0] if value.shape[2]==1 else value+1
            else:value=(value+1).reshape(1,1)
            converted[key]=value
        rising[0,i]=converted
    return {"riseLst":rising,"datR":result["datR"],"evt":events}


def spatial_from_mat_payload(payload,options=None):
    """Restore event/rising-map indices at the explicit MATLAB boundary."""
    events=[np.asarray(p).astype(np.int64).ravel()-1 for p in payload["evt"].ravel(order="F")]
    rising=[]
    for record in payload["riseLst"].ravel(order="F"):
        converted={}
        for key,value in record.items():
            if key in ("rgh","rgw","rgl"):converted[key]=np.asarray(value).astype(np.int64).ravel()-1
            elif key.startswith("dlyMap"):converted[key]=np.asarray(value)-1
            else:converted[key]=float(np.asarray(value).item())-1
        for key in ("dlyMap20","dlyMap50","dlyMap80"):
            converted[key]=converted[key].reshape(tuple(len(converted[k]) for k in ("rgh","rgw","rgl")),order="F")
        rising.append(converted)
    return {"evt":events,"datR":payload["datR"],"riseLst":rising,"opts":scalar_options(options or {})}
