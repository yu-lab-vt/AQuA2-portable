"""Region, landmark distance, and frontier propagation features.

Indices/centers/borders are zero based. The MATLAB boundary adapter adds one.
The legacy Cartesian landmark projection and per-component frame assignment
are preserved deliberately; fixing them would change the scientific baseline.
"""
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist
from numba import njit
from .act import _components
from .se_common import matlab_round
from .fea_geometry import boundaries,bicubic_resize


def _cell(values,column=True):
    output=np.empty((len(values),1) if column else (1,len(values)),object)
    for i,value in enumerate(values):output.ravel()[i]=value
    return output


def mask_pixels(masks,shape):
    result=[]
    if masks is None:masks=[]
    elif isinstance(masks,np.ndarray) and masks.dtype!=object and (masks.dtype==bool or masks.ndim==1):masks=[masks]
    for mask in masks:
        array=np.asarray(mask)
        if array.dtype==bool:
            if array.size!=int(np.prod(shape)):raise ValueError("Feature mask must match working spatial coordinates")
            pixels=np.flatnonzero(array.reshape(shape,order="F").ravel(order="F"))
        else:
            if not np.isfinite(array).all() or not np.equal(array,np.floor(array)).all():raise ValueError("Mask indices must be finite integers")
            pixels=np.unique(array.astype(np.int64).ravel())
            if np.any((pixels<0)|(pixels>=np.prod(shape))):raise ValueError("Feature mask index out of range")
        if not len(pixels):raise ValueError("Region/landmark masks must be nonempty")
        result.append(pixels)
    return result


def polygon_info(polygons,shape):
    masks=[];centers=[];borders=[];radii=[]
    for pixels in polygons:
        mask=np.zeros(shape,bool,order="F");mask.ravel(order="F")[pixels]=True
        coordinates=np.column_stack(np.unravel_index(pixels,shape,order="F"))
        center=matlab_round(coordinates.mean(axis=0)+1)-1
        if shape[2]==1:
            perimeter=mask[:,:,0]&~ndimage.binary_erosion(mask[:,:,0],structure=ndimage.generate_binary_structure(2,1));perimeter=perimeter[:,:,None]
        else:perimeter=mask&~ndimage.binary_erosion(mask,structure=ndimage.generate_binary_structure(3,1))
        border=np.column_stack(np.unravel_index(np.flatnonzero(perimeter.ravel(order="F")),shape,order="F")).astype(float)
        radius=max(1,float(matlab_round(np.median(np.sqrt(np.sum((border-center)**2,axis=1))))))
        masks.append(mask);centers.append(center);borders.append(border);radii.append(radius)
    return masks,np.asarray(centers),borders,np.asarray(radii)


def landmark_distances(events,borders,shape,resolution):
    n=len(events);m=len(borders);spatial=int(np.prod(shape[:3]));per_frame=[];max_per_frame=[];average=np.full((n,m),np.nan);minimum=average.copy()
    for i,event in enumerate(events):
        if not len(event):per_frame.append(np.empty((0,0)));max_per_frame.append(np.empty((0,0)));continue
        times=event//spatial;start,end=int(times.min()),int(times.max());distances=np.full((end-start+1,m),np.nan);maximum=distances.copy()
        for k,t in enumerate(range(start,end+1)):
            pixels=event[times==t]%spatial
            if not len(pixels):
                if k:distances[k]=distances[k-1];maximum[k]=maximum[k-1]
                continue
            coords=np.column_stack(np.unravel_index(pixels,shape[:3],order="F"))
            for j,border in enumerate(borders):
                matrix=cdist(border,coords);distances[k,j]=matrix.min();maximum[k,j]=matrix.max(axis=1).min()
        per_frame.append(distances*resolution);max_per_frame.append(maximum*resolution)
        average[i]=np.nanmean(distances,axis=0)*resolution;minimum[i]=np.nanmin(distances,axis=0)*resolution
    return {"distPerFrame":_cell(per_frame),"distMaxPerFrame":_cell(max_per_frame),"distAvg":average,"distMin":minimum}


@njit(cache=True)
def _chessboard_geodesic(support,seeds):
    # bwdistgeodesic's default is chessboard, not quasi-Euclidean.
    h,w=support.shape;distance=np.full((h,w),np.inf,np.float32);queue=np.empty(h*w,np.int64);head=0;tail=0
    for col in range(w):
        for row in range(h):
            if not support[row,col]:distance[row,col]=np.nan
            elif seeds[row,col]:distance[row,col]=0;queue[tail]=row+col*h;tail+=1
    while head<tail:
        p=queue[head];head+=1;row=p%h;col=p//h
        for dh in range(-1,2):
            for dw in range(-1,2):
                r,c=row+dh,col+dw
                if 0<=r<h and 0<=c<w and support[r,c] and np.isinf(distance[r,c]):
                    distance[r,c]=distance[row,col]+1;queue[tail]=r+c*h;tail+=1
    return distance


@njit(cache=True)
def _frontier_changes(current,previous,weights,distance_maps,single_precision,h):
    n=len(current);landmarks=distance_maps.shape[0];toward=np.zeros((n,landmarks));away=np.zeros_like(toward)
    for i in range(n):
        r,c=current[i]%h,current[i]//h;best=np.inf;winner=0;length=0.0
        for j in range(len(previous)):
            rr,cc=previous[j]%h,previous[j]//h;distance=np.sqrt(float((r-rr)**2+(c-cc)**2));weighted=distance*weights[j]
            if weighted<best:best=weighted;winner=j;length=distance
        rr,cc=previous[winner]%h,previous[winner]//h
        count=max(1,int(np.floor(length+.5)))
        for l in range(landmarks):
            low=np.inf
            for step in range(count+1):
                row=int(np.floor(r+(rr-r)*step/count+.5));col=int(np.floor(c+(cc-c)*step/count+.5));value=distance_maps[l,row,col]
                if not np.isnan(value) and value<low:low=value
            a=distance_maps[l,rr,cc]-low;b=distance_maps[l,r,c]-low
            if single_precision[l]:a=np.float32(a);b=np.float32(b)
            toward[i,l]=0.0 if np.isnan(a) or a<0 else a
            away[i,l]=0.0 if np.isnan(b) or b<0 else b
    return toward,away


def landmark_frontier(data,masks,thresholds):
    h,w,time=data.shape;scale=min(np.sqrt(10000/h/w),1)
    data=bicubic_resize(data,scale);masks=[bicubic_resize(np.asarray(mask,float),scale)>0 for mask in masks]
    h,w,time=data.shape;n=len(masks);frames_toward=np.zeros((len(thresholds),time,n));frames_away=np.zeros_like(frames_toward)
    before=np.zeros(n);after=np.zeros(n);pixels_toward=np.zeros((h*w,n));pixels_away=np.zeros_like(pixels_toward)
    for k,threshold in enumerate(thresholds):
        active=data>threshold;times=np.flatnonzero(active.any(axis=(0,1)))
        if len(times)<2:continue
        start,end=int(times[0]),int(times[-1]);support=active.any(axis=2)
        maps=[];single=[]
        for mask in masks:
            if np.any(mask&support):maps.append(_chessboard_geodesic(support,mask).astype(float));single.append(True)
            else:
                distance=ndimage.distance_transform_edt(~mask).astype(np.float32).astype(float);distance[~support]=np.inf;maps.append(distance);single.append(False)
        distance_maps=np.asarray(maps);single=np.asarray(single,bool)
        groups=[];labels=[];borders=[]
        for t in range(start,end+1):
            cc=_components(active[:,:,t]);label=np.zeros((h,w),int,order="F")
            for j,pixels in enumerate(cc):label.ravel(order="F")[pixels]=j+1
            groups.append(cc);labels.append(label);borders.append(boundaries(active[:,:,t],8,False))
        reached=np.full(n,-1,int);all_toward=np.zeros((end-start+1,n));all_away=np.zeros_like(all_toward)
        for t in range(1,end-start+1):
            for j,mask in enumerate(masks):
                if reached[j]<0 and labels[t][mask].any():reached[j]=t
            previous_labels=labels[t-1].ravel(order="F")
            for current in groups[t]:
                neighbors=np.setdiff1d(previous_labels[current],0)
                previous=[];weights=[]
                for label in neighbors:
                    size=len(groups[t-1][label-1])
                    if size>len(current)/5:
                        boundary=borders[t-1][label-1];pixels=boundary[:,0]+h*boundary[:,1]
                        previous.extend(pixels);weights.extend(np.full(len(pixels),1/size))
                current=current[previous_labels[current]==0]
                if not len(current) or not len(previous):continue
                toward,away=_frontier_changes(current,np.asarray(previous,np.int64),np.asarray(weights),distance_maps,single,h)
                pixels_toward[current]+=toward;pixels_away[current]+=away
                all_toward[t]+=toward.sum(axis=0);all_away[t]+=away.sum(axis=0)
                # Source assigns here (rather than +=), so the final component
                # owns these per-frame values while pixel maps accumulate all.
                frames_toward[k,start+t]=toward.sum(axis=0);frames_away[k,start+t]=away.sum(axis=0)
        for j,t in enumerate(reached):
            if t>=0:before[j]+=all_toward[:t+1,j].sum();after[j]+=all_away[min(t+1,len(all_away)-1):,j].sum()
    factor=scale**3;frames_toward/=factor;frames_away/=factor
    return {"chgTowardThrFrame":frames_toward,"chgAwayThrFrame":frames_away,
        "chgTowardThr":frames_toward.sum(axis=1),"chgAwayThr":frames_away.sum(axis=1),
        "chgToward":frames_toward.sum(axis=(0,1)),"chgAway":frames_away.sum(axis=(0,1)),
        "chgTowardBefReach":before/factor,"chgAwayAftReach":after/factor,
        "pixelToward":pixels_toward.reshape((h,w,n),order="F")/factor,"pixelAway":pixels_away.reshape((h,w,n),order="F")/factor}


def landmark_direction_2d(reconstruction,events,landmarks,resolution,minimum_threshold):
    h,w,_,time=reconstruction.shape;n=len(events);m=len(landmarks);thresholds=minimum_threshold+.1*np.arange(int(np.floor((.9-minimum_threshold)/.1+1e-12))+1)
    output={key:np.zeros((n,m)) for key in ("chgToward","chgAway","chgTowardBefReach","chgAwayAftReach")}
    output.update({key:np.zeros((n,len(thresholds),m)) for key in ("chgTowardThr","chgAwayThr")})
    cells={key:[] for key in ("pixelToward","pixelAway","chgTowardThrFrame","chgAwayThrFrame")}
    landmark_coords=[np.unravel_index(p,(h,w),order="F") for p in landmarks]
    for event in events:
        if not len(event):
            for values in cells.values():values.append(np.empty((0,0)))
            continue
        coords=np.unravel_index(event,(h,w,time),order="F");origin=(max(0,int(coords[0].min())-2),max(0,int(coords[1].min())-2),int(coords[2].min()))
        end=(min(h,int(coords[0].max())+3),min(w,int(coords[1].max())+3),int(coords[2].max())+1);shape=tuple(b-a for a,b in zip(origin,end))
        patch=reconstruction[origin[0]:end[0],origin[1]:end[1],0,origin[2]:end[2]].astype(float)/255
        mask=np.zeros(shape,bool);mask[tuple(a-b for a,b in zip(coords,origin))]=True;patch*=mask
        projected=[]
        for rows,cols in landmark_coords:
            rows=np.clip(rows-origin[0],0,shape[0]-1);cols=np.clip(cols-origin[1],0,shape[1]-1)
            mask=np.zeros(shape[:2],bool)
            # Source msk0(h1ks,w1ks)=1 is a Cartesian product, not paired pixels.
            mask[np.ix_(np.unique(rows),np.unique(cols))]=True;projected.append(mask)
        result=landmark_frontier(patch,projected,thresholds);i=len(cells["pixelToward"])
        for key in output:output[key][i]=result[key]*resolution**3
        for key,values in cells.items():values.append(result[key]*(resolution**2 if key.startswith("pixel") else resolution**3))
    output.update({key:_cell(value) for key,value in cells.items()})
    return output


def landmark_direction_3d(reconstruction,events,landmarks,resolution,minimum_threshold):
    shape=reconstruction.shape;n=len(events);m=len(landmarks);minimum=np.empty((n,m),object);maximum=np.empty((n,m),object)
    thresholds=minimum_threshold+.1*np.arange(int(np.floor((.9-minimum_threshold)/.1+1e-12))+1)
    for i,event in enumerate(events):
        for j,landmark in enumerate(landmarks):
            if not len(event):minimum[i,j]=np.empty((0,0));maximum[i,j]=np.empty((0,0));continue
            coords=np.unravel_index(event,shape,order="F");lcoords=np.unravel_index(landmark,shape[:3],order="F")
            origin=tuple(max(0,min(int(a.min()),int(b.min()))-2) for a,b in zip(coords[:3],lcoords))+(int(coords[3].min()),)
            end=tuple(min(size,max(int(a.max()),int(b.max()))+3) for a,b,size in zip(coords[:3],lcoords,shape[:3]))+(int(coords[3].max())+1,)
            slices=tuple(slice(a,b) for a,b in zip(origin,end));patch=reconstruction[slices].astype(float)/255
            active=np.zeros(patch.shape,bool);active[tuple(a-b for a,b in zip(coords,origin))]=True;patch*=active
            mask=np.zeros(patch.shape[:3],bool);mask[tuple(a-b for a,b in zip(lcoords,origin))]=True
            scale=max(1,int(matlab_round(np.sqrt(patch.shape[0]*patch.shape[1]/10000))))
            patch=bicubic_resize(patch,1/scale);mask=bicubic_resize(mask,1/scale)
            border=mask&~ndimage.binary_erosion(mask,structure=ndimage.generate_binary_structure(3,1));land=np.column_stack(np.nonzero(border))
            lo=np.full((patch.shape[3],len(thresholds)),np.nan);hi=lo.copy()
            for k,threshold in enumerate(thresholds):
                binary=patch>threshold
                for t in range(patch.shape[3]):
                    current=binary[...,t];boundary=current&~ndimage.binary_erosion(current,structure=ndimage.generate_binary_structure(3,1))
                    pixels=np.column_stack(np.nonzero(boundary))
                    if not len(pixels):continue
                    outside=pixels[~mask[tuple(pixels.T)]]
                    if len(outside):
                        delta=(outside[:,None,:]-land[None,:,:])*np.array([scale,scale,1]);distance=np.sqrt(np.sum(delta**2,axis=2));low,high=distance.min(),distance.max()
                    else:low,high=np.inf,0
                    # The source overwrites its tentative zero for overlaps.
                    lo[t,k]=low*resolution;hi[t,k]=high*resolution
            minimum[i,j]=lo;maximum[i,j]=hi
    return {"minDistToLandMark":minimum,"maxDistToLandMark":maximum}


def region_features(events,reconstruction,regions,landmarks,resolution,minimum_threshold,region_names=None,landmark_names=None):
    shape=reconstruction.shape;spatial_shape=shape[:3];output={}
    if landmarks:
        masks,centers,borders,radii=polygon_info(landmarks,spatial_shape)
        output["landMark"]={"mask":_cell(masks),"center":centers,"border":_cell(borders),"centerBorderAvgDist":(radii*resolution).reshape(-1,1)}
        output["landmarkDist"]=landmark_distances(events,borders,shape,resolution)
        output["landmarkDir"]=(landmark_direction_2d if shape[2]==1 else landmark_direction_3d)(reconstruction,events,landmarks,resolution,minimum_threshold)
        if landmark_names is not None:output["landMark"]["name"]=_cell([np.array([name]) for name in landmark_names])
        landmark_masks=masks
    else:
        output.update({key:np.empty((0,0)) for key in ("landMark","landmarkDist","landmarkDir")});landmark_masks=[]
    if regions:
        masks,centers,borders,radii=polygon_info(regions,spatial_shape)
        inclusion=np.full((len(regions),len(landmarks)),np.nan) if landmarks else np.empty((0,0))
        for i,mask in enumerate(masks):
            for j,land in enumerate(landmark_masks):
                if np.any(mask&land):inclusion[i,j]=1
        membership=np.full((len(events),len(regions)),np.nan);distance=membership.copy();normalized=membership.copy()
        for i,event in enumerate(events):
            if not len(event):continue
            coords=np.column_stack(np.unravel_index(event,shape,order="F"))[:,:3];center=matlab_round(coords.mean(axis=0)+1)-1
            pixels=event%int(np.prod(spatial_shape))
            for j,mask in enumerate(masks):
                if mask.ravel(order="F")[pixels].any():
                    membership[i,j]=1;distance[i,j]=np.sqrt(np.sum((borders[j]-center)**2,axis=1)).min();normalized[i,j]=distance[i,j]/radii[j]
        output["cell"]={"mask":_cell(masks),"center":centers,"border":_cell(borders),"centerBorderAvgDist":(radii*resolution).reshape(-1,1),
            "incluLmk":inclusion,"memberIdx":membership,"dist2border":distance*resolution,"dist2borderNorm":normalized}
        if region_names is not None:output["cell"]["name"]=_cell([np.array([name]) for name in region_names])
    else:output["cell"]=np.empty((0,0))
    return output


def region_mat_payload(region):
    if not isinstance(region,dict):return region
    def convert(value):
        if isinstance(value,dict):return {key:convert(v) for key,v in value.items()}
        if isinstance(value,np.ndarray) and value.dtype==object:
            result=np.empty(value.shape,object)
            for i,v in enumerate(value.ravel()):result.ravel()[i]=convert(v)
            return result
        array=np.asarray(value)
        while array.ndim>2 and array.shape[-1]==1:array=array[...,0]
        return array
    result=convert(region)
    for key in ("cell","landMark"):
        if isinstance(result.get(key),dict):
            result[key]["center"]=result[key]["center"]+1
            for i,border in enumerate(result[key]["border"].ravel()):result[key]["border"].ravel()[i]=border+1
    return result
