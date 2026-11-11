"""Directional growth/shrinkage and symmetric boundary displacement."""
import warnings
import numpy as np
from scipy.spatial import cKDTree
from .fea import native_sum
from .fea_geometry import boundaries,close_mask
from .se_common import matlab_round,mean_spatial


def _thresholds(options):
    lo,hi,step=options["propthrmin"],options["propthrmax"],options["propthrstep"]
    if lo==hi or step==0:return np.array([lo]),True
    return lo+step*np.arange(int(np.floor((hi-lo)/step+1e-12))+1),False


def propagation_for_event(volume,resolution,options,north=None):
    thresholds,single=_thresholds(options);dimension=2 if volume.shape[2]==1 else 3;time=volume.shape[3]
    if dimension==2:volume=volume[::-1,:,0,:].copy()
    else:volume=volume.copy()
    n_directions=4 if dimension==2 else 6
    if dimension==3 and not volume.any():
        return {"propGrow":np.zeros((time,6)),"propGrowOverall":np.zeros(6),"propShrink":np.zeros((time,6)),"propShrinkOverall":np.zeros(6),
            "areaChange":np.zeros((time,1)),"areaChangeRate":np.zeros((time,1)),"areaFrame":np.zeros((time,1))}
    volume[volume<thresholds.min()]=0;valid=volume>=thresholds.min()
    pixel_count=np.any(valid,axis=-1).sum() if dimension==2 else valid.sum()
    times=np.flatnonzero(valid.reshape((-1,time)).any(axis=0))
    if not len(times):raise ValueError("No reconstructed sample reaches the propagation threshold")
    start,end=int(times[0]),int(times[-1]);initial=volume[...,start]
    pixels=np.flatnonzero((initial>=thresholds.min()).ravel(order="F"));coords=np.unravel_index(pixels,initial.shape,order="F");weights=initial.ravel(order="F")[pixels]
    center=np.array([np.float32(native_sum((a+1).astype(np.float32)*weights))/np.float32(native_sum(weights)) for a in coords],np.float32)
    grid=np.indices(initial.shape,dtype=float)+1
    if dimension==2:
        a,b=north;h0,w0=matlab_round(center)
        with np.errstate(divide="ignore",invalid="ignore"):
            row,column=grid
            first=(-np.divide(a,b)*(column-w0)).astype(np.float32)+np.float32(h0)
            second=(np.divide(b,a)*(column-w0)).astype(np.float32)+np.float32(h0)
            masks=[row>first,row<first,row>second,row<second]
        directions=np.array([[a,b],[-a,-b],[-b,a],[b,-a]])
    else:
        if not np.equal(center,np.floor(center)).all():
            raise ValueError("MATLAB 3D propagation uses fractional centroid as an array index; source behavior needs review in step 9")
        masks=[grid[1]>=center[1],grid[1]<center[1],grid[0]>=center[0],grid[0]<center[0],grid[2]>=center[2],grid[2]<center[2]]
        directions=np.array([[0,1,0],[0,-1,0],[1,0,0],[-1,0,0],[0,0,1],[0,0,-1]])
    distances=np.full((time,n_directions,len(thresholds)),np.nan);counts=np.zeros((time,len(thresholds)))
    for t in range(start,end+1):
        for k,threshold in enumerate(thresholds):
            active=volume[...,t]>=threshold;counts[t,k]=active.sum()
            for j,mask in enumerate(masks):
                coordinates=np.nonzero(active&mask)
                if len(coordinates[0])<(4 if dimension==2 else 1):continue
                centroid=np.array([axis.mean()+1 for axis in coordinates])
                delta=(centroid-center.astype(float)).astype(np.float32)
                if dimension==2:delta=delta[::-1]
                distances[t,j,k]=native_sum((delta.astype(float)*directions[j]).astype(np.float32))
    change=np.full_like(distances,np.nan);change[1:]=np.diff(distances,axis=0)
    growing=change.copy();growing[growing<0]=np.nan;shrinking=change.copy();shrinking[shrinking>0]=np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        growing=np.nanmax(growing,axis=2);shrinking=np.nanmax(shrinking,axis=2)
    growing=np.nan_to_num(growing);shrinking=np.nan_to_num(shrinking)
    area_change=np.zeros_like(counts);area_change[1:]=np.diff(counts,axis=0)
    rates=area_change/pixel_count;winner=np.argmax(abs(rates),axis=1);rate=rates[np.arange(time),winner,None]
    result={"propGrow":growing*resolution,"propGrowOverall":growing.sum(axis=0)*resolution,
        "propShrink":shrinking*resolution,"propShrinkOverall":shrinking.sum(axis=0)*resolution,
        "areaChange":area_change*resolution**dimension,"areaChangeRate":rate,"areaFrame":counts*resolution**dimension}
    if dimension==2:
        boundary={};speed=np.zeros((time,len(thresholds)))
        for t in range(max(0,start-1),min(time,end+2)):
            for k,threshold in enumerate(thresholds):
                groups=boundaries(close_mask(volume[...,t]>=threshold));boundary[t,k]=np.vstack(groups) if groups else np.empty((0,2))
        for t in range(max(1,start),end+1):
            for k in range(len(thresholds)):
                previous,current=boundary[t-1,k],boundary[t,k]
                if len(previous) and len(current):speed[t,k]=max(cKDTree(previous).query(current)[0].max(),cKDTree(current).query(previous)[0].max())
        result["propMaxSpeed"]=speed*resolution
        if single:result["maxPropSpeed"]=(speed*resolution).max();result["avgPropSpeed"]=mean_spatial(speed*resolution).item()
    return result


def add_propagation(features,reconstruction,events,options):
    shape=reconstruction.shape;output={};north=np.array([options.get("northx",0),options.get("northy",1)],float)
    with np.errstate(divide="ignore",invalid="ignore"):north=north/np.linalg.norm(north)
    for event in events:
        coords=np.unravel_index(event,shape,order="F");origin=tuple(max(0,int(a.min())-1) for a in coords[:3])+(int(coords[3].min()),)
        end=tuple(min(size,int(a.max())+2) for a,size in zip(coords[:3],shape[:3]))+(int(coords[3].max())+1,)
        local_shape=tuple(b-a for a,b in zip(origin,end));volume=np.zeros(local_shape,np.float32)
        local=tuple(a-b for a,b in zip(coords,origin));volume[local]=reconstruction.ravel(order="F")[event].astype(np.float32)/np.float32(255)
        values=propagation_for_event(volume,options["spatialRes"],options,north)
        for key,value in values.items():output.setdefault(key,[]).append(value)
    for key in ("propGrowOverall","propShrinkOverall"):
        if key in output:output[key]=np.asarray(output[key])
    for key in ("maxPropSpeed","avgPropSpeed"):
        if key in output:output[key]=np.asarray(output[key]).reshape(1,-1)
    features["propagation"]=output
    return features
