"""Current multi-scale accelerated seed detector, independently ported."""
import numpy as np

from .act import _components,bw2reg
from .se_common import my_resize,matlab_round,mean_spatial
from .se_stats import get_seed_score,curve_significance


def normalize_and_resize(original,options,scales=(2,4,8)):
    result=[]
    shape=original.shape
    variance=np.asarray(options["tempVarOrg"]).reshape(shape[:3],order="F")
    correction=np.asarray(options["correctPars"]).reshape(shape[:3],order="F").astype(variance.dtype)
    for scale in scales:
        down=my_resize(original,1/scale)
        current=np.nanmean(np.diff(down,axis=3)**2,axis=3)/np.float32(2)
        before=my_resize(variance,1/scale)
        after=my_resize(variance*np.float32(2)/correction,1/scale)
        with np.errstate(invalid="ignore",divide="ignore"):
            standard=np.sqrt(current*after/before)
            down=down/standard[:,:,:,None]
        result.append(np.array(down,order="F"))
    return result


def seed_detect(delta,original,active_regions,options,progress=None):
    shape=delta.shape; height,width,depth,time=shape
    opts=dict(options);opts["scaleRatios"]=(2,4,8)
    data=delta.copy(order="F")
    if opts["maxValueDat1"]==2**opts["BitDepth"]-1:
        data[original==1]=np.inf
    labels=np.zeros(shape,np.uint16,order="F")
    flat=labels.ravel(order="F");spatial=height*width*depth
    sizes=[]
    for i,region in enumerate(active_regions):
        flat[region]=min(i+1,65535)
        sizes.append(len(np.unique(region%spatial)))
    if progress:progress("seed_prepare",0)
    resized_data=normalize_and_resize(original,opts)
    if progress:progress("seed_prepare",1/3)
    resized_delta=[my_resize(data,1/scale) for scale in opts["scaleRatios"]]
    if progress:progress("seed_prepare",2/3)
    valid=[my_resize(labels,1/scale)>0 for scale in opts["scaleRatios"]]
    if progress:progress("seed_prepare",1)
    scores=np.zeros(shape,np.float32,order="F");score_flat=scores.ravel(order="F")
    maximum=float(opts["maxdF1"]);step=float(opts.get("step",.5));minimum=float(opts["thrARScl"])
    thresholds=maximum-step*np.arange(max(0,int(np.floor((maximum-minimum)/step))+1))
    accepted=[]
    for k,threshold in enumerate(thresholds):
        if progress:progress("seeds",k/max(1,len(thresholds)))
        for j,scale in enumerate(opts["scaleRatios"]):
            reduced_shape=resized_delta[j].shape
            selection=(resized_delta[j]>threshold)&valid[j]
            if not selection.any():continue
            vectors=resized_data[j].reshape((-1,time),order="F")
            minimum_voxels=int(np.floor(opts["minSize"]/scale**2*opts["minDur"]/3))+1
            regions=bw2reg(selection,opts,min_voxels=max(1,minimum_voxels))
            regions=[group for group in regions if len(group)>opts["minSize"]/scale**2*opts["minDur"]/3]
            for region in regions:
                h,w,z,t=np.unravel_index(region,reduced_shape,order="F")
                footprint=np.unique(region%(int(np.prod(reduced_shape[:3]))))
                duration=int(t.max()-t.min()+1)
                block=labels[h[0]*scale:min(height,(h[0]+1)*scale),w[0]*scale:min(width,(w[0]+1)*scale),z[0],t[0]]
                parent=np.unique(block);parent=parent[parent>0]
                if not len(parent):raise RuntimeError("Downsampled seed has no source active region")
                size_limit=max(opts["minSize"],sizes[int(parent[0])-1]*opts["seedSzRatio"])
                if duration<opts["minDur"] or len(footprint)<size_limit/scale**2:continue
                # The MATLAB matrices are flattened column-first: row offset
                # varies first, then column offset, then source voxel.
                offsets_h=np.tile(np.arange(scale),scale)
                offsets_w=np.repeat(np.arange(scale),scale)
                hh=h[:,None]*scale+offsets_h;ww=w[:,None]*scale+offsets_w
                zz=np.broadcast_to(z[:,None],hh.shape);tt=np.broadcast_to(t[:,None],hh.shape)
                good=(hh<height)&(ww<width)
                pixels=np.ravel_multi_index((hh.ravel(order="F")[good.ravel(order="F")],ww.ravel(order="F")[good.ravel(order="F")],
                    zz.ravel(order="F")[good.ravel(order="F")],tt.ravel(order="F")[good.ravel(order="F")]),shape,order="F")
                if len(pixels)<size_limit or np.any(score_flat[pixels]>0):continue
                time_scale=max(1,int(matlab_round(duration/opts["TPatch"])))
                statistics=get_seed_score(region,time_scale,vectors,reduced_shape)
                if min(statistics)>opts["sigThr"]:
                    curve=my_resize(mean_spatial(vectors[footprint,:]),1/time_scale)
                    lo=max(1,int(np.floor((t.min()+1)/time_scale)))-1
                    hi=int(np.ceil((t.max()+1)/time_scale))-1
                    if curve_significance(curve,lo,hi,opts["sigThr"]):
                        score_flat[pixels]=max(min(statistics[:2]),0)
                        accepted.append((k,scale,len(region),*statistics))
    seed_map=np.zeros(shape,np.uint16,order="F")
    seed_flat=seed_map.ravel(order="F")
    for i,pixels in enumerate(_components(scores>0)):
        levels=np.unique(score_flat[pixels])
        if len(levels)>1:pixels=pixels[score_flat[pixels]==levels[-1]]
        seed_flat[pixels]=min(i+1,65535)
    regions=_components((seed_map>0)|(labels>0))
    return seed_map,regions,{"scores":scores,"accepted":accepted}
