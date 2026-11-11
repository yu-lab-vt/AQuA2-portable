"""Headless feature pipeline over local or global spatial event results."""
import numpy as np
from .pipeline import scalar_options
from .fea import get_features
from .fea_propagation import add_propagation
from .fea_network import event_network
from .fea_region import mask_pixels,region_features


def compute_features(original,spatial,options,*,channel=None,regions=None,landmarks=None,
                     region_names=None,landmark_names=None,filter_mask=None):
    options=scalar_options(options);options=dict(options)
    for key in ("stdMapOrg","maxValueDat","minValueDat","tempVarOrg","correctPars"):
        channel_key=key+str(channel if channel is not None else 1)
        if channel_key in options:options[key]=options[channel_key]
    events=spatial["evt"];result,_=get_features(original,events,options);features=result["fts"]
    if channel is not None:features["channel"]=channel
    if options.get("propMetric",False) and len(events):add_propagation(features,spatial["datR"],events,options)
    shape=original.shape;regions=mask_pixels(regions,shape[:3]);landmarks=mask_pixels(landmarks,shape[:3])
    def names(value,n):
        if value is None:return [str(i+1) for i in range(n)]
        if len(value)!=n:raise ValueError("Mask name count does not match mask count")
        return [str(i+1) if name=="None" else str(name) for i,name in enumerate(value)]
    filtered=events
    if filter_mask is not None and len(filter_mask)==len(events):
        filtered=[event if keep else np.empty(0,np.int64) for event,keep in zip(events,filter_mask)]
    features["region"]=np.empty((0,0))
    if regions or landmarks:
        features["region"]=region_features(filtered,spatial["datR"],regions,landmarks,options["spatialRes"],options["minShow1"],names(region_names,len(regions)),names(landmark_names,len(landmarks)))
    if options.get("networkFeatures",False):
        features["networkAll"]=event_network(filtered,shape)
        inside=filtered
        if regions:
            allowed=np.zeros(shape[:3],bool,order="F")
            for pixels in regions:allowed.ravel(order="F")[pixels]=True
            size=int(np.prod(shape[:3]));inside=[e if len(e) and allowed.ravel(order="F")[e%size].any() else np.empty(0,np.int64) for e in filtered]
        features["network"]=event_network(inside,shape)
    result["opts"]=options
    return result


def masks_from_mat(path,field):
    """Explicit MAT mask-list boundary: cells contain one-based linear pixels."""
    from .io import read_mat
    payload=read_mat(path,[field]);values=payload[field]
    if values.dtype!=object:
        if values.dtype==bool:return [values]
        raise ValueError("MAT mask lists must be logical arrays or cells of one-based index arrays")
    masks=[]
    for value in values.ravel(order="F"):
        value=np.asarray(value)
        if value.dtype==bool:masks.append(value)
        else:
            if not np.isfinite(value).all() or not np.equal(value,np.floor(value)).all() or np.any(value<1):raise ValueError("MAT mask indices must be positive integers")
            masks.append(value.astype(np.int64).ravel()-1)
    return masks


def mask_options_from_files(regions=None,landmarks=None):
    from .io import read_mat
    output={}
    for path,field,name_field,option in ((regions,"regions","regionNames","region_names"),(landmarks,"landmarks","landmarkNames","landmark_names")):
        if path is None:continue
        output[field]=masks_from_mat(path,field)
        try:
            names=read_mat(path,[name_field])[name_field]
        except KeyError:continue
        output[option]=[str(np.asarray(value).item()) for value in names.ravel(order="F")]
    return output
