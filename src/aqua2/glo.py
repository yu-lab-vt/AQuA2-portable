"""Global-event input reconstruction and reuse of the local detection chain.

Independent global/3D/dual-channel acceptance is deferred to phase 9.
"""
import numpy as np
from numba import njit
from .act import ac_detect
from .se import temporal_from_stage3
from .evt import se_to_events_top


@njit(cache=True)
def _interpolate_removed(data):
    for pixel in range(data.shape[0]):
        start=0
        while start<data.shape[1]:
            if not np.isnan(data[pixel,start]):start+=1;continue
            end=start
            while end+1<data.shape[1] and np.isnan(data[pixel,end+1]):end+=1
            if start==0 and end==data.shape[1]-1:
                raise ValueError("A fully removed pixel has no interpolation anchor (MATLAB source also cannot interpolate it)")
            if start==0:
                for t in range(end+1):data[pixel,t]=data[pixel,end+1]
            elif end==data.shape[1]-1:
                for t in range(start,end+1):data[pixel,t]=data[pixel,start-1]
            else:
                previous=data[pixel,start-1];following=data[pixel,end+1]
                slope=np.float32(np.float32(following-previous)/np.float32(end-start+2))
                for t in range(start,end+1):data[pixel,t]=np.float32(previous+np.float32(slope*np.float32(t-start+1)))
            start=end+1
    return data


def remove_detected(delta,events):
    labels=np.zeros(delta.shape,np.uint16,order="F")
    for i,event in enumerate(events):labels.ravel(order="F")[event]=min(i+1,65535)
    select=(labels[...,:-1]>0)&(labels[...,1:]>0)&(labels[...,:-1]!=labels[...,1:])
    result=np.array(delta,np.float32,order="F",copy=True);result[labels>0]=np.nan
    result[...,:-1][select]=delta[...,:-1][select]
    return np.array(_interpolate_removed(result.reshape((-1,delta.shape[-1]),order="F")).reshape(delta.shape,order="F"),order="F")


def global_detect(delta,original,local_events,options,spatial_mask=None,*,progress=None):
    opts={**options,"minDur":options["gloDur"]}
    residual=remove_detected(delta,local_events)
    # Source intentionally passes channel=1 for both channels here.
    active=ac_detect(residual,opts,spatial_mask,channel=1)
    temporal=temporal_from_stage3(dict(dF1=residual,datOrg1=original,activeRegions1=active,opts=opts),progress=progress)["channel1"]
    if opts.get("needSpa",True):result=se_to_events_top(residual,temporal,opts,progress=progress)
    else:result={"evt":temporal["seLst"],"datR":np.empty((0,0)),"riseLst":[],"opts":opts}
    return result
