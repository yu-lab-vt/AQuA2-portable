"""Stage 4 temporal segmentation, exposing MATLAB-style field names.

Event-list indices, majority coordinates/windows and seLabel are zero based.
Checkpoint seedMap uses 0 for background and positive labels, as in MATLAB.
Delays are signed frame counts; the MAT boundary changes index origins.
"""
import numpy as np

from .pipeline import scalar_options
from .se_seed import seed_detect
from .se_partition import marker_controlled_splitting,major_curve_filter,grow_watershed_result
from .se_majority import get_majority,create_merging_info,merge_super_events


def se_detection(delta,original,active_regions,options,*,progress=None,checkpoints=False):
    """Run the current +se call chain on normalized single-precision HWLT data.

    active_regions contains zero-based F-order voxel arrays. progress receives
    (stage_name, fraction_or_None); None marks an unmeasured step. This API
    does not repeat input preprocessing.
    """
    delta=np.asarray(delta,dtype=np.float32,order="F")
    original=np.asarray(original,dtype=np.float32,order="F")
    if delta.shape!=original.shape or delta.ndim!=4:raise ValueError("Temporal segmentation requires matching HWLT arrays")
    opts=scalar_options(options);opts.setdefault("step",.5);opts.setdefault("overlap",.5)
    seed_map,regions,seed_trace=seed_detect(delta,original,active_regions,opts,progress)
    events,seeds,regions,score=marker_controlled_splitting(seed_map,regions,delta,opts,progress)
    good=[i for i,event in enumerate(events) if len(event)]
    seeds,events=[seeds[i] for i in good],[events[i] for i in good]
    if progress:progress("majority",None)
    majority=get_majority(seeds,events,delta,opts)
    trace={"seedMap":seed_map,"initialEvents":events,"initialSeeds":seeds,"majority":majority} if checkpoints else None
    if opts.get("needRefine",False):
        if progress:progress("refine",None)
        from .se_refine import split_events
        events,majority,seeds=split_events(delta,original,events,seeds,majority,opts)
    if opts.get("needGrow",False):
        if progress:progress("grow",None)
        events,regions=grow_watershed_result(events,majority,delta,opts)
    if progress:progress("curve_filter",None)
    seeds,events,majority=major_curve_filter(original,delta,seeds,events,majority,opts,score)
    if progress:progress("merge_info",None)
    info=create_merging_info(events,majority,regions,delta.shape,opts)
    if progress:progress("merge",None)
    super_events,labels,info=merge_super_events(events,majority,info,delta.shape,regions,opts)
    result={"seLst":super_events,"subEvtLst":events,"seLabel":labels,"majorInfo":majority,
        "sdLst":seeds,"mergingInfo":info,"ccRegions":regions,"opts":opts}
    if checkpoints:result["checkpoints"]=trace
    return result


def _cells(values,column=True,dtype=None,origin=0):
    result=np.empty((len(values),1) if column else (1,len(values)),dtype=object)
    for i,value in enumerate(values):
        result.ravel()[i]=np.asarray(value+origin,dtype=dtype).reshape(-1,1)
    return result


def temporal_mat_payload(result):
    """Serialize indices explicitly, including the scientific Map dictionaries."""
    output={"seLst":_cells(result["seLst"],dtype=np.float64,origin=1),
        "subEvtLst":_cells(result["subEvtLst"],column=False,dtype=np.float64,origin=1),
        "sdLst":_cells(result["sdLst"],column=False,dtype=np.float64,origin=1),
        "ccRegions":_cells(result["ccRegions"],column=False,dtype=np.float64,origin=1),
        "seLabel":(result["seLabel"]+1).astype(np.float64).reshape(1,-1)}
    majors=np.empty((len(result["majorInfo"]),1),dtype=object)
    for i,major in enumerate(result["majorInfo"]):
        majors[i,0]={"ihw":(major["ihw"]+1).astype(np.float64).reshape(-1,1),
            "TW":(major["TW"]+1).astype(np.float64).reshape(1,-1),
            "delays":major["delays"].reshape(-1,1),"curve":major["curve"].reshape(1,-1)}
    output["majorInfo"]=majors
    info=result["mergingInfo"]
    maps=np.empty((len(info["delayDif"]),1),dtype=object)
    for i,mapping in enumerate(info["delayDif"]):
        keys=np.empty((1,len(mapping)),dtype=object);values=np.empty_like(keys)
        for j,key in enumerate(sorted(mapping)):
            keys[0,j]=np.array([[float(key+1)]]);values[0,j]=np.array([[float(mapping[key])]])
        maps[i,0]={"matlabType":np.array(["containers.Map"]),"keyType":np.array(["double"]),"valueType":np.array(["double"]),"mapKeys":keys,"mapValues":values}
    output["mergingInfo"]={"neibLst":_cells(info["neibLst"],dtype=np.uint16,origin=1),"delayDif":maps,
        "evtCCLabel":(info["evtCCLabel"]+1).astype(float).reshape(-1,1),
        "labelsInActRegs":_cells(info["labelsInActRegs"],dtype=np.uint16,origin=1)}
    return output


def temporal_from_stage3(stage3,*,progress=None):
    result={}
    for channel in (1,2):
        if f"dF{channel}" not in stage3:continue
        opts=scalar_options(stage3["opts"])
        if opts.get("needTemp",True):
            result[f"channel{channel}"]=se_detection(stage3[f"dF{channel}"],stage3[f"datOrg{channel}"],
                stage3[f"activeRegions{channel}"],opts,progress=progress)
        else:
            regions=stage3[f"activeRegions{channel}"]
            major=get_majority(regions,regions,stage3[f"dF{channel}"],opts)
            result[f"channel{channel}"]={"seLst":regions,"subEvtLst":regions,"seLabel":np.arange(len(regions)),
                "majorInfo":major,"sdLst":regions,"ccRegions":regions,
                "mergingInfo":create_merging_info(regions,major,regions,stage3[f"dF{channel}"].shape,opts),"opts":opts}
    return result


def read_stage3_payload(path):
    from .io import read_mat
    from .compat import from_matlab_indices
    result=read_mat(path)
    for channel in (1,2):
        key=f"activeRegions{channel}"
        if key in result:
            size=result[f"dF{channel}"].size
            result[key]=[from_matlab_indices(cell.ravel(),size) for cell in result[key].ravel(order="F")]
    result["opts"]=scalar_options(result["opts"])
    return result


def temporal_from_mat_payload(payload,options=None):
    """Restore the scientific fields needed by spatial segmentation from MAT."""
    result={}
    for key in ("seLst","subEvtLst","sdLst","ccRegions"):
        if key in payload:result[key]=[np.asarray(v).astype(np.int64).ravel()-1 for v in payload[key].ravel(order="F")]
    result["seLabel"]=payload["seLabel"].astype(np.int64).ravel()-1
    result["majorInfo"]=[{"ihw":m["ihw"].astype(np.int64).ravel()-1,
        "TW":m["TW"].astype(np.int64).ravel()-1,"delays":m["delays"].ravel(),"curve":m["curve"].ravel()}
        for m in payload["majorInfo"].ravel(order="F")]
    if options is not None:result["opts"]=scalar_options(options)
    return result
