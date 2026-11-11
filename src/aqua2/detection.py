"""Headless input -> local events -> optional global-event detection."""
import numpy as np
from pathlib import Path
from .pipeline import preprocess,scalar_options
from .se import temporal_from_stage3
from .evt import se_to_events_top,spatial_mat_payload


def detect_from_stage3(stage3,*,spatial_mask=None,progress=None,with_features=True,regions=None,landmarks=None,region_names=None,landmark_names=None):
    opts=scalar_options(stage3["opts"]);temporal=temporal_from_stage3(stage3,progress=progress)
    result={"opts":opts}
    for channel in (1,2):
        key=f"channel{channel}"
        if key not in temporal:continue
        delta=stage3[f"dF{channel}"];original=stage3[f"datOrg{channel}"]
        if opts.get("needSpa",True):local=se_to_events_top(delta,temporal[key],opts,progress=progress)
        else:local={"evt":temporal[key]["seLst"],"datR":np.full(delta.shape,255,np.uint8),"riseLst":[],"opts":opts}
        global_result={"evt":[],"datR":np.empty((0,0)),"riseLst":[],"opts":opts}
        if opts.get("detectGlo",False):
            if progress:progress(f"Channel {channel} · Detecting global events",None)
            from .glo import global_detect
            global_result=global_detect(delta,original,local["evt"],opts,spatial_mask,progress=progress)
        if with_features:
            if progress:progress(f"Channel {channel} · Extracting features",None)
            from .features import compute_features
            kwargs=dict(channel=channel,regions=regions,landmarks=landmarks,region_names=region_names,landmark_names=landmark_names)
            local["features"]=compute_features(original,local,opts,**kwargs)
            if len(global_result["evt"]):global_result["features"]=compute_features(original,global_result,opts,**kwargs)
        result[key]={"local":local,"global":global_result}
    return result


def detect(source,options=None,source2=None,*,progress=None,with_features=True,regions=None,landmarks=None,region_names=None,landmark_names=None,**input_options):
    stage3=preprocess(source,options,source2,**input_options)
    mask=input_options.get("spatial_mask")
    if isinstance(mask,(str,Path)):
        from .io import read_working_mask
        mask=read_working_mask(mask,stage3["dF1"].shape[:3],stage3["opts"].get("regMaskGap",0))
    return detect_from_stage3(stage3,spatial_mask=mask,progress=progress,with_features=with_features,regions=regions,landmarks=landmarks,region_names=region_names,landmark_names=landmark_names)


def detection_mat_payload(result):
    payload={"opts":result["opts"]}
    for channel in (1,2):
        key=f"channel{channel}"
        if key in result:
            payload[key]={name:spatial_mat_payload(result[key][name]) for name in ("local","global")}
            from .fea import features_mat_payload
            for name in ("local","global"):
                if "features" in result[key][name]:payload[key][name]["features"]=features_mat_payload(result[key][name]["features"])
    return payload
