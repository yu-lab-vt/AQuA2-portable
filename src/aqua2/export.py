"""Safe built-in feature tables and lossless CSV/numeric result exports.

The original template scripts are mapped explicitly; no MATLAB/Python eval.
"""
import csv
import json
from pathlib import Path
import numpy as np


def feature_table(result,*,channel=None):
    f=result["fts"];n=len(result["dMat"]);names=[];rows=[]
    templates=Path(__file__).with_name("data")/("userFeatures2.csv" if "peri" in f["basic"] else "userFeatures.csv")
    simple={
        "Starting Frame":("curve","tBegin",1),"Basic - Area":("basic","area",0),
        "Basic - Perimeter (only for 2D video)":("basic","peri",0),"Basic - Surface Size (only for 3D video)":("basic","surf",0),
        "Basic - Circularity":("basic","circMetric",0),"Curve - Max Df":("curve","dfMax",0),"Curve - Max Dff":("curve","dffMax",0),
        "Curve - Duration of visualized event overlay":("curve","duration",0),
        "Curve - Duration 50% to 50% based on averge dF/F":("curve","width55",0),"Curve - Duration 10% to 10% based on averge dF/F":("curve","width11",0),
        "Curve - Rising duration 10% to 90% based on averge dF/F":("curve","rise19",0),"Curve - Decaying duration 90% to 10% based on averge dF/F":("curve","fall91",0),
        "Curve - dat AUC":("curve","datAUC",0),"Curve - df AUC":("curve","dfAUC",0),"Curve - dff AUC":("curve","dffAUC",0),"Curve - Decay tau":("curve","decayTau",0)}
    def append(name,values):
        values=np.asarray(values).ravel()
        if values.size==1:values=np.repeat(values,n)
        if values.size!=n:raise ValueError("Feature table length mismatch: "+name)
        names.append(name);rows.append(values)
    with templates.open(encoding="utf-8-sig",newline="") as stream:
        for template in csv.DictReader(stream):
            name=template["Name"]
            if name in simple:
                group,key,offset=simple[name]
                if key in f.get(group,{}):append(name,np.asarray(f[group][key])+offset)
            elif name=="Channel":
                value=f.get("channel",channel)
                if value is not None:append(name,value)
            elif name=="Index":append(name,np.arange(1,n+1))
            elif name=="Curve - P Value on max Dff (-log10)" and "dffMaxPval" in f["curve"]:
                with np.errstate(divide="ignore",invalid="ignore"):append(name,-np.log10(np.asarray(f["curve"]["dffMaxPval"],np.float32)))
            elif name.startswith("Propagation -"):
                key="propGrowOverall" if "onset" in name else "propShrinkOverall"
                if key not in f["propagation"]:continue
                values=np.asarray(f["propagation"][key])
                if "offset" in name:values=abs(values)
                total=values.sum(axis=1)
                if "one direction" in name:
                    for j,direction in enumerate(f["notes"]["propDirectionOrder"]):
                        with np.errstate(divide="ignore",invalid="ignore"):data=values[:,j]/total if "ratio" in name else values[:,j]
                        append(name+" - "+direction,data)
                else:append(name,total)
            elif name.startswith("Landmark -") and isinstance(f.get("region"),dict):
                mapping={"event average distance":("landmarkDist","distAvg"),"event minimum distance":("landmarkDist","distMin"),
                    "event toward landmark":("landmarkDir","chgToward"),"event away from landmark":("landmarkDir","chgAway"),
                    "event toward landmark before reaching":("landmarkDir","chgTowardBefReach"),"event away from landmark after reaching":("landmarkDir","chgAwayAftReach")}
                group,key=mapping[name[len("Landmark - "):]]
                value=f["region"].get(group)
                if isinstance(value,dict) and key in value:
                    for j in range(value[key].shape[1]):append(name+" - landmark "+str(j+1),value[key][:,j])
            elif name.startswith("Region -") and isinstance(f.get("region"),dict):
                cell=f["region"].get("cell")
                if isinstance(cell,dict):
                    key="dist2borderNorm" if "normalized" in name else "dist2border"
                    value=cell[key];minimum=np.array([np.nanmin(row) if np.isfinite(row).any() else np.nan for row in value]);append(name,minimum)
            elif name.startswith("Network -") and isinstance(f.get("network"),dict):
                network=f["network"]
                value=network["nOccurSameTime"] if "same time" in name else network["nOccurSameLoc"][:,1 if "similar" in name else 0]
                append(name,value)
    return {"names":names,"values":rows,"event_count":n}


def export_features(directory,result,*,prefix="events",channel=None,rising_maps=None,include_curves=True):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);table=feature_table(result,channel=channel)
    path=directory/(prefix+"_features.csv")
    def write_table(path,table,selection=None):
        selected=np.arange(table["event_count"]) if selection is None else np.asarray(selection,int)
        with path.open("w",encoding="utf-8-sig",newline="") as stream:
            writer=csv.writer(stream);writer.writerow(["Feature"]+["Event "+str(i+1) for i in selected])
            for name,row in zip(table["names"],table["values"]):writer.writerow([name]+[float(row[i]) for i in selected])
    write_table(path,table)
    if include_curves:
        with (directory/(prefix+"_curves.csv")).open("w",encoding="utf-8-sig",newline="") as stream:
            writer=csv.writer(stream);writer.writerow(["Event ID","10% Rise time"]+["df/f0 at "+("+" if i>0 else "")+str(i) for i in range(-10,101)])
            for i,row in enumerate(result["dffAlignedMat"]):writer.writerow(["Event "+str(i+1),float(result["fts"]["curve"]["dff1Begin"][i])+1]+[float(v) for v in row])
        np.savez_compressed(directory/(prefix+"_curves.npz"),dMat=result["dMat"],dffMat=result["dffMat"],dffAlignedMat=result["dffAlignedMat"])
    cell=result["fts"].get("region",{})
    if isinstance(cell,dict) and isinstance(cell.get("cell"),dict):
        cell=cell["cell"];folder=directory/"regions";folder.mkdir(exist_ok=True)
        for j in range(cell["memberIdx"].shape[1]):
            selected=np.flatnonzero(cell["memberIdx"][:,j]>0)
            if len(selected):write_table(folder/(prefix+"_region_"+str(j+1)+".csv"),table,selected)
    if rising_maps:
        folder=directory/"rising_maps";folder.mkdir(exist_ok=True)
        for i,record in enumerate(rising_maps):np.savez_compressed(folder/(prefix+"_event_"+str(i+1)+".npz"),**record)
    metadata={"schema_version":1,"csv_event_and_frame_origin":1,"npz_rising_coordinate_origin":0,
        "seconds_per_frame":result["opts"]["frameRate"],"spatial_resolution":result["opts"]["spatialRes"],
        "curve_axes":"event,time,original_or_other_events_removed","aligned_offsets_frames":[-10,100],
        "units":{"dff":"dimensionless","df/raw":"stored intensity","curve_durations_and_tau":"seconds","riseTime":"frames",
            "AUC":"sum of samples, no frame-duration factor","propMaxSpeed":"spatial units per frame (source convention)",
            "landmark_frontier_2D_totals":"source spatialRes^3 convention"},
        "curves_exported":bool(include_curves),
        "renderer_note":"Numeric rising maps exported; use aqua2 render or gui for visualization"}
    if not include_curves:metadata['missing_curves_reason']='Historical result did not save full or aligned curves; feature values remain available.'
    region=result["fts"].get("region")
    if isinstance(region,dict):
        for group,key in (("cell","region_names"),("landMark","landmark_names")):
            if isinstance(region.get(group),dict) and "name" in region[group]:metadata[key]=[str(np.asarray(v).item()) for v in region[group]["name"].ravel()]
    (directory/(prefix+"_metadata.json")).write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return path
