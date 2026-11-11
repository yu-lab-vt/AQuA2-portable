"""Curve and basic event features; public coordinate/frame indices are zero based."""
import numpy as np
from numba import njit
from scipy import optimize,special
from skimage.measure import marching_cubes,mesh_surface_area
from .pipeline import scalar_options
from .pre import avg_dat_noise_est,obtain_bias
from .util import impute_mov
from .se_common import mean_spatial,matlab_round
from .fea_geometry import perimeter,boundaries,close_mask


@njit(cache=True)
def native_sum(values):
    values=values.ravel()
    accum=np.zeros(4,dtype=values.dtype)
    if len(values)<44:
        for value in values:accum[0]+=value
    else:
        for i,value in enumerate(values):accum[i%4]+=value
    total=accum[0]
    for j in range(1,4):total+=accum[j]
    return total


@njit(cache=True)
def indexed_spatial_mean(values):
    """R2024b indexed-mean reduction uses blocks of 4096 selected pixels.

    This differs from mean of a pre-existing dense matrix for large regions;
    original/clean curve oracles independently cover the distinction.
    """
    output=np.empty(values.shape[1],np.float32)
    for column in range(values.shape[1]):
        total=np.float32(0)
        for start in range(0,len(values),4096):
            end=min(len(values),start+4096);accum=np.zeros(4,np.float32)
            for row in range(start,end):accum[0 if end-start<44 else (row-start)%4]+=values[row,column]
            subtotal=accum[0]
            for lane in range(1,4):subtotal+=accum[lane]
            total+=subtotal
        output[column]=total/len(values)
    return output


def curve_baseline(curve,window,cut):
    time=len(curve);window=int(window);positions=np.arange(time)
    lo=np.maximum(0,positions-window//2);hi=np.minimum(time,positions+(window-1)//2+1)
    prefix=np.r_[0,np.cumsum(curve,dtype=float)]
    # movmean rounds its accumulated sum back to the input class BEFORE
    # dividing by the window count; rounding only the quotient differs by ULP.
    smooth=(prefix[hi]-prefix[lo]).astype(curve.dtype)/(hi-lo).astype(curve.dtype)
    step=int(matlab_round(.5*cut));segments=max(1,int(np.ceil(time/step))-1)
    output=np.zeros(time)
    for k in range(segments):
        lo=k*step;hi=min(time,lo+int(cut)+1);position=lo+int(np.argmin(smooth[lo:hi]));value=smooth[position]
        if k==0:output[:position+1]=value
        else:
            with np.errstate(invalid="ignore",divide="ignore"):
                slope=np.float32((value-previous)/np.float32(position-previous_position))
                output[previous_position:position+1]=previous+slope*np.arange(position-previous_position+1,dtype=np.float32)
        if k==segments-1:output[position:]=value
        previous,previous_position=value,position
    return output


def extend_event_curve(curve,others,window):
    start,end=int(window[0]),int(window[-1]);left=np.flatnonzero(others[:start+1]);right=np.flatnonzero(others[end:])
    lo=int(left[-1]) if len(left) else 0;hi=end+int(right[0]) if len(right) else len(curve)-1
    start=lo+int(np.argmin(curve[lo:start+1]));end=end+int(np.argmin(curve[end:hi+1]))
    return curve[start:end+1],np.arange(start,end+1)


def curve_statistics(curve,seconds_per_frame,ignore_tau=True):
    curve=np.asarray(curve).ravel();peak=int(np.argmax(curve));maximum=curve[peak];pp=np.zeros((3,2),float)
    for k,threshold in enumerate((.1,.5,.9)):
        selected=np.flatnonzero(curve<maximum*threshold);left=selected[selected<=peak];right=selected[selected>=peak]
        pp[k]=[left[-1] if len(left) else 0,right[0] if len(right) else len(curve)-1]
    tau=np.nan;decay=curve[peak:int(pp[0,1])+1].astype(float)
    if len(decay)>=2 and np.isfinite(decay).all() and not ignore_tau:
        if len(decay)==2:decay=np.r_[decay,decay[-1]]
        decay-=decay.min()
        with np.errstate(invalid="ignore",divide="ignore"):decay=decay/decay.max()+.05
        if not np.isfinite(decay).all():raise ValueError("Exponential fit has nonfinite normalized samples")
        x=np.arange(len(decay),dtype=float)
        # Uniform-time exponential initial value from half-sequence sums.
        # Project a out analytically, then optimize the same exp1 residuals;
        # no new bounds on b and no assumption of a unimodal objective.
        half=len(decay)//2
        initial=np.log((decay[half:2*half].sum()/decay[:half].sum())**(1/half))
        def projected(beta,derivative=False):
            b=beta[0];t=x-(x[-1] if b>=0 else x[0]);e=np.exp(b*t)
            dot=e@e;numerator=e@decay;a=numerator/dot
            if not derivative:return a*e-decay
            de=t*e;da=((de@decay)*dot-numerator*2*(e@de))/(dot*dot)
            return (da*e+a*de)[:,None]
        fit=optimize.least_squares(projected,[initial],jac=lambda beta:projected(beta,True),
            method="trf",ftol=1e-12,xtol=1e-12,gtol=1e-12,max_nfev=100)
        if not fit.success:raise RuntimeError("Exponential decay fit did not converge")
        tau=-seconds_per_frame/fit.x[0]
    rising=0.0
    for threshold in (.4,.5,.6):
        previous=np.flatnonzero(curve[:peak+1]<=maximum*threshold)
        rising+=((previous[-1] if len(previous) else 0)+1)/3
    rising-=1
    return {"rise19":(pp[2,0]-pp[0,0]+1)*seconds_per_frame,"fall91":(pp[0,1]-pp[2,1]+1)*seconds_per_frame,
        "width55":(pp[1,1]-pp[1,0])*seconds_per_frame,"width11":(pp[0,1]-pp[0,0])*seconds_per_frame,
        "decayTau":tau,"pp":pp,"riseTime":rising}


def basic_features(mask,resolution):
    if mask.shape[2]==1:
        mask=mask[:,:,0];area=float(mask.sum())*resolution*resolution;peri=perimeter(mask)*resolution
        with np.errstate(divide="ignore",invalid="ignore"):circularity=np.divide(4*np.pi*area,peri**2)
        return {"map":mask,"area":area,"peri":peri,"circMetric":circularity}
    volume=float(mask.sum());padded=np.pad(mask,1).astype(np.float32)
    vertices,faces,_,_=marching_cubes(padded,.5,method="lorensen")
    surface=float(mesh_surface_area(vertices,faces));area=volume*resolution**3
    return {"map":mask,"area":area,"surf":surface*resolution**2,"circMetric":np.pi**(1/3)*(6*area)**(2/3)/surface}


def get_features(original,events,options):
    opts=scalar_options(options);shape=original.shape;spatial=int(np.prod(shape[:3]));time=shape[3];n=len(events)
    data=np.asarray(original,np.float32,order="F");labels=np.zeros(shape,np.uint16,order="F")
    for i,p in enumerate(events):labels.ravel(order="F")[p]=min(i+1,65535)
    imputed=data.copy(order="F");imputed[labels>0]=np.nan;imputed=impute_mov(imputed.reshape((spatial,time),order="F"));vectors=data.reshape((spatial,time),order="F")
    dmat=np.zeros((n,time,2),np.float32);dffmat=np.zeros_like(dmat);aligned=np.full((n,111),np.nan,np.float32)
    features={"basic":{},"propagation":{},"loc":{},"curve":{},"bds":[],"notes":{
        "propDirectionOrder":["Anterior","Posterior","Left","Right"] if shape[2]==1 else ["+X","-X","+Y","-Y","+Z","-Z"]}}
    for i,event in enumerate(events):
        if not len(event):raise ValueError("Feature input contains an empty event; remove empty labels first")
        c=np.unravel_index(event,shape,order="F");pixels=np.unique(event%spatial);start,end=int(c[3].min()),int(c[3].max());window=np.arange(start,end+1)
        value_range=opts["maxValueDat"]-opts["minValueDat"]
        raw=(indexed_spatial_mean(vectors[pixels,:]).astype(float)*value_range).astype(np.float32)+np.float32(opts["minValueDat"])
        noise=avg_dat_noise_est(raw,pixels,opts)
        bias=np.float32(obtain_bias(opts["movAvgWin"],opts["cut"])*float(noise))
        baseline=curve_baseline(raw,opts["movAvgWin"],opts["cut"]).astype(np.float32)-bias
        df=raw-baseline;dff=df/(baseline+np.float32(1e-4))
        sigma=max(1e-4,np.sqrt(mean_spatial(np.diff(dff)**2)/np.float32(2)));maximum=dff[window].max()
        # omitnan is needed only for entirely imputed pixel rows; normal rows
        # use the same native spatial accumulation order as raw curves.
        selected=imputed[pixels,:]
        average=indexed_spatial_mean(selected) if not np.isnan(selected).any() else np.nanmean(selected,axis=0)
        clean=(average.astype(float)*value_range).astype(np.float32)+np.float32(opts["minValueDat"]);clean[window]=raw[window]
        clean_dff=(clean-baseline)/(baseline+np.float32(1e-4));peak=int(np.argmax(clean_dff[window]));high=clean_dff[window][peak]
        left=clean_dff[window][:peak+1].min();right=clean_dff[window][peak:].min();z=np.float32(max((high-left+high-right)/np.float32(sigma)/np.float32(2),0))
        extended,extended_window=extend_event_curve(dff,raw!=clean,window)
        stats=curve_statistics(extended,opts["frameRate"],bool(opts["ignoreTau"]))
        dmat[i,:,0]=raw;dmat[i,:,1]=clean;dffmat[i,:,0]=dff;dffmat[i,:,1]=clean_dff
        curve={"tBegin":start,"tEnd":end,"rgt1":np.array([extended_window[0],extended_window[-1]]),"dffMax":maximum,"dfMax":df[window].max(),
            "dffMax2":high,"dffMaxFrame":peak+start,"dffMaxZ":z,"dffMaxPval":np.float32(1)-special.ndtr(z),"duration":(end-start+1)*opts["frameRate"],
            **{k:stats[k] for k in ("rise19","fall91","width55","width11","decayTau")},
            "riseTime":stats["riseTime"]+extended_window[0],"dff1Begin":stats["pp"][0,0]+extended_window[0],"dff1End":stats["pp"][0,1]+extended_window[0],
            "datAUC":native_sum(raw[window]),"dfAUC":native_sum(df[window]),"dffAUC":native_sum(dff[window])}
        frame0=int(curve["dff1Begin"]);frames=frame0+np.arange(-10,101);valid=(frames>=0)&(frames<time)
        aligned[i,valid]=dff[frames[valid]]
        for key,value in curve.items():features["curve"].setdefault(key,[]).append(value)
        for key,value in {"t0":start,"t1":end,"xSpaTemp":event,"xSpa":pixels}.items():features["loc"].setdefault(key,[]).append(value)
        lo=tuple(max(0,int(a.min())-1) for a in c[:3]);hi=tuple(min(size,int(a.max())+2) for a,size in zip(c[:3],shape[:3]));local_shape=tuple(b-a for a,b in zip(lo,hi))
        mask=np.zeros(local_shape,bool,order="F");mask[tuple(axis-offset for axis,offset in zip(c[:3],lo))]=True
        basic=basic_features(mask,opts["spatialRes"]);basic["center"]=matlab_round(np.array([a.mean()+1 for a in c[:3]]))-1
        for key,value in basic.items():features["basic"].setdefault(key,[]).append(value)
        if shape[2]==1:
            lo2=tuple(max(0,int(a.min())-2) for a in c[:2]);hi2=tuple(min(size,int(a.max())+3) for a,size in zip(c[:2],shape[:2]))
            mask=np.zeros(tuple(b-a for a,b in zip(lo2,hi2)),bool);mask[c[0]-lo2[0],c[1]-lo2[1]]=True
            features["bds"].append([b+np.asarray(lo2) for b in boundaries(close_mask(mask),4)])
    return {"fts":features,"dMat":dmat,"dffMat":dffmat,"dffAlignedMat":aligned},opts


def _cells(values,column=False,origin=0,dtype=None):
    output=np.empty((len(values),1) if column else (1,len(values)),object)
    for i,value in enumerate(values):
        array=np.asarray(value,dtype=dtype);output.ravel()[i]=array+origin if origin else array
    return output


def features_mat_payload(result):
    """Explicit one-based MATLAB interchange; physical measurements unchanged."""
    f=result["fts"];output={}
    for group in ("basic","loc","curve"):
        fields={}
        for key,value in f[group].items():
            if key in ("map","center","xSpaTemp","xSpa"):
                values=[]
                for v in value:
                    v=np.asarray(v)
                    if key in ("xSpaTemp","xSpa"):v=(v+1).astype(float).reshape(-1,1)
                    elif key=="center":v=(v+1).astype(float).reshape(1,-1)
                    values.append(v)
                fields[key]=_cells(values)
            else:
                array=np.asarray(value,dtype=np.float32 if key in ("datAUC","dfAUC","dffAUC","dffMax","dfMax","dffMax2","dffMaxZ","dffMaxPval") else float)
                if key in ("t0","t1","tBegin","tEnd","rgt1","dffMaxFrame","riseTime","dff1Begin","dff1End"):array=array+1
                fields[key]=array if key=="rgt1" else array.reshape(1,-1)
        output[group]=fields if fields else np.empty((0,0))
    output["propagation"]={}
    for key,value in f.get("propagation",{}).items():
        output["propagation"][key]=_cells(value) if key in ("propGrow","propShrink","areaChange","areaChangeRate","areaFrame","propMaxSpeed") else np.asarray(value)
    if not output["propagation"]:output["propagation"]=np.empty((0,0))
    if f["bds"]:
        output["bds"]=_cells([_cells([(b+1).astype(float) for b in group],column=True) for group in f["bds"]],column=True)
    else:output["bds"]=np.empty((0,0)) if len(result["dMat"]) and f["notes"]["propDirectionOrder"][0]=="+X" else np.empty((0,1),object)
    output["notes"]={"propDirectionOrder":_cells([np.array([s]) for s in f["notes"]["propDirectionOrder"]])}
    if "channel" in f:output["channel"]=np.array([[float(f["channel"])]] )
    for group in ("region","networkAll","network"):
        if group in f:
            if group=="region":
                from .fea_region import region_mat_payload
                output[group]=region_mat_payload(f[group])
            else:
                from .fea_network import network_mat_payload
                output[group]=network_mat_payload(f[group]) if isinstance(f[group],dict) else f[group]
    return {"fts":output,"dMat":result["dMat"],"dffMat":result["dffMat"],"dffAlignedMat":result["dffAlignedMat"]}


def features_from_mat_payload(payload,options=None):
    """Inverse boundary for persisted/imported features; no scientific recompute."""
    from copy import deepcopy
    f=payload['fts'];output={}
    indexed={'t0','t1','tBegin','tEnd','rgt1','dffMaxFrame','riseTime','dff1Begin','dff1End'}
    for group in ('basic','loc','curve'):
        output[group]={}
        if not isinstance(f[group],dict):continue
        for key,value in f[group].items():
            if key in ('map','center','xSpaTemp','xSpa'):
                values=[]
                for v in value.ravel(order='F'):
                    v=np.asarray(v)
                    if key in ('xSpaTemp','xSpa'):v=v.astype(np.int64).ravel()-1
                    elif key=='center':v=v.ravel()-1
                    values.append(v)
                output[group][key]=values
            else:
                a=np.asarray(value);a=a-1 if key in indexed else a
                output[group][key]=a if key=='rgt1' else a.ravel()
    output['propagation']={}
    if isinstance(f.get('propagation'),dict):
        for key,value in f['propagation'].items():output['propagation'][key]=list(value.ravel(order='F')) if value.dtype==object else np.asarray(value)
    output['bds']=[[np.asarray(v)-1 for v in group.ravel(order='F')] for group in f['bds'].ravel(order='F')] if np.asarray(f['bds']).dtype==object else []
    output['notes']={'propDirectionOrder':[str(np.asarray(v).item()) for v in f['notes']['propDirectionOrder'].ravel(order='F')]}
    if 'channel' in f:output['channel']=int(np.asarray(f['channel']).item())
    if 'region' in f:
        region=deepcopy(f['region'])
        if isinstance(region,dict):
            for key in ('cell','landMark'):
                if isinstance(region.get(key),dict):
                    region[key]['center']=region[key]['center']-1
                    for i,border in enumerate(region[key]['border'].flat):region[key]['border'].flat[i]=border-1
        output['region']=region
    for key in ('network','networkAll'):
        if key not in f:continue
        network=deepcopy(f[key])
        if isinstance(network,dict):
            for field in ('occurSameLocList','occurSameTimeList'):
                for i,indices in enumerate(network[field].flat):
                    if indices.size:network[field].flat[i]=indices.astype(np.int64)-1
        output[key]=network
    return {'fts':output,**{key:payload.get(key,np.empty((len(output['curve'].get('tBegin',[])),0),np.float32)) for key in ('dMat','dffMat','dffAlignedMat')},'opts':scalar_options(options or {})}
