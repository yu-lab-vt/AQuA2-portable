"""Independent CFU detection/analysis; API event, voxel and CFU IDs start at 0."""
from pathlib import Path
import numpy as np
from .pipeline import scalar_options
from .se_common import matlab_round
from .cfu_spatial import prepare_events,minimum_measure,materialize_memberships,membership_time_windows
from .cfu_merge import compute_curves,iterative_merge,default_merge_parameters,moving_mean
from .cfu_dependency import all_dependencies,group_cfus
from .io import MatStructArray

INFO_FIELDS=('id','eventIds','weightMap','occurrence','meanCurve','meanDff','timeWindow','nonTimeWindow',
    'frequencyStats','grayEventIds','spatialClass','isManual','parentId','memberships')


def default_options():
    return {'cfuDetect':dict(overlapThr1=.5,overlapThr2=.5,minNumEvt1=3,minNumEvt2=3),
        'cfuAnalysis':dict(maxDist=10,shift=0),'cfuGroup':dict(pValueThr=1e-5,cfuNumThr=3)}


def frequency_statistics(peaks,seconds_per_frame):
    peaks=np.asarray(peaks);dt=np.diff(np.sort(peaks))*seconds_per_frame;dt=dt[dt>0]
    frequency=0.;method='N/A';peak80=np.nan
    if len(dt):
        cv=(np.std(dt,ddof=1) if len(dt)>1 else 0)/dt.mean()
        if cv>1:frequency=np.median(1/dt);method='Med'
        else:frequency=1/dt.mean();method='Mean'
        if len(dt)>=5:peak80=np.percentile(1/dt,80,method='hazen')
    return dict(count=len(peaks),mainFreq=frequency,method=method,peakFreq80=peak80,dt=dt)


def rising_time(curve,start,end,occupied,thresholds=(.4,.5,.6)):
    maximum=curve[start:end+1].max();peak=start+int(np.argmax(curve[start:end+1]))
    start+=int(np.argmin(curve[start:peak+1]));previous=np.flatnonzero(occupied[:start+1]);previous=int(previous[-1]+1) if len(previous) else 0
    start=max(start-10,previous)
    # The source returns [] through its catch for this empty extension;
    # indexing occurrence(:,[]) leaves that event unmarked.
    if start>peak:return None
    minimum=curve[start:peak+1].min();bottom=start+int(np.argmin(curve[start:peak+1]))
    if bottom==peak:return float(peak)
    scaled=(curve[bottom:peak+1]-minimum)/(maximum-minimum);rise=[]
    for threshold in thresholds:
        selected=np.flatnonzero(scaled>threshold)
        if not len(selected):return float(bottom)
        rise.append(selected[0]+1)
    return float(np.mean(rise)+bottom-1)


def detect_channel(movie,events,options,cfu_options=None,*,channel=1,peak_frames=None,progress=None,return_stages=False):
    opts=scalar_options(options);settings=default_options() if cfu_options is None else cfu_options
    settings={k:scalar_options(v) for k,v in settings.items()};shape=movie.shape
    events=[np.asarray(e,np.int64).ravel() for e in events]
    if len(shape)!=4:raise ValueError('CFU movie must use HWLT axes')
    detect=settings['cfuDetect'];threshold=float(detect[f'overlapThr{channel}']);minimum=int(detect[f'minNumEvt{channel}'])
    if not 0<=threshold<=1 or minimum<1:raise ValueError('Invalid CFU overlap threshold or minimum event count')
    if peak_frames is None:raise ValueError('Peak frames are required to preserve frequency statistics from event features')
    if len(peak_frames)!=len(events):raise ValueError('Peak-frame count must match event count')
    pre=prepare_events(events,shape);cfus=materialize_memberships(minimum_measure(pre,shape,threshold,minimum),events,shape)
    curves,dff=compute_curves(cfus,movie,opts['movAvgWin'],opts['cut'])
    initial=(cfus,curves.copy(),dff.copy()) if return_stages else None
    parameters=default_merge_parameters();key='minimumCurveCorrelation' if channel==1 else 'minimumCurveCorrelation2'
    if key in detect:parameters['MinimumCurveCorrelation']=float(np.clip(detect[key],.85,1.))
    cfus,curves,dff,merge=iterative_merge(cfus,curves,dff,movie,opts['movAvgWin'],opts['cut'],parameters,progress)
    own,other,overlap=membership_time_windows(cfus,shape);info=[];size=int(np.prod(shape[:3]))
    for i,c in enumerate(cfus):
        smoothed=moving_mean(curves[i],2);occurrence=np.zeros(shape[3],bool)
        for event_id in c['events']:
            times=events[event_id]//size;rise=rising_time(smoothed,int(times.min()),int(times.max()),own[i])
            if rise is None:continue
            frame=int(matlab_round(rise+1))-1
            if not 0<=frame<shape[3]:raise ValueError('CFU rising time is outside movie bounds')
            occurrence[frame]=True
        record={key:np.empty((0,0)) for key in INFO_FIELDS}
        record.update(id=i,eventIds=c['events'],weightMap=c['region'],occurrence=occurrence,meanCurve=curves[i],meanDff=dff[i],
            timeWindow=own[i],nonTimeWindow=other[i],frequencyStats=frequency_statistics(np.asarray(peak_frames)[c['events']],opts['frameRate']),
            parentId=c['parent'],memberships=c['memberships'])
        info.append(record)
    if return_stages:return info,dict(pre=pre,initial=initial,cfus=cfus,curves=curves,dffCurves=dff,own=own,other=other,overlapping=overlap,**merge)
    return info


def analyze(info1,info2=(),options=None,progress=None):
    settings=default_options() if options is None else options
    analysis=scalar_options(settings['cfuAnalysis']);group=scalar_options(settings['cfuGroup'])
    relations=all_dependencies(info1,info2,analysis['maxDist'],analysis['shift'],progress)
    groups=group_cfus(info1,info2,relations,group['pValueThr'],group['cfuNumThr'])
    return relations,groups


def run_cfu(movies,events,options,peak_frames,cfu_options=None,*,progress=None):
    settings=default_options() if cfu_options is None else cfu_options;infos=[]
    if len(movies)!=len(events) or len(movies)!=len(peak_frames) or len(movies) not in (1,2):raise ValueError('Provide one or two corresponding movie/event/peak-frame channels')
    for i,(movie,evts,peaks) in enumerate(zip(movies,events,peak_frames)):
        infos.append(detect_channel(movie,evts,options,settings,channel=i+1,peak_frames=peaks,progress=progress))
    info1=infos[0];info2=infos[1] if len(infos)>1 else []
    relation,groups=analyze(info1,info2,settings,progress)
    return dict(cfuInfo1=info1,cfuInfo2=info2,cfuRelation=relation,cfuGroupInfo=groups,cfuOpts=settings,channel_count=len(movies))


def cfu_from_saved_result(path,cfu_options=None,*,progress=None):
    """Use saved datOrg intensities directly; no re-normalization or crop."""
    from .io import read_mat,read_saved_dat_org
    opts=read_mat(path,['res/opts'])['res/opts'];channels=1 if bool(opts['singleChannel'].item()) else 2;infos=[]
    settings=default_options() if cfu_options is None else cfu_options
    for channel in range(1,channels+1):
        selected=read_mat(path,[f'res/evt{channel}',f'res/fts{channel}/curve/dffMaxFrame'])
        evts=[np.asarray(p,np.int64).ravel()-1 for p in selected[f'res/evt{channel}'].ravel(order='F')]
        peaks=np.asarray(selected[f'res/fts{channel}/curve/dffMaxFrame']).ravel()-1
        movie=read_saved_dat_org(path,channel=channel)
        infos.append(detect_channel(movie,evts,opts,settings,channel=channel,peak_frames=peaks,progress=progress));del movie
    info1=infos[0];info2=infos[1] if channels==2 else []
    relation,groups=analyze(info1,info2,settings,progress)
    return dict(cfuInfo1=info1,cfuInfo2=info2,cfuRelation=relation,cfuGroupInfo=groups,cfuOpts=settings,channel_count=channels)


def _cells(values,column=False):
    out=np.empty((len(values),1) if column else (1,len(values)),object)
    for i,value in enumerate(values):out.ravel()[i]=value
    return out


def membership_mat_payload(member):
    output={}
    for key,value in member.items():
        if key in ('SpatialPixels','VoxelIndices'):
            output[key]=_cells([(np.asarray(p,float)+1).reshape(-1,1) for p in value])
        elif key in ('EventID','MergedFromCFUs','MergedParentIDs'):output[key]=(np.asarray(value,float)+1).reshape(1,-1)
        elif key in ('SupportPixels','PrimaryPixels'):output[key]=(np.asarray(value,float)+1).reshape(-1,1)
        elif key=='SupportWeights':output[key]=np.asarray(value).reshape(-1,1)
        else:output[key]=np.asarray(value,float).reshape(1,-1)
    return output


def spatial_mat_payload(cfus):
    return dict(regions=_cells([c['region'] for c in cfus],True),
        lists=_cells([(c['events']+1).astype(float).reshape(1,-1) for c in cfus],True),
        parents=np.array([c['parent']+1 for c in cfus],float).reshape(-1,1),
        memberships=_cells([membership_mat_payload(c['memberships']) for c in cfus],True))


def info_mat_payload(records,*,empty_channel=False):
    if not records and empty_channel:return np.empty((0,0))
    out=np.empty((len(records),1),object).view(MatStructArray);out.mat_fields=INFO_FIELDS
    for i,r in enumerate(records):
        value=dict(r)
        for key in ('id','parentId'):value[key]=np.array([[float(r[key]+1)]]) if np.asarray(r[key]).size else r[key]
        if np.asarray(r['eventIds']).size:value['eventIds']=(np.asarray(r['eventIds'],float)+1).reshape(1,-1)
        if np.asarray(r['grayEventIds']).size:
            gray=np.asarray(r['grayEventIds'],float)+1;value['grayEventIds']=gray.reshape(1,-1) if gray.ndim==1 else gray
        for key in ('occurrence','meanCurve','meanDff','timeWindow','nonTimeWindow'):
            if np.asarray(r[key]).size:value[key]=np.asarray(r[key]).reshape(1,-1)
        if isinstance(r['frequencyStats'],dict):
            value['frequencyStats']={k:np.array([str(v)]) if k=='method' else np.asarray(v,float).reshape(1,-1) for k,v in r['frequencyStats'].items()}
            value['frequencyStats']['dt']=np.asarray(r['frequencyStats']['dt']).reshape(-1,1) if r['frequencyStats']['count']>=2 else np.empty((0,0))
        if isinstance(r['memberships'],dict):value['memberships']=membership_mat_payload(r['memberships'])
        out[i,0]=value
    return out


def cfu_mat_payload(result):
    relation=np.array(result['cfuRelation'],float,copy=True)
    if relation.size:relation[:,:2]+=1
    else:relation=np.empty((0,0))
    groups=np.empty((len(result['cfuGroupInfo']),4),object)
    for i,g in enumerate(result['cfuGroupInfo']):
        groups[i]=[np.array([[float(g['id']+1)]]),(g['cfuIds']+1).astype(float).reshape(-1,1),g['delays'].reshape(-1,1),g['pvalues'].reshape(-1,1)]
    return dict(cfuInfo1=info_mat_payload(result['cfuInfo1']),cfuInfo2=info_mat_payload(result['cfuInfo2'],empty_channel=result.get('channel_count',1)==1),cfuRelation=relation,cfuGroupInfo=groups,cfuOpts=result['cfuOpts'])


def normalize_info(value,*,one_based=True):
    """Read the current 14-field schema and older N-by-column cell schema."""
    if isinstance(value,dict):raw=[value]
    elif isinstance(value,MatStructArray):raw=list(value.ravel(order='F'))
    elif isinstance(value,np.ndarray) and value.dtype==object:
        if value.ndim!=2:raise ValueError('Legacy CFU cells must be two dimensional')
        raw=[dict(zip(INFO_FIELDS,row[:len(INFO_FIELDS)])) for row in value]
    elif np.asarray(value).size==0:return []
    else:raise ValueError('CFU information must be a struct array or legacy cell table')
    output=[];offset=int(one_based)
    for i,original in enumerate(raw):
        record={key:np.empty((0,0)) for key in INFO_FIELDS};record.update(original)
        for key in ('id','parentId'):
            v=np.asarray(record[key]);record[key]=int(v.item())-offset if v.size else (i if key=='id' else v)
        for key in ('eventIds',):
            v=np.asarray(record[key]);record[key]=v.astype(np.int64).ravel()-offset if v.size else v
        gray=np.asarray(record['grayEventIds']);record['grayEventIds']=gray.astype(np.int64)-offset if gray.size else gray
        for key in ('occurrence','timeWindow','nonTimeWindow','meanCurve','meanDff'):
            v=np.asarray(record[key]);record[key]=v.ravel() if v.size else v
        if isinstance(record['frequencyStats'],dict):
            record['frequencyStats']={k:(np.asarray(v).ravel() if k=='dt' else np.asarray(v).item()) for k,v in record['frequencyStats'].items()}
        if isinstance(record['memberships'],dict):
            member={}
            for key,v in record['memberships'].items():
                if key in ('SpatialPixels','VoxelIndices'):member[key]=[np.asarray(p,np.int64).ravel()-offset for p in v.ravel(order='F')]
                elif key in ('EventID','MergedFromCFUs','MergedParentIDs','SupportPixels','PrimaryPixels'):member[key]=np.asarray(v,np.int64).ravel()-offset
                else:member[key]=np.asarray(v).ravel()
            record['memberships']=member
        output.append(record)
    return output


def read_cfu_result(path):
    """Read standalone or res-nested CFU data without loading saved movies."""
    from .io import read_mat
    fields=['cfuInfo1','cfuInfo2','cfuRelation','cfuGroupInfo','cfuOpts']
    try:raw=read_mat(path,fields)
    except KeyError:raw={k:v for k,v in ((k,read_mat(path,['res/'+k])['res/'+k]) for k in fields)}
    relation=np.asarray(raw['cfuRelation'],float).reshape(-1,4).copy();relation[:,:2]-=1;groups=[]
    for row in raw['cfuGroupInfo']:
        groups.append(dict(id=int(np.asarray(row[0]).item())-1,cfuIds=np.asarray(row[1],np.int64).ravel()-1,delays=np.asarray(row[2]).ravel(),pvalues=np.asarray(row[3]).ravel()))
    second=raw['cfuInfo2']
    channels=2 if isinstance(second,(MatStructArray,dict)) or np.asarray(second).size>0 or (isinstance(second,np.ndarray) and second.dtype==object and second.ndim==2 and second.shape[1]>0) else 1
    return dict(cfuInfo1=normalize_info(raw['cfuInfo1']),cfuInfo2=normalize_info(raw['cfuInfo2']),cfuRelation=relation,cfuGroupInfo=groups,cfuOpts=raw['cfuOpts'],channel_count=channels)
