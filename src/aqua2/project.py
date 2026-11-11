"""Scientific project model shared by CLI, worker and desktop viewer."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import h5py
import numpy as np
from . import __version__
from .project_io import load_tree,save_tree,ArrayRef,matlab_array,materialize
from .io import read_mat
from .pipeline import scalar_options
from .cfu import read_cfu_result,cfu_mat_payload,normalize_info


def new_project(title,source=None):
    return dict(schema_version=1,package_version=__version__,title=title,created_utc=datetime.now(timezone.utc).isoformat(),
        source=source or {},stages={},movies={},annotations={'favorites':{},'hidden':{},'regions':[],'landmarks':[]},history=[],bindings={},
        coordinates={'movie_axes':'HWLT','stage_indices':'MATLAB one-based','annotation_indices':'zero-based Fortran','frameRate_unit':'seconds/frame','time_origin':'working movie frame 0'})


def compact_project_data(data):
    """Persist MATLAB-equivalent results, not normalized/debug/workflow caches."""
    result=dict(data);result.pop('runtime',None);stages=dict(data['stages']);stages.pop('preprocessing',None)
    result['movies']={channel:{key:value for key,value in movie.items() if key in ('stored','dF')} for channel,movie in data['movies'].items()}
    if 'detection' in stages:
        result.pop('opts',None)
        detection=dict(stages['detection'])
        for channel in result['movies']:
            if channel not in detection:continue
            scopes={}
            for scope,spatial in detection[channel].items():
                spatial=dict(spatial);spatial.pop('seLst',None);spatial.pop('opts',None)
                if 'features' in spatial:
                    features=dict(spatial['features']);features.pop('opts',None);features.pop('dffAlignedMat',None);spatial['features']=features
                if 'ov' in spatial:spatial.pop('datR',None)
                elif 'datR' in spatial and spatial['datR'].size:
                    values=np.asarray(spatial['datR']);pixels=np.flatnonzero(values.ravel(order='F'))
                    if pixels.nbytes+len(pixels)*values.dtype.itemsize<values.nbytes:
                        spatial['ov']={'shape':values.shape,'pix':pixels,'val':values.ravel(order='F')[pixels]};spatial.pop('datR')
                scopes[scope]=spatial
            detection[channel]=scopes
        stages['detection']=detection
    result['stages']=stages
    return result


class Project:
    def __init__(self,data,path=None):
        # Older workbenches used a localized event-mode prefix in annotations.
        annotations=data.get('annotations',{});updated={}
        for kind in ('favorites','hidden'):
            values=annotations.get(kind,{})
            if any(key.startswith('\u4e8b\u4ef6:') for key in values):
                merged={}
                for key,ids in values.items():
                    key='Event:'+key.split(':',1)[1] if key.startswith('\u4e8b\u4ef6:') else key
                    merged[key]=sorted(set(merged.get(key,[]))|set(int(i) for i in ids))
                updated[kind]=merged
        if updated:data={**data,'annotations':{**annotations,**updated}}
        self.data=data;self.path=Path(path).resolve() if path else None;self._cache={}
    @classmethod
    def open(cls,path):
        project=cls(load_tree(path),path);data=project.data
        if not isinstance(data,dict) or data.get('schema_version')!=1 or not all(key in data for key in ('movies','stages','annotations','title')):raise ValueError('Not a complete AQuA2 analysis project')
        shape=project.shape
        if len(shape)!=4 or min(shape)<1 or not project.channels:raise ValueError('Project has invalid HWLT dimensions/channels')
        for channel in project.channels:
            movie=project.movie(channel)
            if movie is None or tuple(movie.shape)!=shape:raise ValueError('Project movie shape does not match its analysis coordinates')
        return project
    def save(self,path=None,*,embed_external=False):
        target=path or self.path
        if target is None:raise ValueError('Choose a project path')
        save_tree(target,compact_project_data(self.data),embed_external=embed_external)
        return Project.open(target)
    @property
    def opts(self):
        stages=self.data['stages']
        return scalar_options(stages.get('detection',stages.get('preprocessing',{})).get('opts',self.data.get('opts',{})))
    @property
    def channels(self):return sorted(self.data['movies'])
    @property
    def shape(self):return tuple(int(n) for n in np.asarray(self.opts['sz']).ravel())
    def movie(self,channel='channel1',kind='stored'):
        values=self.data['movies'][channel]
        return values.get(kind,values.get('working'))
    def spatial(self,channel='channel1',scope='local'):
        spatial=self.data['stages'].get('detection',{}).get(channel,{}).get(scope)
        if spatial is not None and 'datR' not in spatial and 'ov' in spatial:
            from .reconstruction import Reconstruction
            overlay=spatial['ov'];spatial['datR']=Reconstruction(tuple(overlay['shape']),overlay['pix'],overlay['val'],overlay['val'].dtype)
        return spatial
    def events(self,channel='channel1',scope='local'):
        key=('events',channel,scope)
        if key not in self._cache:
            spatial=self.spatial(channel,scope)
            self._cache[key]=[np.asarray(v,np.int64).ravel()-1 for v in spatial['evt'].ravel(order='F')] if spatial else []
        return self._cache[key]
    def features(self,channel='channel1',scope='local'):
        spatial=self.spatial(channel,scope)
        return spatial.get('features') if spatial else None
    def cfus(self,channel='channel1'):
        key=('cfus',channel)
        if key not in self._cache:
            value=self.data['stages'].get('cfu',{}).get('cfuInfo'+channel[-1],np.empty((0,0)))
            self._cache[key]=normalize_info(value)
        return self._cache[key]
    def frame(self,t=0,z=0,channel='channel1',kind='stored'):
        if not 0<=t<self.shape[3] or not 0<=z<self.shape[2]:raise IndexError('Frame or Z slice out of range')
        movie=self.data['movies'][channel].get(kind)
        if movie is None:
            if kind=='dF':raise ValueError('This project does not contain a dF movie')
            movie=self.movie(channel)
        return np.asarray(movie[:,:,z,t])
    def projection(self,channel='channel1',z=0,maximum=False):
        key=('projection',channel,z,maximum)
        if key not in self._cache:
            movie=self.movie(channel);result=None;count=0
            for start in range(0,self.shape[3],32):
                block=np.asarray(movie[:,:,z,start:start+32],dtype=np.float32)
                part=block.max(axis=-1) if maximum else block.sum(axis=-1,dtype=np.float64)
                result=part if result is None else (np.maximum(result,part) if maximum else result+part);count+=block.shape[-1]
            self._cache[key]=result if maximum else result/count
        return self._cache[key]
    def working_movie(self,channel='channel1'):
        working=self.data['movies'][channel].get('working')
        if working is not None:return np.asarray(working,np.float32)
        opts=self.opts;number=channel[-1];low=np.float32(opts['minValueDat'+number]);high=np.float32(opts['maxValueDat'+number])
        return (np.asarray(self.movie(channel),np.float32)-low)/(high-low)
    def summary(self):
        return dict(title=self.data['title'],shape_hwlt=list(self.shape),channels=self.channels,
            events={c:len(self.events(c)) for c in self.channels},cfus={c:len(self.cfus(c)) for c in self.channels},
            stages=list(self.data['stages']),external_movies=any(isinstance(v,ArrayRef) and v.matlab for m in self.data['movies'].values() for v in m.values()))


def import_saved_project(source,output=None,*,cfu_file=None,title=None,embed_external=False):
    """Read selected v7.3 fields; MATLAB UI objects and RGB overlays are skipped."""
    source=Path(source).resolve()
    if not h5py.is_hdf5(source):raise ValueError('Saved AQuA2 import currently requires a v7.3 res MAT')
    opts=read_mat(source,['res/opts'])['res/opts']
    with source.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    data=new_project(title or source.stem,{'path':str(source),'mode':'import_saved','bytes':source.stat().st_size,'sha256':digest})
    detection={'opts':opts}
    with h5py.File(source,'r') as file:fields=set(file['res'])
    if 'scl' in fields:
        scale=read_mat(source,['res/scl'])['res/scl']
        if isinstance(scale,dict):
            data['display']={'matlab_scale':{key:float(np.asarray(scale[key]).item()) for key in ('min','max') if key in scale and np.asarray(scale[key]).size==1}}
    for ch in (1,2):
        if ch==2 and bool(np.asarray(opts['singleChannel']).item()):continue
        key=f'channel{ch}';data['movies'][key]={'stored':matlab_array(source,f'res/datOrg{ch}')};detection[key]={}
        if f'dF{ch}' in fields:
            delta=matlab_array(source,f'res/dF{ch}')
            if delta.size:data['movies'][key]['dF']=delta
        for scope,prefix,fts_prefix in (('local','evt','fts'),('global','gloEvt','ftsGlo')):
            name=prefix+str(ch)
            if name not in fields:continue
            events=read_mat(source,['res/'+name])['res/'+name]
            if scope=='global' and not events.size:continue
            spatial={'evt':events,'datR':np.empty((0,0)),'riseLst':np.empty((1,0),object)}
            rise=('riseLst' if scope=='local' else 'gloRiseLst')+str(ch)
            if rise in fields:spatial['riseLst']=read_mat(source,['res/'+rise])['res/'+rise]
            fts_name=fts_prefix+str(ch)
            if fts_name in fields:
                fts=read_mat(source,['res/'+fts_name])['res/'+fts_name]
                if isinstance(fts,dict):
                    features={'fts':fts}
                    for target in ('dMat','dffMat','dffAlignedMat'):
                        field=target+('Glo' if scope=='global' else '')+str(ch)
                        features[target]=read_mat(source,['res/'+field])['res/'+field] if field in fields else np.empty((events.size,0),np.float32)
                    spatial['features']=features
            detection[key][scope]=spatial
    data['stages']['detection']=detection;data['bindings']['events']='Explicit saved-result import; source generating version unknown'
    project=Project(data)
    if cfu_file:
        result=read_cfu_result(cfu_file);validate_cfu_binding(project,result)
        data['stages']['cfu']=cfu_mat_payload(result);data['source']['cfu_file']=str(Path(cfu_file).resolve())
        with Path(cfu_file).open('rb') as stream:data['source']['cfu_sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
    return project.save(output,embed_external=embed_external) if output else project


def validate_cfu_binding(project,result):
    """Reject incompatible CFU imports, including local masks from other events."""
    shape=project.shape;spatial=int(np.prod(shape[:3]))
    for ch in (1,2):
        key=f'channel{ch}';infos=result[f'cfuInfo{ch}'];events=project.events(key)
        if infos and key not in project.channels:raise ValueError('CFU channel does not exist in project')
        for record in infos:
            ids=np.asarray(record['eventIds'],np.int64)
            if np.any((ids<0)|(ids>=len(events))):raise ValueError('CFU event IDs do not match this project')
            if np.asarray(record['weightMap']).shape!=(shape[:2] if shape[2]==1 else shape[:3]) or np.asarray(record['occurrence']).size!=shape[3]:raise ValueError('CFU spatial/time dimensions do not match project')
            member=record.get('memberships')
            if isinstance(member,dict) and 'VoxelIndices' in member:
                for event,voxels in zip(member['EventID'],member['VoxelIndices']):
                    if not 0<=event<len(events) or not np.isin(voxels,events[event]).all():raise ValueError('CFU local membership belongs to different event data')


def relink_saved_source(project,new_path,*,save=False):
    """Relocate a linked MAT only when its content hash still matches."""
    new_path=Path(new_path).resolve();expected=project.data['source'].get('sha256')
    if not expected:raise ValueError('This older project has no source hash; import it again to create a verifiable link')
    with new_path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
    if actual!=expected:raise ValueError('Selected MAT is different from the original source; link was not changed')
    old=project.data['source']['path']
    def replace(value):
        if isinstance(value,ArrayRef) and value.matlab and str(Path(value.path).resolve())==str(Path(old).resolve()):return ArrayRef(str(new_path),value.node,value.shape,value.dtype,True)
        if isinstance(value,dict):return {k:replace(v) for k,v in value.items()}
        if isinstance(value,list):return [replace(v) for v in value]
        return value
    project.data['movies']=replace(project.data['movies']);project.data['source']['path']=str(new_path)
    project.data['history'].append({'action':'relink_source','sha256':actual});return project.save() if save else project


def export_project(project,directory,*,selection=None,progress=None,display_settings=None):
    """Selected CSV tables, masked TIFF movies and propagation maps; no MAT."""
    from .result_export import export_results
    return export_results(project,directory,selection=selection,progress=progress,display_settings=display_settings)
