"""Stage 3–7 integration in memory, with explicit project saving and progress."""
from pathlib import Path
import hashlib,json
import numpy as np
from .pipeline import preprocess,scalar_options
from .config import read_preset
from .project import Project,new_project
from .project_io import materialize
from .detection import detect_from_stage3,detection_mat_payload
from .cfu import run_cfu,cfu_mat_payload,default_options
from .io import read_mat


def _file_identity(path):
    path=Path(path).resolve()
    before=path.stat()
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Input changed while being hashed')
    return {'path':str(path),'sha256':digest,'bytes':after.st_size,'mtime_ns':after.st_mtime_ns}


def stored_intensities(working,opts,channel=1):
    """Match aqua_cmd_batch's save boundary before CFU, including quantization."""
    low=float(opts[f'minValueDat{channel}']);high=float(opts[f'maxValueDat{channel}'])
    value=(np.asarray(working,dtype=np.float64)*(high-low)).astype(np.float32)+np.float32(low)
    dtype=np.uint8 if int(opts.get('BitDepth',16))==8 else np.uint16
    return np.floor(np.clip(np.nan_to_num(value,nan=0),0,np.iinfo(dtype).max)+.5).astype(dtype)


def _native_detections(payload):
    from .evt import spatial_from_mat_payload
    from .fea import features_from_mat_payload
    output={'opts':scalar_options(payload['opts'])}
    for key in ('channel1','channel2'):
        if key not in payload:continue
        output[key]={}
        for scope,stored in payload[key].items():
            spatial=spatial_from_mat_payload(stored,payload['opts'])
            if 'features' in stored:spatial['features']=features_from_mat_payload(stored['features'],payload['opts'])
            output[key][scope]=spatial
    return output


def run_workflow(source,output=None,config=None,*,resume=False,overwrite=False,progress=None,save_project=False):
    """Compute in memory. Only an explicit save_project request writes a project."""
    if resume:raise ValueError('Intermediate checkpoints have been removed; open a manually saved project instead')
    config={'preset':1,'with_cfu':True,'saved_result':False,'variable':None,**dict(config or {})}
    source=Path(source).resolve();output=Path(output).resolve() if output else None
    if config.get('options'):
        config['options']={k:(v.item() if isinstance(v,np.generic) else v) for k,v in scalar_options(config['options']).items()}
        if any(isinstance(v,np.ndarray) for v in config['options'].values()):raise ValueError('Configuration overrides must be scalar parameters; use input arrays through the scientific APIs')
    if config.get('cfu_settings'):config['cfu_settings']={k:scalar_options(v) for k,v in config['cfu_settings'].items()}
    if isinstance(config.get('frames'),slice):
        frame_slice=config['frames'];config['frames']=[frame_slice.start,frame_slice.stop,frame_slice.step]
    for key in ('parameters','channel2','mask','regions','landmarks','cfu_options'):
        if config.get(key):config[key]=str(Path(config[key]).resolve())
    if save_project:
        if output is None or output==source or output.suffix.lower()!='.h5':raise ValueError('Choose a separate .aqua.h5 output to save the project')
        if output.exists() and not overwrite:raise FileExistsError('Project exists; choose a new path or explicitly overwrite')
    def emit(stage,detail='',done=None,total=None):
        if progress:progress(dict(stage=stage,detail=str(detail),done=done,total=total))
    emit('input','Checking source and parameters');identity=_file_identity(source)
    parameters=config.get('parameters') or Path(__file__).with_name('data')/'parameters_for_batch.csv'
    options=None if config.get('saved_result') and not config.get('parameters') else read_preset(parameters,int(config.get('preset',1)))
    if config.get('options'):options={**dict(options or {}),**scalar_options(config['options'])}
    frames=config.get('frames');frames=slice(*frames) if frames is not None and not isinstance(frames,slice) else frames
    emit('preprocessing','Running')
    stage3=preprocess(source,options,config.get('channel2'),variable=config.get('variable'),spatial_mask=config.get('mask'),
        saved_result=config.get('saved_result',False),frames=frames,gui_median=config.get('gui_median',False),
        progress=lambda detail,done=None,total=None:emit('preprocessing',detail,done,total))
    for channel in (1,2):stage3.pop(f'intermediate{channel}',None)
    stage3['opts']=dict(stage3['opts']);stage3['opts']['sz']=np.asarray(stage3['opts']['sz'],dtype=float).reshape(1,4);stage3['opts']['cut']=float(stage3['opts']['cut'])
    from .features import mask_options_from_files
    from .io import read_working_mask
    masks=mask_options_from_files(config.get('regions'),config.get('landmarks'))
    mask=read_working_mask(config['mask'],stage3['datOrg1'].shape[:3],stage3['opts'].get('regMaskGap',0)) if config.get('mask') else None
    emit('detection','Running')
    result=detect_from_stage3(stage3,spatial_mask=mask,with_features=True,**masks,
        progress=lambda *args:emit('detection',args[0],args[1] if len(args)>1 else None,args[2] if len(args)>2 else None))
    detected=detection_mat_payload(result);native=_native_detections(detected);opts=stage3['opts'];movies={}
    for ch in (1,2):
        if f'datOrg{ch}' in stage3:movies[f'channel{ch}']={'working':stage3[f'datOrg{ch}'],'stored':stored_intensities(stage3[f'datOrg{ch}'],opts,ch),'dF':stage3[f'dF{ch}']}
    cfus=None
    if config.get('with_cfu',True):
        from .config import read_parameter_bundle
        emit('cfu','Running');settings=config.get('cfu_settings') or read_parameter_bundle(parameters,int(config.get('preset',1)))[1]
        if config.get('cfu_options'):
            path=Path(config['cfu_options']);settings=json.loads(path.read_text(encoding='utf-8')) if path.suffix.lower()=='.json' else read_mat(path,['cfuOpts'])['cfuOpts']
        arrays=[];events=[];peaks=[]
        for channel in movies:
            local=native[channel]['local'];arrays.append(movies[channel]['stored']);events.append(local['evt']);peaks.append(np.asarray(local['features']['fts']['curve'].get('dffMaxFrame',[])))
        cfus=cfu_mat_payload(run_cfu(arrays,events,opts,peaks,settings,
            progress=lambda *args:emit('cfu',*args[:1],done=args[1] if len(args)>1 else None,total=args[2] if len(args)>2 else None)))
    data=new_project(config.get('title') or source.stem,identity);data['config']=config
    data['bindings']={'input_mode':'saved' if config.get('saved_result') else 'raw','cfu_intensity':'MATLAB saved uint8/uint16 quantization boundary'}
    data['movies']=movies;data['stages']={'detection':detected}
    from .fea_region import mask_pixels
    for key,name_key in (('regions','region_names'),('landmarks','landmark_names')):
        data['annotations'][key]=mask_pixels(masks.get(key),movies['channel1']['stored'].shape[:3])
        if name_key in masks:data['annotations'][name_key]=masks[name_key]
    if cfus is not None:data['stages']['cfu']=cfus
    data['history'].append({'action':'run'})
    current=source.stat()
    if (current.st_size,current.st_mtime_ns)!=(identity['bytes'],identity['mtime_ns']):raise ValueError('Source changed during analysis')
    project=Project(data)
    if save_project:
        emit('saving','Writing complete project');project=project.save(output)
    emit('complete','Analysis complete (not saved)' if project.path is None else str(project.path))
    return project


def run_batch(manifest,*,progress=None):
    """JSON jobs resolve relative to their manifest; failures do not erase successes."""
    path=Path(manifest).resolve();settings=json.loads(path.read_text(encoding='utf-8'));jobs=settings['jobs'];results=[];outputs=set()
    for job in jobs:
        output=(path.parent/job['output']).resolve() if job.get('output') else None
        if output is not None and output in outputs:raise ValueError('Batch output paths must be unique')
        outputs.add(output)
    for i,job in enumerate(jobs):
        config={**settings.get('defaults',{}),**job.get('config',{})}
        for key in ('parameters','channel2','mask','regions','landmarks','cfu_options'):
            if config.get(key):config[key]=str((path.parent/config[key]).resolve())
        try:
            project=run_workflow(path.parent/job['input'],path.parent/job['output'] if job.get('output') else None,config,save_project=job.get('save_project',False),
                overwrite=job.get('overwrite',False),progress=(lambda value:progress({'job':i+1,'jobs':len(jobs),**value})) if progress else None)
            results.append({'input':job['input'],'passed':True,'project':str(project.path) if project.path else None,'summary':project.summary()})
        except Exception as error:results.append({'input':job['input'],'passed':False,'error':str(error)})
    return {'passed':all(r['passed'] for r in results),'jobs':results}


def recompute_cfu(path,settings=None,*,progress=None,save=False):
    """Use the project's stored intensities and stable local-event IDs."""
    project=path if isinstance(path,Project) else Project.open(path);arrays=[];events=[];peaks=[]
    for channel in project.channels:
        features=project.features(channel)
        if not features:raise ValueError('Extract event features before CFU detection')
        arrays.append(np.asarray(project.movie(channel)));events.append(project.events(channel))
        peaks.append(np.asarray(features['fts']['curve'].get('dffMaxFrame',[])).ravel()-1)
    result=run_cfu(arrays,events,project.opts,peaks,settings or default_options(),progress=progress)
    project.data['stages']['cfu']=cfu_mat_payload(result);project._cache.clear();project.data['history'].append({'action':'recompute_cfu'})
    return project.save() if save else project


def refresh_annotation_features(project,options=None,*,progress=None):
    """Update only region/network fields; reuse event curves and peak IDs.

    This avoids rebuilding working movies or changing CFU inputs merely
    because a user edited a polygon after event detection.
    """
    from types import SimpleNamespace
    from .fea_region import mask_pixels,region_features,region_mat_payload
    from .fea_network import event_network,network_mat_payload
    annotations=project.data['annotations'];opts=dict(project.opts);opts.update(options or {})
    dirty=annotations.get('regions_dirty',False)
    if not dirty and bool(opts.get('networkFeatures',False))==bool(project.opts.get('networkFeatures',False)):return project
    regions=mask_pixels(annotations.get('regions'),project.shape[:3]);landmarks=mask_pixels(annotations.get('landmarks'),project.shape[:3])
    region_names=annotations.get('region_names');landmark_names=annotations.get('landmark_names')
    if region_names is None:region_names=[str(i+1) for i in range(len(regions))]
    if landmark_names is None:landmark_names=[str(i+1) for i in range(len(landmarks))]
    pending=[]
    for channel in project.channels:
        for scope in ('local','global'):
            spatial=project.spatial(channel,scope)
            if spatial is None or not spatial.get('features'):continue
            events=project.events(channel,scope);reconstruction=spatial.get('datR')
            if landmarks and (reconstruction is None or not reconstruction.size):
                raise ValueError('Imported MATLAB file lacks reconstructed datR; rerun event detection to calculate landmark features.')
            if reconstruction is None or not reconstruction.size:reconstruction=SimpleNamespace(shape=project.shape)
            features=dict(spatial['features']);fts=dict(features['fts'])
            region=region_features(events,reconstruction,regions,landmarks,opts['spatialRes'],opts.get('minShow1',.2),region_names,landmark_names) if regions or landmarks else np.empty((0,0))
            fts['region']=region_mat_payload(region)
            if opts.get('networkFeatures',False):
                fts['networkAll']=network_mat_payload(event_network(events,project.shape));inside=events
                if regions:
                    allowed=np.zeros(int(np.prod(project.shape[:3])),bool)
                    for pixels in regions:allowed[pixels]=True
                    inside=[event if len(event) and allowed[event%len(allowed)].any() else np.empty(0,np.int64) for event in events]
                fts['network']=network_mat_payload(event_network(inside,project.shape))
            else:fts.pop('network',None);fts.pop('networkAll',None)
            features['fts']=fts;pending.append((channel,scope,features))
            if progress:progress('annotations',channel,scope)
    data=dict(project.data);stages=dict(data['stages']);detection=dict(stages['detection'])
    for channel in project.channels:detection[channel]=dict(detection[channel])
    for channel,scope,features in pending:detection[channel][scope]={**detection[channel][scope],'features':features}
    detection['opts']={**detection['opts'],'networkFeatures':bool(opts.get('networkFeatures',False))};stages['detection']=detection
    data['stages']=stages;data['annotations']={**annotations,'regions_dirty':False}
    data['history']=[*data['history'],{'action':'refresh_annotation_features'}]
    return Project(data,project.path)


def recompute_features(path,*,progress=None,save=False):
    from .features import compute_features
    from .fea import features_mat_payload
    from .evt import spatial_from_mat_payload
    project=path if isinstance(path,Project) else Project.open(path);annotations=project.data['annotations'];peaks_changed=False
    # Noise/correction maps can be lazy project arrays. Scientific kernels
    # receive concrete NumPy options, while the viewer keeps sliced I/O.
    opts=scalar_options(materialize(project.opts))
    for channel in project.channels:
        for scope in ('local','global'):
            spatial=project.spatial(channel,scope)
            if spatial and spatial['evt'].size and not spatial['datR'].size:
                raise ValueError('Imported MATLAB file lacks reconstructed datR. Existing features remain viewable; rerun detection to recompute propagation/landmark features.')
    for channel in project.channels:
        working=project.working_movie(channel)
        for scope in ('local','global'):
            spatial=project.spatial(channel,scope)
            if spatial is None or not spatial['evt'].size:continue
            old=spatial.get('features',{}).get('fts',{}).get('curve',{}).get('dffMaxFrame',np.empty(0))
            native=spatial_from_mat_payload(materialize(spatial),opts)
            result=compute_features(working,native,opts,channel=int(channel[-1]),regions=annotations.get('regions'),landmarks=annotations.get('landmarks'),
                region_names=annotations.get('region_names'),landmark_names=annotations.get('landmark_names'))
            new=features_mat_payload(result);spatial['features']=new
            if scope=='local' and not np.array_equal(old,new['fts']['curve'].get('dffMaxFrame',np.empty(0))):peaks_changed=True
            if progress:progress('features',channel,scope)
    if peaks_changed:project.data['stages'].pop('cfu',None)
    annotations['regions_dirty']=False;project.data['history'].append({'action':'recompute_features','cfu_invalidated':peaks_changed})
    return project.save() if save else project


def recompute_dependencies(path,settings,*,progress=None,save=False):
    from .cfu import normalize_info,analyze,cfu_mat_payload
    project=path if isinstance(path,Project) else Project.open(path);stored=project.data['stages'].get('cfu')
    if stored is None:raise ValueError('Run or import CFUs before dependency analysis')
    info1=normalize_info(stored['cfuInfo1']);info2=normalize_info(stored['cfuInfo2'])
    relation,groups=analyze(info1,info2,settings,progress=progress)
    payload=cfu_mat_payload(dict(cfuInfo1=info1,cfuInfo2=info2,cfuRelation=relation,cfuGroupInfo=groups,cfuOpts=settings,channel_count=len(project.channels)))
    # Preserve the original record containers and maps exactly.
    for key in ('cfuRelation','cfuGroupInfo'):stored[key]=payload[key]
    previous=stored['cfuOpts'];stored['cfuOpts']={**previous,'cfuAnalysis':settings['cfuAnalysis'],'cfuGroup':settings['cfuGroup']}
    project.data['history'].append({'action':'recompute_dependencies'});return project.save() if save else project


def execute_job(job,progress=None,*,return_project=False):
    """Dispatch operations without implicit project saves."""
    operation=job['operation'];target=Project(job['project_data'],job.get('project_path')) if 'project_data' in job else job.get('project')
    if operation=='run':project=run_workflow(job['source'],job.get('output'),job.get('config'),overwrite=job.get('overwrite',False),save_project=job.get('save_project',False),progress=progress)
    elif operation=='load_input':
        from .workflow_steps import load_input
        project=load_input(job['source'],job.get('config'),progress=progress)
    elif operation in ('preprocess','detect'):
        from .workflow_steps import preprocess_project,detect_project
        if not isinstance(target,Project):target=Project.open(target)
        project=(preprocess_project if operation=='preprocess' else detect_project)(target,job.get('options'),progress=progress)
    elif operation=='import':
        from .project import import_saved_project
        project=import_saved_project(job['source'],cfu_file=job.get('cfu_file'))
    elif operation=='cfu':
        target=target if isinstance(target,Project) else Project.open(target)
        target=refresh_annotation_features(target,job.get('feature_options'),progress=(lambda *args:progress({'stage':'features','detail':str(args)})) if progress else None)
        project=recompute_cfu(target,job.get('settings'),progress=(lambda *args:progress({'stage':'cfu','detail':str(args)})) if progress else None)
    elif operation=='features':project=recompute_features(target,progress=(lambda *args:progress({'stage':'features','detail':str(args)})) if progress else None)
    elif operation=='dependencies':project=recompute_dependencies(target,job['settings'],progress=(lambda *args:progress({'stage':'dependencies','detail':str(args)})) if progress else None)
    elif operation=='relink':
        from .project import relink_saved_source
        project=relink_saved_source(target if isinstance(target,Project) else Project.open(target),job['source'])
    elif operation=='export':
        from .project import export_project
        project=target if isinstance(target,Project) else Project.open(target)
        directory=export_project(project,job['directory'],selection=job.get('selection'),progress=progress,display_settings=job.get('display_settings'))
        return {'exported':str(directory)}
    else:raise ValueError('Unknown operation: '+operation)
    if return_project:return project
    return {'project':str(project.path) if project.path else None,'summary':project.summary()}
