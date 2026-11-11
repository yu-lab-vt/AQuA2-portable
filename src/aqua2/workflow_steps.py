"""Interactive, in-memory input -> preprocessing -> detection -> CFU steps.

Reuse the scientific kernels without quantizing the working input between
steps. No intermediate checkpoints or implicit project writes.
"""
import copy
from pathlib import Path
import numpy as np
from .project import Project,new_project
from .project_io import materialize
from .config import read_preset
from .pipeline import preprocess,scalar_options
from .workflow import _file_identity,stored_intensities


def phase(project):
    if 'cfu' in project.data['stages']:return 'cfu'
    if 'detection' in project.data['stages']:return 'detection'
    if project.data.get('analysis_state',{}).get('preprocessed'):return 'preprocess'
    return 'input'


def can_detect(project):
    return bool(project and project.data.get('runtime',{}).get('preprocessed')
                and all('working' in project.data['movies'][c] and 'dF' in project.data['movies'][c] for c in project.channels))


def display_for(project):
    stage=phase(project)
    return ('CFU','Mean Projection') if stage=='cfu' else ('Event','Overlay') if stage=='detection' else ('Event','dF / σ') if stage=='preprocess' else ('Event','Original')


def preprocessing_input(project):
    """Only send source/settings/annotations when rereading the input.

    Sending an old multi-GB working movie to a worker that will discard it
    delays a parameter rerun and doubles memory without any scientific use.
    """
    data={key:value for key,value in project.data.items() if key not in ('movies','stages','opts','runtime')}
    data.update(opts=project.opts,movies={},stages={},analysis_state={'preprocessed':phase(project)!='input'})
    return data


def input_options(source,config):
    from .io import read_mat
    if config.get('saved_result'):
        opts=scalar_options(read_mat(source,['res/opts'])['res/opts'])
    else:
        opts=read_preset(config.get('parameters') or Path(__file__).with_name('data')/'parameters_for_batch.csv',int(config.get('preset',1)))
    return {**opts,**scalar_options(config.get('options',{}))}


def _unchanged(project):
    identity=project.data['source'];path=Path(identity['path']);current=path.stat()
    if 'mtime_ns' in identity and (current.st_size,current.st_mtime_ns)!=(identity['bytes'],identity['mtime_ns']):
        raise ValueError('Input changed. Reload it before analysis.')
    return path


def load_input(source,config=None,*,progress=None):
    """Read the input only. Show uncropped raw pixels before any correction."""
    from .io import read_tiff_hwlt,read_saved_dat_org
    from .pipeline import _read_raw
    config=copy.deepcopy(config or {});source=Path(source).resolve()
    if progress:progress(dict(stage='input',detail='Loading image'))
    if config.get('auto_channels'):
        from .input_formats import is_saved_input
        config['saved_result']=is_saved_input(source)
    identity=_file_identity(source);opts=input_options(source,config);movies={}
    if not config.get('cfu_settings'):
        from .config import read_parameter_bundle
        _,config['cfu_settings']=read_parameter_bundle(config.get('parameters') or Path(__file__).with_name('data')/'parameters_for_batch.csv',int(config.get('preset',1)))
    sources=[source]
    if config.get('saved_result'):
        sources=[source] if bool(opts['singleChannel']) else [source,source]
    elif config.get('channel2'):sources.append(Path(config['channel2']).resolve())
    identities=[identity]+[_file_identity(path) for path in sources[1:] if path!=source]
    detected=None
    if config.get('auto_channels') and not config.get('saved_result') and source.suffix.lower() in ('.tif','.tiff') and len(sources)==1:
        from .input_formats import read_tiff_channels
        detected=read_tiff_channels(source);sources=[source]*len(detected)
    elif config.get('auto_channels') and not config.get('saved_result') and source.suffix.lower()=='.mat' and len(sources)==1:
        from .input_formats import read_mat_channels
        detected=read_mat_channels(source,config.get('variable'));sources=[source]*len(detected)
    for ch,path in enumerate(sources,1):
        if config.get('saved_result'):array=read_saved_dat_org(path,channel=ch)
        elif detected is not None:array=detected[ch-1]
        elif path.suffix.lower() in ('.tif','.tiff'):
            import tifffile
            with tifffile.TiffFile(path) as image:
                series=image.series[0]
                if len(image.series)==1 and len(series.axes)==4 and set(series.axes)==set('TZYX'):
                    array=series.asarray().transpose(tuple(series.axes.index(axis) for axis in 'YXZT'))
                else:array=read_tiff_hwlt(path)
        else:array=_read_raw(path,config.get('variable'))[0]
        # Original normalization is still performed on the complete source
        # during preprocessing. A preview's frame subset must not change it.
        if not config.get('saved_result'):
            opts[f'minValueDat{ch}']=float(np.nanmin(array));opts[f'maxValueDat{ch}']=float(np.nanmax(array))
        if config.get('frames') is not None:array=array[:,:,:,slice(*config['frames'])]
        if min(array.shape)<1:raise ValueError('Frame range is empty.')
        if movies and array.shape!=movies['channel1']['stored'].shape:raise ValueError('Channel dimensions must match.')
        movies[f'channel{ch}']={'stored':array}
    opts['sz']=movies['channel1']['stored'].shape;opts['singleChannel']=len(movies)==1
    data=new_project(config.get('title') or source.stem,identity)
    data.update(config=config,opts=opts,movies=movies,input_identities=identities,analysis_state={'preprocessed':False})
    data['history'].append({'action':'load_input'})
    project=Project(data)
    for item in identities:
        current=Path(item['path']).stat()
        if (current.st_size,current.st_mtime_ns)!=(item['bytes'],item['mtime_ns']):raise ValueError('Input changed during import')
    return project


def preprocess_project(project,options=None,*,progress=None):
    """Always start from the source, not an already corrected/quantized movie."""
    from .features import mask_options_from_files
    from .fea_region import mask_pixels
    source=_unchanged(project)
    for identity in project.data.get('input_identities',[]):
        current=Path(identity['path']).stat()
        if (current.st_size,current.st_mtime_ns)!=(identity['bytes'],identity['mtime_ns']):raise ValueError('Input channel changed. Reload the image.')
    config=copy.deepcopy(project.data.get('config',{}))
    config.setdefault('saved_result',project.data['source'].get('mode')=='import_saved')
    base=input_options(source,config);base.update(options or {})
    # Do not carry data-dependent noise/correction maps back into prep1.
    config['options']={**config.get('options',{}),**dict(options or {})}
    frames=config.get('frames');frames=slice(*frames) if frames is not None else None
    stage=preprocess(source,base,config.get('channel2'),variable=config.get('variable'),saved_result=config.get('saved_result',False),
        frames=frames,spatial_mask=config.get('mask'),gui_median=config.get('gui_median',False),detect_activity=False,
        progress=(lambda detail,done=None,total=None:progress(dict(stage='preprocessing',detail=detail,done=done,total=total))) if progress else None,
        auto_channels=config.get('auto_channels',False))
    for ch in (1,2):stage.pop(f'intermediate{ch}',None)
    opts=dict(stage['opts']);opts['sz']=np.asarray(opts['sz'],dtype=float).reshape(1,4);opts['cut']=float(opts['cut']);movies={}
    for ch in (1,2):
        if f'datOrg{ch}' in stage:
            movie=stage[f'datOrg{ch}'];movies[f'channel{ch}']=dict(working=movie,stored=stored_intensities(movie,opts,ch),dF=stage[f'dF{ch}'])
    data=dict(project.data);data.update(config=config,opts=opts,movies=movies,stages={'preprocessing':{'opts':opts}},
        runtime={'preprocessed':True},analysis_state={'preprocessed':True})
    annotations=copy.deepcopy(project.data['annotations']);annotations['favorites']={};annotations['hidden']={}
    if phase(project)=='input' or tuple(project.shape[:3])!=tuple(movies['channel1']['stored'].shape[:3]):
        masks=mask_options_from_files(config.get('regions'),config.get('landmarks'))
        for key in ('regions','landmarks'):
            target=movies['channel1']['stored'].shape[:3];source_shape=config.get('preview_mask_coordinates',{}).get(key)
            if source_shape is not None:
                source_shape=tuple(source_shape);gap=int(opts.get('regMaskGap',0));converted=[]
                for pixels in mask_pixels(masks.get(key),source_shape):
                    coords=np.array(np.unravel_index(pixels,source_shape,order='F'));coords[:2]-=gap
                    inside=np.all((coords>=0)&(coords<np.array(target)[:,None]),axis=0)
                    if not inside.any():raise ValueError('Mask is empty after cropping: '+key)
                    converted.append(np.ravel_multi_index(coords[:,inside],target,order='F'))
                annotations[key]=converted
            else:annotations[key]=mask_pixels(masks.get(key),target)
            annotations.pop(key[:-1]+'_names',None)
            if key[:-1]+'_names' in masks:annotations[key[:-1]+'_names']=masks[key[:-1]+'_names']
    annotations['regions_dirty']=bool(annotations.get('regions') or annotations.get('landmarks'))
    data['annotations']=annotations;data['history']=[*data['history'],{'action':'preprocess'}]
    _unchanged(project)
    return Project(data,project.path)


def detect_project(project,options=None,*,progress=None):
    from .act import ac_detect
    from .detection import detect_from_stage3,detection_mat_payload
    from .io import read_working_mask
    if not can_detect(project):raise ValueError('Run preprocessing first, including after reopening a saved project, to retain full precision.')
    opts=scalar_options(materialize(project.opts));opts.update(options or {})
    # Preprocessing parameters cannot be silently applied to an old dF.
    from .viewer_steps import PREPROCESS_KEYS
    for key in PREPROCESS_KEYS:
        if key in (options or {}) and not np.array_equal(options[key],project.opts.get(key)):
            raise ValueError('Parameters changed. Rerun preprocessing: '+key)
    config=project.data.get('config',{});stage={'opts':opts}
    mask=read_working_mask(config['mask'],project.shape[:3],opts.get('regMaskGap',0)) if config.get('mask') else None
    for ch in (1,2):
        channel=f'channel{ch}'
        if channel not in project.channels:continue
        movie=project.data['movies'][channel];stage[f'datOrg{ch}']=np.asarray(movie['working']);stage[f'dF{ch}']=np.asarray(movie['dF'])
        emit=(lambda detail,done=None,total=None:progress(dict(stage='activity',detail=detail,done=done,total=total))) if progress else None
        stage[f'activeRegions{ch}']=ac_detect(stage[f'dF{ch}'],opts,mask,ch,progress=emit)
    annotations=copy.deepcopy(project.data['annotations'])
    masks={key:annotations.get(key) for key in ('regions','landmarks','region_names','landmark_names')}
    result=detect_from_stage3(stage,spatial_mask=mask,with_features=True,**masks,
        progress=(lambda detail,done=None,total=None:progress(dict(stage='detection',detail=detail,done=done,total=total))) if progress else None)
    data=dict(project.data);data['stages']={**data['stages'],'detection':detection_mat_payload(result)};data['stages'].pop('cfu',None)
    data['config']={**config,'options':{**config.get('options',{}),**dict(options or {})}}
    annotations['favorites']={};annotations['hidden']={};annotations['regions_dirty']=False;data['annotations']=annotations
    data['history']=[*data['history'],{'action':'detection'}]
    return Project(data,project.path)
