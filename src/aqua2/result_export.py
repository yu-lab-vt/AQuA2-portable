"""User-selected, MAT-free exports using the saved scientific values."""
import csv,json,os
from datetime import datetime
from pathlib import Path
import numpy as np
import tifffile
from .project_io import materialize

EXPORT_CHOICES={'event_table':'Event Features','cfu_table':'CFU Features','masked_movie':'Movie with overlay','mask_only':'Overlay only','propagation':'Rising Maps'}

def favorites(project,channel,scope,mode):
    source=project.data.get('annotations',{}).get('favorites',{});values=set()
    keys=[f'{mode}:{channel}:{scope}']
    if mode=='Event':keys += [f'{channel}:{scope}',f'Event:{channel}:{scope}']
    else:keys += [f'CFU:{channel}:local',f'CFU:{channel}:global']
    for key in keys:values.update(int(i) for i in source.get(key,[]))
    return values

def event_table(project,path,channel,scope):
    from .fea import features_from_mat_payload
    from .export import feature_table
    stored=project.features(channel,scope)
    if not stored:return False
    result=features_from_mat_payload(materialize(stored),project.opts);table=feature_table(result,channel=int(channel[-1]));n=table['event_count'];fav=favorites(project,channel,scope,'Event')
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['Feature']+[f'Event {i+1}' for i in range(n)])
        writer.writerow(['Favorite']+[int(i in fav) for i in range(n)])
        for name,row in zip(table['names'],table['values']):writer.writerow([name]+[float(x) for x in row])
    return True

def cfu_table(project,path,channel):
    records=project.cfus(channel)
    if not records:return False
    fav=favorites(project,channel,'local','CFU')
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['Channel','CFU ID','Favorite','Parent ID','Event count','Event IDs (1-based)','Spatial support (pixels/voxels)','Occurrence count','Occurrence frames (1-based)','Main frequency (Hz)','Frequency method','Peak frequency 80 (Hz)','Inter-event intervals (s)'])
        for i,r in enumerate(records):
            stats=r.get('frequencyStats',{});stats=stats if isinstance(stats,dict) else {};frames=np.flatnonzero(r['occurrence'])
            writer.writerow([int(channel[-1]),i+1,int(i in fav),int(r.get('parentId',-1))+1,len(r['eventIds']),' '.join(str(int(e)+1) for e in r['eventIds']),np.count_nonzero(np.asarray(r['weightMap'])>.1),len(frames),' '.join(str(t+1) for t in frames),np.asarray(stats.get('mainFreq',np.nan)).item(),str(stats.get('method','')),np.asarray(stats.get('peakFreq80',np.nan)).item(),' '.join(str(float(v)) for v in np.asarray(stats.get('dt',[])).ravel())])
    return True

def _export_frame(project,reader,channel,scope,mode,t,z,view='Overlay',display_settings=None):
    from .viewer_render import FrameRequest,prepare_frame
    count=len(project.cfus(channel)) if mode=='CFU' else len(project.events(channel,scope))
    settings=display_settings or {}
    from .viewer_display import DisplaySettings
    shared=DisplaySettings(**settings['display']) if 'display' in settings else None
    current=settings.get('mode')==mode and settings.get('channel')==channel and (mode=='CFU' or settings.get('scope',scope)==scope)
    if current and mode!='CFU' and not settings.get('view','').startswith('Rising Map'):view=settings.get('view',view)
    request=FrameRequest(project,t,z,channel,scope,mode,view,tuple(settings.get('selected',())) if current else (),tuple(range(count)),bool(settings.get('all_time',False)),float(settings.get('gamma',1)),float(settings.get('opacity',.5)),True,False,False,0,bool(settings.get('auto_contrast',False)),
        annotate=bool(settings.get('annotate',False)),label_font=int(settings.get('label_font',10)),display=shared)
    image=prepare_frame(request,reader)[0]
    return image[::-1] if settings.get('y_up',False) else image


def masked_movie(project,path,channel,scope,mode,emit,display_settings=None):
    from .viewer_render import FrameReader
    h,w,zcount,tcount=project.shape;reader=FrameReader()
    def frames():
        for t in range(tcount):
            for z in range(zcount):yield _export_frame(project,reader,channel,scope,mode,t,z,display_settings=display_settings)
            if t%max(1,tcount//100)==0:emit(f'{channel} {scope} {mode} masked image',t+1,tcount)
    temporary=path.with_name(path.name+'.partial')
    try:
        with tifffile.TiffWriter(temporary,bigtiff=True) as writer:
            writer.write(frames(),shape=(tcount,zcount,h,w,3),dtype=np.uint8,photometric='rgb',compression='deflate',compressionargs={'level':1},metadata={'axes':'TZYXS','TimeIncrement':float(project.opts['frameRate'])})
        os.replace(temporary,path)
    finally:
        reader.close()
        if temporary.exists():temporary.unlink()


def cfu_mean_image(project,path,channel,emit,display_settings=None):
    from .viewer_render import FrameReader
    h,w,depth,_=project.shape;reader=FrameReader()
    def planes():
        for z in range(depth):
            yield _export_frame(project,reader,channel,'local','CFU',0,z,'Mean Projection',display_settings)
            emit(f'{channel} CFUMean Projection',z+1,depth)
    temporary=path.with_name(path.name+'.partial')
    try:
        with tifffile.TiffWriter(temporary) as writer:
            writer.write(planes(),shape=(depth,h,w,3),dtype=np.uint8,photometric='rgb',compression='deflate',compressionargs={'level':1},metadata={'axes':'ZYXS','projection':'temporal mean; one frame per Z'})
        os.replace(temporary,path)
    finally:
        reader.close()
        if temporary.exists():temporary.unlink()


def pure_mask_tiff(project,path,channel,scope,mode,emit):
    """GUI label colors at full intensity, no source/reconstruction background."""
    from .viewer_model import label_layer
    h,w,depth,frames=project.shape;cfu=mode=='CFU';frames=1 if cfu else frames
    def planes():
        for t in range(frames):
            for z in range(depth):
                labels,_,palette=label_layer(project,mode,channel,scope,t,z)
                yield np.floor(palette[labels]*255+.5).astype(np.uint8)
            if t%max(1,frames//100)==0:emit(f'{channel} {scope} {mode} masks',t+1,frames)
    temporary=path.with_name(path.name+'.partial')
    try:
        with tifffile.TiffWriter(temporary,bigtiff=not cfu) as writer:
            writer.write(planes(),shape=(depth,h,w,3) if cfu else (frames,depth,h,w,3),dtype=np.uint8,
                photometric='rgb',compression='deflate',compressionargs={'level':1},
                metadata={'axes':'ZYXS' if cfu else 'TZYXS','TimeIncrement':float(project.opts['frameRate']),'mask':'GUI colors; black background; full intensity; row 0 at top'})
        os.replace(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()


def propagation_maps(project,directory,channel,scope,emit):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    spatial=project.spatial(channel,scope);items=spatial.get('riseLst',np.empty(0,object));records=[]
    for i,record in enumerate(items.ravel(order='F')):
        if 'dlyMap50' not in record:continue
        directory.mkdir(parents=True,exist_ok=True)
        rows=np.asarray(record['rgh']).ravel();cols=np.asarray(record['rgw']).ravel();planes=np.asarray(record.get('rgl',[[1]])).ravel()
        values=np.asarray(record['dlyMap50']).reshape((len(rows),len(cols),len(planes)),order='F');seconds=(values-1)*float(project.opts['frameRate'])
        prefix=f'{channel}_{scope}_event_{i+1}_rising50';path=directory/(prefix+'.tif')
        tifffile.imwrite(path,seconds.transpose(2,0,1),photometric='minisblack',metadata={'axes':'ZYX','unit':'s','origin_yxz_1based':[int(rows[0]),int(cols[0]),int(planes[0])]})
        pngs=[]
        for z,plane in enumerate(planes):
            image=seconds[:,:,z];finite=image[np.isfinite(image)]
            if not len(finite):continue
            figure=Figure(figsize=(5,4),dpi=120);FigureCanvasAgg(figure);ax=figure.add_subplot(111)
            plot=ax.imshow(np.ma.masked_invalid(image),origin='upper',cmap='jet',extent=(cols[0]-.5,cols[-1]+.5,rows[-1]+.5,rows[0]-.5),vmin=float(finite.min()),vmax=float(finite.max()) if finite.max()>finite.min() else float(finite.min())+float(project.opts['frameRate']))
            figure.colorbar(plot,ax=ax,label='Time (s)');ax.set_title(f'Event {i+1} · 50% rise · Z {int(plane)}');ax.set_xlabel('X (pixel, 1-based)');ax.set_ylabel('Y (pixel, 1-based)');figure.tight_layout()
            png=directory/(prefix+f'_z{int(plane)}.png');figure.savefig(png);pngs.append(png.name);figure.clear()
        records.append(dict(channel=channel,scope=scope,event_id=i+1,favorite=i in favorites(project,channel,scope,'Event'),numeric_file=path.name,png_files=pngs,origin_yxz_1based=[int(rows[0]),int(cols[0]),int(planes[0])],shape_yxz=list(seconds.shape)))
        if i%max(1,len(items)//100)==0:emit(f'{channel} {scope}Rising Maps',i+1,len(items))
    return records

def export_results(project,directory,*,selection=None,progress=None,display_settings=None):
    selected={key:True for key in EXPORT_CHOICES} if selection is None else {key:bool(selection.get(key,False)) for key in EXPORT_CHOICES}
    if selection is not None and set(selection)-set(EXPORT_CHOICES):raise ValueError('Unknown export selection')
    if not any(selected.values()):raise ValueError('Select at least one export item.')
    directory=Path(directory)
    if directory.exists() and any(directory.iterdir()):directory=directory/('AQuA2_export_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True,exist_ok=True)
    manifest=dict(complete=False,selection=selected,files=[],skipped=[],propagation=[],index_origin=1,annotation_index_origin=0,seconds_per_frame=float(project.opts['frameRate']),masked_movie='uint8 RGB display, events: time series; CFU: temporal mean single frame per Z; GUI intensity scaling and additive reconstructed overlay with display_settings, row 0 at top; not raw computation input',propagation_unit='seconds since frame 0',summary=project.summary())
    meta=directory/'export_metadata.json'
    settings=display_settings or {}
    manifest['display_settings']=dict(settings) if 'display' in settings else {'gamma':settings.get('gamma',1.),'opacity':settings.get('opacity',.5),'auto_contrast':settings.get('auto_contrast',False)}
    manifest['mask_only']='uint8 RGB GUI label colors, black background, full intensity; events: TZYXS; CFU: ZYXS; no raw image or reconstructed intensity'
    def write():meta.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    def emit(detail,done=None,total=None):
        if progress:progress(dict(stage='export',detail=detail,done=done,total=total))
    write()
    for channel in project.channels:
        for scope in ('local','global'):
            spatial=project.spatial(channel,scope)
            if spatial is None or not len(project.events(channel,scope)):continue
            if selected['event_table']:
                name=f'{channel}_{scope}_features.csv'
                if event_table(project,directory/name,channel,scope):manifest['files'].append(name)
                else:manifest['skipped'].append(f'{channel} {scope}: no saved event features')
            if selected['masked_movie']:
                name=f'{channel}_{scope}_events_masked.tif';masked_movie(project,directory/name,channel,scope,'Event',emit,display_settings);manifest['files'].append(name)
            if selected['mask_only']:
                name=f'{channel}_{scope}_events_mask_only.tif';pure_mask_tiff(project,directory/name,channel,scope,'Event',emit);manifest['files'].append(name)
            if selected['propagation']:
                maps=propagation_maps(project,directory/'rising_map',channel,scope,emit)
                for record in maps:
                    record['numeric_file']='rising_map/'+record['numeric_file'];record['png_files']=['rising_map/'+name for name in record['png_files']]
                manifest['propagation'].extend(maps)
                for record in maps:manifest['files'].extend([record['numeric_file'],*record['png_files']])
                if len(maps)<len(project.events(channel,scope)):manifest['skipped'].append(f'{channel} {scope}: some events have no saved rising map')
        if selected['cfu_table']:
            name=f'{channel}_cfu_features.csv'
            if cfu_table(project,directory/name,channel):manifest['files'].append(name)
            else:manifest['skipped'].append(f'{channel}: no CFUs')
        if selected['masked_movie'] and project.cfus(channel):
            name=f'{channel}_cfu_mean_masked.tif';cfu_mean_image(project,directory/name,channel,emit,display_settings);manifest['files'].append(name)
        if selected['mask_only'] and project.cfus(channel):
            name=f'{channel}_cfu_mask_only.tif';pure_mask_tiff(project,directory/name,channel,'local','CFU',emit);manifest['files'].append(name)
    manifest['complete']=True;write();emit('Export complete',1,1);return directory
