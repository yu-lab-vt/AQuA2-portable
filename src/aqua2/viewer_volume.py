"""Float display volumes for GPU ray casting; scientific arrays stay intact."""
from dataclasses import dataclass
import numpy as np


@dataclass
class VolumeScene:
    volumes:tuple
    overlay:np.ndarray
    side:bool
    z_scale:float
    y_up:bool
    opacity:float
    gradient:float


def display_volume(array,limits,settings):
    low,high=limits;span=max(float(high-low),1e-12)
    minimum=low+span*settings.minimum;maximum=low+span*settings.maximum
    values=np.clip((np.asarray(array,dtype=np.float32)-minimum)/(maximum-minimum),0,1)
    if settings.contrast!=.5:values=(values-.5)*2**(4*(settings.contrast-.5))+.5
    values*=2**(4*(settings.brightness-.5))
    return np.clip(np.nan_to_num(values,nan=0.,posinf=1.,neginf=0.),0,1).astype(np.float32)


def prepare_volume(request,reader=None):
    from .viewer_render import intensity_limits
    from .viewer_model import label_layer
    from .viewer_display import DisplaySettings
    p=request.project;c=request.channel;settings=request.display or DisplaySettings()
    def frames(kind='stored'):
        return np.stack([(reader.frame(p,request.t,z,c,kind) if reader else p.frame(request.t,z,c,kind)) for z in range(p.shape[2])],axis=2)
    raw=display_volume(frames(),intensity_limits(p,c),settings)
    if request.view in ('Mean Projection','Max Projection'):
        volume=display_volume(np.stack([p.projection(c,z,request.view=='Max Projection') for z in range(p.shape[2])],axis=2),intensity_limits(p,c),settings)
    elif request.view=='dF / σ':volume=display_volume(frames('dF'),intensity_limits(p,c,'dF'),settings)
    else:volume=raw
    overlay=np.zeros((*raw.shape,4),np.float32)
    if request.overlay and request.view!='Original':
        for z in range(p.shape[2]):
            labels,alpha,palette=label_layer(p,request.mode,c,request.scope,request.t,z,all_time=request.all_time,visible=request.visible)
            overlay[:,:,z,:3]=palette[labels];overlay[:,:,z,3]=alpha*(labels>0)*settings.overlay_brightness
            for index in request.selected:
                overlay[:,:,z,:3][labels==index+1]=(.55,.85,1.) if request.mode=='CFU' else (1.,.92,.15)
    for kind,tint in (('regions',(0.,.85,.9)),('landmarks',(1.,.8,.2))):
        for pixels in p.data.get('annotations',{}).get(kind,[]):
            coords=np.unravel_index(np.asarray(pixels,np.int64),p.shape[:3],order='F')
            overlay[coords]=(*tint,.25)
    return VolumeScene((raw,volume),overlay,request.side,request.z_scale,request.y_up,1-settings.transparency,request.gradient_opacity),None,None,None
