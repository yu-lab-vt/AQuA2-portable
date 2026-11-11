"""Display-only isosurfaces. No scientific data are resampled or modified."""
from dataclasses import dataclass
import numpy as np
from skimage.measure import marching_cubes
from .viewer_display import display_gray


@dataclass
class SurfaceScene:
    vertices:np.ndarray  # interleaved XYZ, normal, RGB, alpha, result-only
    side:bool


def surface_triangles(volume,level,colour=(.8,.8,.8),opacity=1.,result_only=False,spacing=(1,1,1)):
    """Pad to close boundary-touching objects; coordinates are original XYZ."""
    if not np.isfinite(volume).any() or np.nanmax(volume)<=level:return np.empty((0,11),np.float32)
    clean=np.nan_to_num(volume,nan=0.,posinf=0.,neginf=0.).astype(np.float32)
    vertices,faces,normals,_=marching_cubes(np.pad(clean,1),level=level,spacing=spacing,allow_degenerate=False)
    vertices-=np.asarray(spacing);vertices=vertices[:,[1,0,2]]
    normals/=np.asarray(spacing);normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12);normals=normals[:,[1,0,2]]
    out=np.empty((faces.size,11),np.float32);out[:,:3]=vertices[faces.ravel()];out[:,3:6]=normals[faces.ravel()]
    out[:,6:9]=colour;out[:,9]=opacity;out[:,10]=float(result_only)
    return out


def prepare_surface(request,reader=None):
    from .viewer_render import intensity_limits
    from .viewer_model import label_layer,color
    p=request.project;c=request.channel;kind='dF' if request.view=='dF / σ' else 'stored'
    # Slice reader preserves lazy HDF I/O and MATLAB axis order.
    volume=np.stack([(reader.frame(p,request.t,z,c,kind) if reader else p.frame(request.t,z,c,kind)) for z in range(p.shape[2])],axis=2)
    if request.view in ('Mean Projection','Max Projection'):
        volume=np.stack([p.projection(c,z,request.view=='Max Projection') for z in range(p.shape[2])],axis=2)
    limits=intensity_limits(p,c,kind);settings=request.display
    gray=display_gray(volume,limits,settings).astype(np.float32)/255
    # Bound background mesh complexity; masks stay at full spatial resolution.
    step=max(1,int(np.ceil(max(volume.shape)/128)));spacing=(step,step,step*request.z_scale)
    peak=float(gray.max());level=max(1e-6,peak*request.surface_level)
    shade=float(np.clip(.8*2**(4*(settings.brightness-.5)),0,1))
    comparison=request.side and request.view not in ('Original','Overlay')
    pieces=[surface_triangles(gray[::step,::step,::step],level,(shade,)*3,opacity=1-settings.transparency,result_only=comparison,spacing=spacing)]
    if comparison:
        raw=np.stack([(reader.frame(p,request.t,z,c) if reader else p.frame(request.t,z,c)) for z in range(p.shape[2])],axis=2)
        raw=display_gray(raw,intensity_limits(p,c),settings).astype(np.float32)/255
        mesh=surface_triangles(raw[::step,::step,::step],max(1e-6,float(raw.max())*request.surface_level),(shade,)*3,1-settings.transparency,spacing=spacing)
        mesh[:,10]=-1;pieces.append(mesh)
    if request.overlay and request.view!='Original':
        labels=np.stack([label_layer(p,request.mode,c,request.scope,request.t,z,all_time=request.all_time,visible=request.visible)[0] for z in range(p.shape[2])],axis=2)
        for value in np.unique(labels):
            if value==0:continue
            selected=int(value)-1 in request.selected
            tint=(.55,.85,1.) if selected and request.mode=='CFU' else (1,.92,.15) if selected else color(int(value)-1)
            pieces.append(surface_triangles((labels==value).astype(np.float32),.5,tint,max(.05,settings.overlay_brightness),True,(1,1,request.z_scale)))
    for kind,tint in (('regions',(0.,.85,.9)),('landmarks',(1.,.8,.2))):
        for pixels in p.data.get('annotations',{}).get(kind,[]):
            mask=np.zeros(p.shape[:3],np.float32,order='F');mask.ravel(order='F')[np.asarray(pixels,np.int64)]=1
            pieces.append(surface_triangles(mask,.5,tint,.4,False,(1,1,request.z_scale)))
    vertices=np.concatenate(pieces)
    if len(vertices):
        shape=np.asarray([p.shape[1]-1,p.shape[0]-1,(p.shape[2]-1)*request.z_scale]);scale=max(shape.max(),1)
        vertices[:,:3]=(vertices[:,:3]-shape/2)/scale*2
        if request.y_up:vertices[:,1]*=-1;vertices[:,4]*=-1
    return SurfaceScene(vertices,request.side),None,None,None
