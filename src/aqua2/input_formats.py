"""GUI format detection shared by preview and subsequent preprocessing."""
from pathlib import Path
import h5py
import numpy as np
import tifffile


def is_saved_input(path):
    path=Path(path)
    if path.suffix.lower()!='.mat' or not h5py.is_hdf5(path):return False
    with h5py.File(path,'r') as root:return 'res/datOrg1' in root and 'res/opts' in root


def read_tiff_channels(path):
    """Respect named T/Z/C axes; infer a size-two *extra* axis, not two pages.

    Three/four-component RGB overlays are not scientific input. Plain ZYX
    pages retain MATLAB's time-series convention; explicit TZ keeps depth.
    """
    with tifffile.TiffFile(path) as image:
        if len(image.series)!=1:raise ValueError('Load TIFF series separately.')
        series=image.series[0];axes=series.axes;shape=series.shape
        if 'Y' not in axes or 'X' not in axes:raise ValueError('TIFF has no spatial axes.')
        channels=[i for i,a in enumerate(axes) if a=='C' or a=='S']
        if not channels and len(shape)>=4:
            channels=[i for i,a in enumerate(axes) if a not in 'TYXZ' and shape[i]==2]
        if len(channels)>1:raise ValueError('Ambiguous TIFF channel axis.')
        ci=channels[0] if channels else None
        if ci is not None and shape[ci] not in (1,2):raise ValueError('Only one or two grayscale channels are supported. For RGB overlays, import the original MAT.')
        array=series.asarray()
    output=[]
    for c in range(shape[ci] if ci is not None else 1):
        selection=[slice(None)]*array.ndim
        if ci is not None:selection[ci]=c
        data=array[tuple(selection)]
        labels=axes[:ci]+axes[ci+1:] if ci is not None else axes
        extra=[a for a in labels if a not in 'YX']
        if len(extra)>2 or (len(extra)==2 and set(extra)!=set('TZ')):raise ValueError(f'Ambiguous TIFF time/depth axes: {axes}')
        order=[labels.index('Y'),labels.index('X')]+[labels.index(a) for a in ('Z','T') if a in extra]
        if len(extra)==1:order=[labels.index('Y'),labels.index('X'),labels.index(extra[0])]
        data=data.transpose(order)
        if not extra:data=data[:,:,None,None]
        elif len(extra)==1:data=data[:,:,None,:]
        output.append(data)
    return output


def read_mat_channels(path,variable=None):
    from .io import read_mat
    values=read_mat(path,[variable] if variable else None)
    array=values[variable] if variable else next(iter(values.values()))
    if not isinstance(array,np.ndarray) or array.dtype.kind not in 'buif':raise ValueError('Raw MAT requires a numeric image variable.')
    candidates=[i for i in range(2,array.ndim) if array.shape[i]==2] if array.ndim in (4,5) else []
    if len(candidates)>1:raise ValueError('Ambiguous MAT channel axis. Use TIFF with channel metadata.')
    if candidates:
        parts=[]
        for c in (0,1):
            selection=[slice(None)]*array.ndim;selection[candidates[0]]=c;parts.append(array[tuple(selection)])
    else:parts=[array]
    result=[]
    for data in parts:
        if data.ndim==2:data=data[:,:,None,None]
        elif data.ndim==3:data=data[:,:,None,:]
        if data.ndim!=4:raise ValueError('MAT requires HWT, HWLT or an array with a channel axis.')
        result.append(data)
    return result
