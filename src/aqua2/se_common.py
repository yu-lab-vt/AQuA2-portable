"""Temporal segmentation primitives. All public indices are zero based."""
from itertools import product
import numpy as np
from numba import njit
from skimage.morphology import reconstruction

def matlab_round(value):
    value=np.asarray(value)
    return np.sign(value)*np.floor(np.abs(value)+.5)


@njit(cache=True)
def _mean_four_lanes(values):
    output=np.empty(values.shape[1],dtype=values.dtype)
    for column in range(values.shape[1]):
        # R2024b uses the scalar kernel below 44 single-precision rows.
        # The transition is independently covered for every size 1..80.
        if values.shape[0]<44:
            value=values[0,column]
            for row in range(1,values.shape[0]):value+=values[row,column]
            output[column]=value/values.shape[0]
            continue
        accumulators=np.zeros(4,dtype=values.dtype)
        for row in range(values.shape[0]):accumulators[row%4]+=values[row,column]
        value=accumulators[0]
        for lane in range(1,4):value+=accumulators[lane]
        output[column]=value/values.shape[0]
    return output


def mean_spatial(values):
    """R2024b single spatial mean: scalar <44 rows, four lanes otherwise."""
    values=np.asarray(values)
    if values.shape[0]==0:return np.full(values.shape[1:] or (),np.nan,dtype=values.dtype)
    vector=values.ndim==1
    matrix=values[:,None] if vector else values
    result=_mean_four_lanes(np.ascontiguousarray(matrix))
    return result[0] if vector else result


@njit(cache=True)
def _resize_axis(values,indices,weights,integer):
    out=np.empty((indices.shape[0],values.shape[1]),dtype=values.dtype)
    for i in range(out.shape[0]):
        for column in range(out.shape[1]):
            total=0.0
            for k in range(indices.shape[1]):
                total+=values[indices[i,k],column]*weights[i,k]
            if integer:
                total=np.floor(total+.5)
            out[i,column]=total
    return out


@njit(cache=True)
def _resize_movie_axis(data,indices,weights,axis,integer):
    h,w,z,t=data.shape
    if axis==0:h=indices.shape[0]
    else:w=indices.shape[0]
    # Allocate F order without converting a full recording to a C matrix.
    output=np.empty((t,z,w,h),dtype=data.dtype).transpose((3,2,1,0))
    if axis==0:
        for frame in range(t):
            for depth in range(z):
                for col in range(w):
                    for row in range(h):
                        total=0.0
                        for k in range(indices.shape[1]):total+=data[indices[row,k],col,depth,frame]*weights[row,k]
                        if integer:total=np.floor(total+.5)
                        output[row,col,depth,frame]=total
    else:
        for frame in range(t):
            for depth in range(z):
                for col in range(w):
                    for row in range(h):
                        total=0.0
                        for k in range(indices.shape[1]):total+=data[row,indices[col,k],depth,frame]*weights[col,k]
                        if integer:total=np.floor(total+.5)
                        output[row,col,depth,frame]=total
    return output


def my_resize(data,scale):
    """The repo's custom equal-weight resize, NOT standard imresize bicubic."""
    original=np.asarray(data)
    logical=original.dtype==bool
    if logical:original=original.astype(np.uint8)*np.uint8(255)
    was_vector=original.ndim==1
    array=original[None,:] if was_vector else original
    if array.ndim<2:
        raise ValueError("Resize requires an array or curve")
    scales=(float(scale),)*2 if np.isscalar(scale) else tuple(float(s) for s in scale)
    if len(scales)!=2 or any(s<=0 or s>1 for s in scales):
        raise ValueError("Custom temporal resize supports positive downsampling scales <=1")
    result=array
    # dimensionOrder sorts scale ascending, preserving the row axis on ties.
    for axis in np.argsort(scales,kind="stable"):
        factor=scales[axis]
        if factor==1:
            continue
        n=result.shape[axis]; count=int(np.ceil(n*factor))
        width=1/factor; taps=int(np.floor(width+.5))
        u=np.arange(1,count+1)/factor+.5*(1-1/factor)
        left=np.floor(u-width/2).astype(np.int64)+1
        indices=left[:,None]+np.arange(taps)
        weights=np.full(indices.shape,factor,dtype=np.float64)
        excess=count*taps-n
        if excess>0:
            weights[-1,taps-excess:]=0
        weights/=weights.sum(axis=1,keepdims=True)
        mirror=np.r_[np.arange(n),np.arange(n-1,-1,-1)]
        indices=mirror[(indices-1)%(2*n)]
        alive=np.any(weights!=0,axis=0)
        indices,weights=indices[:,alive],weights[:,alive]
        if result.ndim==4:
            result=_resize_movie_axis(result,indices,weights,int(axis),result.dtype.kind in "ui")
        else:
            moved=np.moveaxis(result,axis,0)
            flat=np.ascontiguousarray(moved.reshape(n,-1))
            resized=_resize_axis(flat,indices,weights,result.dtype.kind in "ui")
            result=np.moveaxis(resized.reshape((count,)+moved.shape[1:]),0,axis)
    if logical:result=result==255
    return np.array(result.ravel() if was_vector else result,order="F")


def directions(dimensions):
    return [offset for offset in product((-1,0,1),repeat=dimensions) if any(offset)]


@njit(cache=True)
def _neighbor_indices_kernel(indices,shape,offset):
    result=np.empty(len(indices),np.int64)
    for i in range(len(indices)):
        remaining=indices[i];stride=1;value=0
        for axis in range(len(shape)):
            coordinate=remaining%shape[axis];remaining//=shape[axis]
            shifted=min(max(coordinate+offset[axis],0),shape[axis]-1)
            value+=shifted*stride;stride*=shape[axis]
        result[i]=value
    return result


def neighbor_indices(indices,shape,offset):
    indices=np.asarray(indices)
    if np.any(indices<0) or np.any(indices>=np.prod(shape)):
        raise ValueError("index out of bounds for neighbor coordinates")
    return _neighbor_indices_kernel(indices.ravel(),np.asarray(shape,np.int64),np.asarray(offset,np.int64)).reshape(indices.shape)


def label_lists(labels):
    flat=np.asarray(labels).ravel(order="F")
    indices=np.flatnonzero(flat)
    if not len(indices): return []
    indices=indices[np.argsort(flat[indices],kind="stable")]
    groups=np.split(indices,np.flatnonzero(np.diff(flat[indices]))+1)
    by_label={int(flat[group[0]]):group for group in groups}
    return [by_label.get(i,np.empty(0,np.int64)) for i in range(1,int(flat.max())+1)]


def region_crop(indices,shape,extension=0):
    coordinates=np.unravel_index(indices,shape,order="F")
    start=tuple(max(0,int(c.min())-extension) for c in coordinates)
    stop=tuple(min(size,int(c.max())+extension+1) for c,size in zip(coordinates,shape))
    slices=tuple(slice(a,b) for a,b in zip(start,stop))
    local_shape=tuple(b-a for a,b in zip(start,stop))
    local=np.ravel_multi_index(tuple(c-a for c,a in zip(coordinates,start)),local_shape,order="F")
    return slices,start,local_shape,local


def to_global(indices,local_shape,start,shape):
    return np.ravel_multi_index(tuple(c+a for c,a in zip(np.unravel_index(indices,local_shape,order="F"),start)),shape,order="F")


def correlate_curves(reference,curves):
    # MATLAB corr(single, single) returns single; keep native centering.
    reference=np.asarray(reference).ravel()
    curves=np.atleast_2d(curves)
    centered=reference-reference.mean()
    centered_curves=curves-curves.mean(axis=1,keepdims=True)
    with np.errstate(divide="ignore",invalid="ignore"):
        result=np.sum(centered_curves*centered,axis=1)/np.sqrt(np.sum(centered_curves**2,axis=1)*np.sum(centered**2))
    return np.clip(result,-1,1)


def impose_minima(surface,markers):
    dtype=surface.dtype
    image=np.asarray(surface)
    finite=image[np.isfinite(image)]
    span=np.ptp(finite) if len(finite) else 0
    height=.1 if span==0 else span*.001
    floor=np.where(markers,-np.inf,np.inf).astype(dtype)
    mask=np.minimum(image+np.array(height,dtype=dtype),floor)
    # MATLAB implements erosion reconstruction through imcomplement. Keeping
    # both complements is essential for its single-precision plateau values.
    imposed=reconstruction(np.array(1,dtype=dtype)-floor,np.array(1,dtype=dtype)-mask,
        method="dilation",footprint=np.ones((3,)*image.ndim,bool))
    return (np.array(1,dtype=dtype)-imposed).astype(dtype)


def marker_watershed(surface,markers):
    imposed=impose_minima(surface,markers)
    from .se_watershed import meyer_watershed
    return meyer_watershed(imposed,markers)


def fill_gaps(labels,pixels,offsets=None):
    shape=labels.shape; flat=labels.ravel(order="F")
    pending=np.asarray(pixels,dtype=np.int64)
    offsets=directions(len(shape)) if offsets is None else offsets
    while len(pending):
        before=len(pending)
        for offset in offsets:
            neighbors=neighbor_indices(pending,shape,offset)
            select=flat[neighbors]>0
            flat[pending[select]]=flat[neighbors[select]]
            pending=pending[~select]
        if len(pending)==before:
            raise RuntimeError("Watershed gap has no path to a labeled voxel")
    return labels
