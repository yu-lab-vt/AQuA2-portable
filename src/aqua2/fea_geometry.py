"""Pixel-center boundaries, morphology, and MATLAB-style bicubic resize."""
import numpy as np
from scipy import ndimage
from numba import njit


def close_mask(mask):
    footprint=np.ones((3,)*mask.ndim,bool)
    return ndimage.binary_erosion(ndimage.binary_dilation(mask,structure=footprint),structure=footprint,border_value=0)


@njit(cache=True)
def _trace(mask,start,connectivity):
    offsets=np.array([[0,-1],[-1,-1],[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1]],np.int64)
    h,w=mask.shape;n=len(offsets);r,c=start[0],start[1];back_r,back_c=r,c-1
    result=[(r,c)];first_r=-1;first_c=-1
    for _ in range(16*mask.size+1):
        begin=0
        for j in range(n):
            if r+offsets[j,0]==back_r and c+offsets[j,1]==back_c:begin=j;break
        found=False
        for step in range(n):
            j=(begin+step)%n;rr=r+offsets[j,0];cc=c+offsets[j,1]
            if connectivity==4 and j%2:continue
            if 0<=rr<h and 0<=cc<w and mask[rr,cc]:
                found=True;previous=(j-1)%n;break
        if not found:
            result.append((r,c));return np.asarray(result,np.int64)
        if first_r<0:first_r,first_c=rr,cc
        elif r==start[0] and c==start[1] and rr==first_r and cc==first_c:return np.asarray(result,np.int64)
        back_r,back_c=r+offsets[previous,0],c+offsets[previous,1];r,c=rr,cc;result.append((r,c))
    raise RuntimeError("Boundary tracing failed to close")


def boundaries(mask,connectivity=8,holes=True):
    mask=np.asarray(mask,bool)
    structure=ndimage.generate_binary_structure(2,1) if connectivity==4 else np.ones((3,3),bool)
    labels,n=ndimage.label(mask,structure=structure)
    groups=[]
    for label in range(1,n+1):
        pixels=np.flatnonzero((labels==label).ravel(order="F"));groups.append(pixels)
    groups.sort(key=lambda p:p[0]);output=[]
    for pixels in groups:
        p=np.unravel_index(pixels[0],mask.shape,order="F")
        output.append(_trace(mask,np.asarray(p,np.int64),connectivity))
    if holes:
        other=4 if connectivity==8 else 8
        filled=ndimage.binary_fill_holes(mask,structure=ndimage.generate_binary_structure(2,1 if other==4 else 2))
        output.extend(boundaries(filled&~mask,other,False))
    return output


def perimeter(mask):
    total=0.0
    for boundary in boundaries(mask,8,False):
        delta=np.diff(boundary,axis=0)**2
        if len(delta)>1:
            corner=np.any(np.diff(np.vstack((delta,delta[0])),axis=0)!=0,axis=1)
            even=np.any(delta==0,axis=1)
            total+=even.sum()*.980+(~even).sum()*1.406-corner.sum()*.091
    return total


def _cubic(x):
    a=np.abs(x);return np.where(a<=1,1.5*a**3-2.5*a**2+1,np.where(a<=2,-.5*a**3+2.5*a**2-4*a+2,0))


def bicubic_resize(data,scale):
    """Standard imresize default: cubic, antialias, symmetric boundary."""
    result=np.asarray(data)
    if scale==1:return result.copy()
    logical=result.dtype==bool
    if logical:result=result.astype(np.uint8)*np.uint8(255)
    dtype=result.dtype
    if not 0<scale<=1:raise ValueError("Only positive downsampling is used by landmark features")
    for axis in (0,1):
        n=result.shape[axis];count=int(np.ceil(n*scale));width=4/scale
        position=np.arange(1,count+1)/scale+.5*(1-1/scale)
        left=np.floor(position-width/2).astype(int)
        indices=left[:,None]+np.arange(int(np.ceil(width))+2)
        weights=scale*_cubic(scale*(position[:,None]-indices));weights/=weights.sum(axis=1,keepdims=True)
        keep=np.any(weights!=0,axis=0);indices=indices[:,keep];weights=weights[:,keep]
        mirror=np.r_[np.arange(n),np.arange(n-1,-1,-1)]
        indices=mirror[(indices-1)%(2*n)]
        moved=np.moveaxis(result,axis,0)
        expanded=moved[indices]
        result=np.moveaxis(np.sum(expanded*weights.reshape(weights.shape+(1,)*(moved.ndim-1)),axis=1),0,axis)
        if dtype.kind in "ui":result=np.floor(np.clip(result,np.iinfo(dtype).min,np.iinfo(dtype).max)+.5).astype(dtype)
        elif dtype==np.float32:result=result.astype(np.float32)
    return result>128 if logical else result
