"""Meyer flooding, deterministic F-order FIFO ties, one queue entry per voxel."""
import numpy as np
from numba import njit
import heapq
from .se_common import directions


@njit(cache=True)
def _neighbor(pixel,offset,shape,strides):
    result=pixel
    for axis in range(len(shape)):
        coordinate=(pixel//strides[axis])%shape[axis]
        shifted=coordinate+offset[axis]
        if shifted<0 or shifted>=shape[axis]:return -1
        result+=offset[axis]*strides[axis]
    return result


@njit(cache=True, inline="always")
def _is_interior(pixel,shape,strides):
    for axis in range(len(shape)):
        coordinate=(pixel//strides[axis])%shape[axis]
        if coordinate==0 or coordinate==shape[axis]-1:return False
    return True


@njit(cache=True)
def _meyer(image,labels,shape,offsets):
    n=len(image);queued=np.zeros(n,np.uint8)
    strides=np.ones(len(shape),np.int64)
    for axis in range(1,len(shape)):strides[axis]=strides[axis-1]*shape[axis-1]
    # Interior voxels need no repeated coordinate conversion for each neighbor.
    # Keep the exact offset order, heap priorities and FIFO age used before.
    steps=np.zeros(len(offsets),np.int64)
    for j in range(len(offsets)):
        for axis in range(len(shape)):steps[j]+=offsets[j,axis]*strides[axis]
    heap=[(0.0,np.int64(0),np.int64(0))];heap.pop();age=np.int64(0)
    for p in range(n):
        if labels[p]>0:queued[p]=2;continue
        interior=_is_interior(p,shape,strides)
        for j in range(len(offsets)):
            q=p+steps[j] if interior else _neighbor(p,offsets[j],shape,strides)
            if q>=0 and labels[q]>0:
                heapq.heappush(heap,(np.float64(image[p]),age,np.int64(p)));age+=1;queued[p]=1;break
    while heap:
        height,_,p=heapq.heappop(heap)
        interior=_is_interior(p,shape,strides)
        label=0;conflict=False
        for j in range(len(offsets)):
            q=p+steps[j] if interior else _neighbor(p,offsets[j],shape,strides)
            if q>=0 and labels[q]>0:
                if label==0:label=labels[q]
                elif label!=labels[q]:conflict=True;break
        queued[p]=2
        if conflict or label==0:continue
        labels[p]=label
        for j in range(len(offsets)):
            q=p+steps[j] if interior else _neighbor(p,offsets[j],shape,strides)
            if q>=0 and queued[q]==0:
                heapq.heappush(heap,(np.float64(image[q]),age,np.int64(q)));age+=1;queued[q]=1
    return labels


def meyer_watershed(image,markers):
    from .act import _components
    labels=np.zeros(image.shape,np.int32,order="F")
    for i,group in enumerate(_components(markers)):labels.ravel(order="F")[group]=i+1
    offsets=np.asarray(directions(image.ndim),dtype=np.int64)
    output=_meyer(image.ravel(order="F"),labels.ravel(order="F"),np.asarray(image.shape,np.int64),offsets)
    return np.array(output.reshape(image.shape,order="F"),order="F")
