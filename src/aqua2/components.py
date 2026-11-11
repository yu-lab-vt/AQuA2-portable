"""Connected voxels in canonical F order, filtering sizes before Python objects."""
import numpy as np
from numba import njit,types
from numba.typed import Dict
from scipy import ndimage
from itertools import product

_SPARSE_BINARY_LIMIT=200000


@njit(cache=True)
def _group_label_pixels(flat,count,minimum):
    sizes=np.zeros(count+1,np.int64)
    for label in flat:
        if label:sizes[label]+=1
    positions=np.full(count+1,-1,np.int64);total=0;kept=0
    for label in range(1,count+1):
        if sizes[label]>=minimum:
            positions[label]=total;total+=sizes[label];kept+=1
    bounds=np.empty(kept+1,np.int64);j=0
    for label in range(1,count+1):
        if positions[label]>=0:bounds[j]=positions[label];j+=1
    bounds[kept]=total
    pixels=np.empty(total,np.int64)
    for index in range(len(flat)):
        label=flat[index]
        if label and positions[label]>=0:
            position=positions[label];pixels[position]=index;positions[label]=position+1
    return pixels,bounds


@njit(cache=True)
def _sparse_roots(pixels,shape,offsets,steps,use_hash=False):
    n=len(pixels);parent=np.arange(n);sizes=np.ones(n,np.int64)
    lookup=Dict.empty(key_type=types.int64,value_type=types.int64)
    if use_hash:
        for i in range(n):lookup[pixels[i]]=i
    strides=np.ones(len(shape),np.int64)
    for axis in range(1,len(shape)):strides[axis]=strides[axis-1]*shape[axis-1]
    coords=np.empty(len(shape),np.int64)
    for i in range(n):
        for axis in range(len(shape)):coords[axis]=(pixels[i]//strides[axis])%shape[axis]
        for k in range(len(steps)):
            valid=True
            for axis in range(len(shape)):
                coordinate=coords[axis]+offsets[k,axis]
                if coordinate<0 or coordinate>=shape[axis]:valid=False;break
            if not valid:continue
            neighbor=pixels[i]+steps[k]
            if use_hash:
                found=lookup.get(neighbor)
                if found is None:continue
                j=int(found)
            else:j=np.searchsorted(pixels,neighbor)
            if j<0 or j==n or pixels[j]!=neighbor:continue
            first=i;second=j
            while parent[first]!=first:
                parent[first]=parent[parent[first]];first=parent[first]
            while parent[second]!=second:
                parent[second]=parent[parent[second]];second=parent[second]
            if first!=second:
                if sizes[first]<sizes[second]:first,second=second,first
                parent[second]=first;sizes[first]+=sizes[second]
    for i in range(n):
        root=i
        while parent[root]!=root:root=parent[root]
        parent[i]=root
    return parent+1


def connected_components(mask,minimum=1):
    """Same full connectivity and sorted pixel/component order as legacy code.

    Singleton axes do not change adjacency or F-linear indices. A component
    with fewer than minimum voxels cannot pass any later spatial area bound
    of minimum pixels. No signal threshold or connectivity is approximated.
    """
    minimum=max(1,int(np.ceil(minimum)));mask=np.asarray(mask).squeeze()
    if mask.ndim==0:return [np.array([0],np.int64)] if mask and minimum<=1 else []
    positive=np.count_nonzero(mask)
    if not positive:return []
    if positive<=4000000 and positive*32<mask.size:
        pixels=np.flatnonzero(mask.ravel(order='F'))
        shape=np.asarray(mask.shape,np.int64);strides=np.r_[1,np.cumprod(shape[:-1])]
        offsets=np.asarray(list(product((-1,0,1),repeat=mask.ndim)),np.int64)
        steps=offsets@strides;selected=steps>0
        labels=_sparse_roots(pixels,shape,offsets[selected],steps[selected],positive>_SPARSE_BINARY_LIMIT)
        positions,bounds=_group_label_pixels(labels,len(pixels),minimum)
        groups=[pixels[positions[bounds[i]:bounds[i+1]]] for i in range(len(bounds)-1)]
        return sorted(groups,key=lambda group:group[0])
    dtype=np.int64 if mask.size>=2**31 else np.int32
    labels=np.empty(mask.shape,dtype=dtype,order='F')
    count=ndimage.label(mask,structure=np.ones((3,)*mask.ndim,bool),output=labels)
    if not count:return []
    pixels,bounds=_group_label_pixels(labels.ravel(order='F'),count,minimum)
    groups=[pixels[bounds[i]:bounds[i+1]] for i in range(len(bounds)-1)]
    return sorted(groups,key=lambda group:group[0])
