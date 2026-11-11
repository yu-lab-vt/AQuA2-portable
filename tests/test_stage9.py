"""Equivalence against the original flooding loop, including FIFO ties/borders."""
import heapq, unittest
import numpy as np
from aqua2.se_common import directions
from aqua2.se_watershed import _meyer


def original(image,labels,shape,offsets):
    # Frozen pre-stage9 semantics; intentionally use direct coordinate checks.
    strides=np.r_[1,np.cumprod(shape[:-1])];queued=np.zeros(len(image),np.uint8);heap=[];age=0
    def neighbor(p,offset):
        coords=(p//strides)%shape;new=coords+offset
        return -1 if np.any(new<0) or np.any(new>=shape) else int(p+offset@strides)
    for p in range(len(image)):
        if labels[p]>0:queued[p]=2;continue
        for offset in offsets:
            q=neighbor(p,offset)
            if q>=0 and labels[q]>0:heapq.heappush(heap,(float(image[p]),age,p));age+=1;queued[p]=1;break
    while heap:
        _,_,p=heapq.heappop(heap);label=0;conflict=False
        for offset in offsets:
            q=neighbor(p,offset)
            if q>=0 and labels[q]>0:
                if label==0:label=labels[q]
                elif label!=labels[q]:conflict=True;break
        queued[p]=2
        if conflict or label==0:continue
        labels[p]=label
        for offset in offsets:
            q=neighbor(p,offset)
            if q>=0 and queued[q]==0:heapq.heappush(heap,(float(image[q]),age,q));age+=1;queued[q]=1
    return labels


class FloodingEquivalence(unittest.TestCase):
    def test_ties_borders_and_singleton_axes(self):
        rng=np.random.default_rng(20261105)
        for shape in [(1,9),(9,1),(7,8),(6,7,5),(4,5,1,4),(3,4,3,3)]:
            shape=np.array(shape,np.int64);size=int(np.prod(shape));offsets=np.array(directions(len(shape)),np.int64)
            for flat in [True,False]:
                image=np.zeros(size,np.float32) if flat else rng.integers(-2,4,size).astype(np.float32)
                labels=np.zeros(size,np.int32);labels[[0,size//2,size-1]]=[1,2,3]
                np.testing.assert_array_equal(original(image,labels.copy(),shape,offsets),_meyer(image,labels.copy(),shape,offsets))
    def test_no_markers(self):
        shape=np.array([7,8,5],np.int64);image=np.zeros(np.prod(shape),np.float32);labels=np.zeros(len(image),np.int32)
        np.testing.assert_array_equal(labels,_meyer(image,labels.copy(),shape,np.array(directions(3),np.int64)))

    def test_clipped_neighbor_indices(self):
        from aqua2.se_common import neighbor_indices
        rng=np.random.default_rng(92)
        for shape in [(1,7),(7,8,5),(5,6,1,7),(4,5,3,6)]:
            indices=rng.integers(0,np.prod(shape),size=100)
            for offset in directions(len(shape)):
                coords=np.unravel_index(indices,shape,order="F")
                expected=np.ravel_multi_index(tuple(np.clip(c+d,0,n-1) for c,d,n in zip(coords,offset,shape)),shape,order="F")
                np.testing.assert_array_equal(expected,neighbor_indices(indices,shape,offset))

    def test_sorted_pixel_union(self):
        from aqua2.cfu_spatial import _pixel_union
        rng=np.random.default_rng(71)
        for n,m in [(0,0),(0,10),(10,0),(1,1),(500,700),(5000,100)]:
            a=np.unique(rng.integers(0,10000,n));b=np.unique(rng.integers(0,10000,m))
            np.testing.assert_array_equal(np.union1d(a,b),_pixel_union(a,b))
