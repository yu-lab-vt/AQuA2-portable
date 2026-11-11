"""Original integer-shift registration; independent validation is step 9."""
import numpy as np
from scipy import fft


def cal_cc(moving, reference):
    shape = tuple(2*s-1 for s in moving.shape)
    return fft.ifftn(fft.fftn(moving.astype(np.float64),shape)*fft.fftn(np.flip(reference).astype(np.float64),shape)).real


def reg_cross_correlation(data1, data2=None):
    if data1.shape[3] < 10:
        raise ValueError("MATLAB registration requires the first 10 frames")
    one = np.array(data1,copy=True,order="F")
    two = None if data2 is None else np.array(data2,copy=True,order="F")
    reference = np.mean(one[:,:,:,0:10],axis=3)
    reference = reference-np.median(reference)
    shape = one.shape[:3]
    shifts = []
    for t in range(one.shape[3]):
        moving = one[:,:,:,t]-np.median(one[:,:,:,t])
        correlation = cal_cc(moving,reference)
        peak = np.unravel_index(np.argmax(correlation.ravel(order="F")),correlation.shape,order="F")
        shift = tuple(size-index-1 for size,index in zip(shape,peak))
        shifts.append(shift)
        source = tuple(slice(max(0,-s),min(size,size-s)) for size,s in zip(shape,shift))
        target = tuple(slice(max(0,s),min(size,size+s)) for size,s in zip(shape,shift))
        one[target+(t,)] = one[source+(t,)].copy()
        if two is not None:
            two[target+(t,)] = two[source+(t,)].copy()
    shifts = np.array(shifts)
    starts,stops = shifts.max(axis=0),np.array(shape)+shifts.min(axis=0)
    if np.any(starts < 0) or np.any(stops > shape):
        raise ValueError("Original registration crop falls outside the movie bounds")
    crop = tuple(slice(int(start),int(stop)) for start,stop in zip(starts,stops))
    return one[crop+(slice(None),)],None if two is None else two[crop+(slice(None),)]
