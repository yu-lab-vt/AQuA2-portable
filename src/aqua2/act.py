"""Active regions; zero-based MATLAB/Fortran-order voxel indices."""
import numpy as np
from scipy import ndimage
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _disk_footprints():
    from .io import read_mat
    return read_mat(Path(__file__).with_name("data") / "disk-footprints-r2024b.mat")["diskFootprints"].ravel()


def _components(mask, min_voxels=1):
    from .components import connected_components
    return connected_components(mask,min_voxels)


def spatial_footprint(radius, dimensions):
    if int(radius) != radius or radius < 0:
        raise ValueError("Structuring-element radius must be a nonnegative integer")
    radius = int(radius)
    if dimensions == 2:
        if radius == 0:
            return np.ones((1,1),bool)
        footprints = _disk_footprints()
        if radius > len(footprints):
            raise ValueError("R2024b disk radii 1..128 are supported; extend the exact resource for larger radii")
        return footprints[radius-1]
    coordinates = np.ogrid[tuple(slice(-radius,radius+1) for _ in range(dimensions))]
    return sum(axis**2 for axis in coordinates) <= radius**2


def bw2reg(mask, options, *, min_voxels=1):
    distance = options.get("spaMergeDist",0)
    if distance <= 0:
        return _components(mask,min_voxels)
    if mask.shape[2] == 1:
        footprint = spatial_footprint(distance,2)[:,:,None,None]
    else:
        footprint = spatial_footprint(distance,3)[:,:,:,None]
    grown = ndimage.binary_dilation(mask,structure=footprint)
    original = mask.ravel(order="F")
    return [selected for group in _components(grown,min_voxels) if len(selected := group[original[group]])>=min_voxels]


def ac_detect(delta, options, spatial_mask=None, channel=1, *, progress=None):
    shape = delta.shape
    if delta.ndim != 4:
        raise ValueError("Expected HWLT array")
    mask = np.ones(shape[:3],bool) if spatial_mask is None else np.asarray(spatial_mask,bool).reshape(shape[:3],order="F")
    data = np.where(mask[:,:,:,None],delta,-1)
    threshold = float(options["thrARScl"])
    maximum = float(options.get(f"maxdF{channel}",min(100,np.nanmax(data))))
    max_size = float(options["maxSize"])
    circularity = float(options["circularityThr"])
    if threshold > maximum or (max_size >= mask.size and circularity == 0):
        thresholds = [threshold]
    else:
        if maximum == threshold:
            raise ValueError("MATLAB multi-threshold range has a zero step")
        thresholds = np.linspace(threshold,maximum,11)
    active = np.zeros(shape,dtype=np.uint16,order="F")
    count = 0
    for level,value in enumerate(thresholds):
        if progress:progress(f"Labeling active regions (threshold {value:g}, {level+1}/{len(thresholds)})")
        components = bw2reg((data > value) & (active == 0),options,min_voxels=max(1,int(np.ceil(options["minSize"]))) if options["compress"]>=0 else 1)
        if progress:progress("Filtering active regions", 0, len(components))
        for index,group in enumerate(components):
            if progress and index % 100 == 0:progress("Filtering active regions", index, len(components))
            h,w,z,t = np.unravel_index(group,shape,order="F")
            duration = int(t.max()-t.min()+1)
            local_shape = tuple(int(axis.max()-axis.min()+1) for axis in (h,w,z,t))
            local = np.zeros(local_shape,bool)
            local[h-h.min(),w-w.min(),z-z.min(),t-t.min()] = True
            footprint = np.sum(local,axis=3) > float(options["compress"])*duration
            size = int(footprint.sum())
            if size > max_size or size < options["minSize"] or duration < options["minDur"]:
                continue
            if circularity:
                dimension = 2 if shape[2] == 1 else 3
                region = footprint[:,:,0] if dimension == 2 else footprint
                eroded = ndimage.binary_erosion(region,structure=spatial_footprint(1,dimension),border_value=1)
                surface = size-int(eroded.sum())
                measure = (4*np.pi*size/(surface**2) if dimension == 2 else np.pi**(1/3)*(6*size)**(2/3)/surface) if surface else np.inf
                if measure <= circularity:
                    continue
            count += 1
            active.ravel(order="F")[group] = min(count,np.iinfo(np.uint16).max)
        if progress:progress("Filtering active regions", len(components), len(components))
        del components
    if progress:progress(f"Indexing active regions (accepted {count}  )")
    flat = active.ravel(order="F")
    indices = np.flatnonzero(flat)
    if not len(indices):
        if progress:progress("Active regions detected: 0  ", 1, 1)
        return []
    indices = indices[np.argsort(flat[indices],kind="stable")]
    groups = np.split(indices,np.flatnonzero(np.diff(flat[indices]))+1)
    by_label = {int(flat[group[0]]):group for group in groups}
    result = [by_label.get(i,np.empty(0,np.int64)) for i in range(1,int(flat.max())+1)]
    if progress:progress(f"Active regions detected: {len(result)}  ", 1, 1)
    return result
