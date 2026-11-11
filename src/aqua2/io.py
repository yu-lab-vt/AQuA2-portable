"""TIFF and scientific MAT exchange; no GUI objects or implicit squeezing."""

from pathlib import Path
import warnings

import h5py
import numpy as np
from scipy.io import loadmat, savemat
from scipy.sparse import issparse
import tifffile


class MatStructArray(np.ndarray):
    """Distinguish MATLAB struct arrays from cells containing scalar structs."""

    def __array_finalize__(self, parent):
        self.mat_fields = getattr(parent, "mat_fields", ())


def read_tiff_hwlt(path: str | Path) -> np.ndarray:
    """Read grayscale stacks as [H,W,1,T], treating ZYX pages as time like MATLAB."""
    with tifffile.TiffFile(path) as image:
        if len(image.series) != 1:
            raise ValueError("Multiple TIFF series require an explicit selection")
        series = image.series[0]
        if series.axes == "YX":
            return series.asarray()[:, :, None, None]
        # MATLAB readTiffSeq treats a page stack as 2D+time even when ImageJ
        # describes its pages as Z slices. Preserve that default for ZYX.
        if series.axes not in ("TYX", "IYX", "QYX", "ZYX"):
            raise ValueError(f"Unsupported TIFF axes {series.axes!r}; no axis guessing")
        return np.transpose(series.asarray(), (1, 2, 0))[:, :, None, :]


def read_saved_dat_org(path: str | Path, *, channel=1, frames=None, normalized=False) -> np.ndarray:
    """Read saved v7.3 res/datOrgN in HWLT order, without another crop.

    Stored intensities are appropriate for the existing CFU script. For
    detection, normalized=True applies the saved global min/max in single
    precision. This cannot recover pre-save floating-point quantization.
    Saved data may already include registration/bleach correction; do not
    treat this result container as a raw MAT input to burst.prep1.
    """
    if isinstance(channel, bool) or channel not in (1, 2):
        raise ValueError("channel must be 1 or 2")
    frames = slice(None) if frames is None else frames
    if not isinstance(frames, slice) or (frames.step is not None and frames.step <= 0):
        raise ValueError("frames must be a slice with a positive step")
    with h5py.File(path, "r") as root:
        dataset = root[f"res/datOrg{channel}"]
        if dataset.attrs.get("MATLAB_empty", 0) or dataset.ndim != 4:
            raise ValueError("Expected a nonempty four-dimensional saved movie")
        expected = tuple(root["res/opts/sz"][()].ravel().astype(int))
        if tuple(reversed(dataset.shape)) != expected:
            raise ValueError("Saved movie shape differs from opts.sz")
        movie = dataset[frames, :, :, :].transpose(3, 2, 1, 0)
        if normalized:
            if movie.dtype.kind not in "ui":
                raise TypeError("Float saved movies require an explicit intensity convention")
            low = np.float32(root[f"res/opts/minValueDat{channel}"][()].item())
            high = np.float32(root[f"res/opts/maxValueDat{channel}"][()].item())
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                raise ValueError("Invalid saved intensity range")
            movie = (movie.astype(np.float32) - low) / (high - low)
        return movie


def read_working_mask(path, shape_hwz, crop_gap=0):
    """Batch mask union in working coordinates; independent checks in step 9."""
    path = Path(path)
    shape_hwz = tuple(shape_hwz)
    if path.suffix.lower() == ".mat":
        mask = read_mat(path,["bd0"])["bd0"]
        if mask.dtype == object:
            result = np.zeros(shape_hwz,bool,order="F")
            for region in mask.ravel(order="F"):
                pixels = np.asarray(region.ravel(order="F")[1]).ravel()
                from .compat import from_matlab_indices
                result.ravel(order="F")[from_matlab_indices(pixels,result.size)] = True
            return result
        mask = np.asarray(mask) != 0
    elif path.suffix.lower() in (".tif",".tiff"):
        with tifffile.TiffFile(path) as image:
            mask = image.pages[0].asarray()
        if mask.ndim == 3:
            mask = np.max(mask,axis=2)
        mask = mask != 0
    else:
        raise ValueError("Mask must be bd0 MAT or binary TIFF")
    if mask.shape == shape_hwz or (shape_hwz[2] == 1 and mask.shape == shape_hwz[:2]):
        return mask.reshape(shape_hwz,order="F")
    gap = int(crop_gap)
    if shape_hwz[2] == 1 and mask.shape == tuple(s+2*gap for s in shape_hwz[:2]):
        return mask[gap:-gap,gap:-gap,None] if gap else mask[:,:,None]
    raise ValueError("Mask dimensions do not match working coordinates")


def _v5(value):
    if issparse(value):
        raise TypeError("Sparse MAT arrays require an explicit export")
    if type(value).__name__ in ("MatlabOpaque", "MatlabObject", "MatlabFunction"):
        raise TypeError("MATLAB objects require an explicit numeric/struct export")
    if isinstance(value, np.ndarray) and value.dtype.names is not None:
        fields = value.dtype.names
        result = np.empty(value.shape, dtype=object).view(MatStructArray)
        result.mat_fields = fields
        for index in np.ndindex(value.shape):
            result[index] = {name: _v5(value[name][index]) for name in fields}
        return result[0, 0] if value.shape == (1, 1) else result
    if hasattr(value, "_fieldnames"):
        return {name: _v5(getattr(value, name)) for name in value._fieldnames}
    if isinstance(value, np.ndarray) and value.dtype == object:
        # Scalar MATLAB structs become dictionaries; numeric/cell axes remain.
        if value.size == 1 and hasattr(value.flat[0], "_fieldnames"):
            return _v5(value.flat[0])
        result = np.empty(value.shape, dtype=object)
        for index in np.ndindex(value.shape):
            result[index] = _v5(value[index])
        return result
    return value


def _matlab_class(node):
    raw = node.attrs.get("MATLAB_class", b"")
    return raw.decode() if isinstance(raw, bytes) else str(raw)


def _v73(node, root):
    cls = _matlab_class(node)
    if cls in ("opaque", "object", "function_handle") or "MATLAB_object_decode" in node.attrs:
        raise TypeError(f"Unsupported MATLAB object at {node.name}; export numeric fields first")
    if isinstance(node, h5py.Group):
        if cls not in ("struct", "") or "MATLAB_sparse" in node.attrs:
            raise TypeError(f"Unsupported MATLAB group {cls!r} at {node.name}")
        # Struct arrays store one reference array per field.
        fields = {key: _v73(child, root) for key, child in node.items() if key != "#refs#"}
        # Struct-array fields are untyped reference datasets. A scalar struct
        # may also contain typed cell datasets with different shapes; those
        # are values, not references describing the struct-array dimensions.
        ref_fields = [child for child in node.values() if isinstance(child, h5py.Dataset)
                      and h5py.check_dtype(ref=child.dtype) and not _matlab_class(child)]
        if cls == "struct" and ref_fields:
            shape = tuple(reversed(ref_fields[0].shape))
            if any(tuple(reversed(child.shape)) != shape for child in ref_fields):
                raise ValueError(f"Inconsistent struct dimensions at {node.name}")
            if shape == (1, 1):
                return {key: value[0, 0] for key, value in fields.items()}
            result = np.empty(shape, dtype=object).view(MatStructArray)
            result.mat_fields = tuple(fields)
            for index in np.ndindex(shape):
                result[index] = {key: value[index] for key, value in fields.items()}
            return result
        return fields
    if node.attrs.get("MATLAB_empty", 0):
        if cls == "char":
            return np.array([], dtype="U1")
        types = {"double": "float64", "single": "float32", "logical": "bool"}
        if cls == "cell":
            return np.empty(tuple(int(v) for v in node[()]), dtype=object)
        return np.empty(tuple(int(v) for v in node[()]), dtype=types.get(cls, cls))
    data = node[()]
    if h5py.check_dtype(ref=node.dtype):
        refs = data.T
        result = np.empty(refs.shape, dtype=object)
        for index in np.ndindex(refs.shape):
            if not refs[index]:
                raise ValueError(f"Null MATLAB reference at {node.name}")
            result[index] = _v73(root[refs[index]], root)
        return result
    data = data.transpose(tuple(reversed(range(data.ndim))))
    if cls == "logical":
        return data.astype(bool)
    if cls == "char":
        if data.ndim != 2:
            raise TypeError("Only two-dimensional MATLAB char arrays are supported")
        return np.array([row.astype("<u2").tobytes().decode("utf-16-le") for row in data])
    if data.dtype.names:
        raise TypeError(f"Compound/complex MATLAB arrays require explicit support: {node.name}")
    return data


def read_mat(path: str | Path, fields: list[str] | None = None) -> dict:
    """Read numeric/logical/char/struct/cell data from MAT v7 or v7.3.

    Optional slash paths select only needed datasets in large v7.3 files.
    MATLAB containers.Map and arbitrary objects are deliberately unsupported.
    For lossless v7 exchange, materialize MATLAB uninitialized cells before
    saving (python/matlab/portable_scientific.m). SciPy otherwise exposes their
    null-cell placeholder as 1x0 although MATLAB interprets it as double 0x0.
    """
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as root:
            keys = fields if fields is not None else [k for k in root if not k.startswith("#")]
            return {key: _v73(root[key], root) for key in keys}
    variables = list({key.split("/")[0] for key in fields}) if fields is not None else None
    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        try:
            raw = loadmat(path, struct_as_record=True, squeeze_me=False,
                          mat_dtype=True, variable_names=variables)
        except np.exceptions.ComplexWarning as error:
            raise TypeError("Complex MAT arrays require an explicit real/imaginary export") from error
    loaded = {key: _v5(value) for key, value in raw.items() if not key.startswith("__")}
    if fields is None:
        return loaded
    selected = {}
    for key in fields:
        value = loaded
        for part in key.split("/"):
            value = value[part]
        selected[key] = value
    return selected


def write_mat(path: str | Path, values: dict):
    """Write v7-compatible exchange files (not a full AQuA2 GUI project)."""
    def prepare(value):
        if isinstance(value, dict):
            return {key: prepare(child) for key, child in value.items()}
        if isinstance(value, np.ndarray) and value.dtype == object:
            if isinstance(value, MatStructArray):
                keys = list(value.mat_fields if value.mat_fields or not value.size else value.flat[0])
                if any(set(child) != set(keys) for child in value.flat):
                    raise ValueError("Struct array elements must share field names")
                result = np.empty(value.shape, dtype=[(key, object) for key in keys])
                for index in np.ndindex(value.shape):
                    for key in keys:
                        result[key][index] = prepare(value[index][key])
                return result
            result = np.empty(value.shape, dtype=object)
            for index in np.ndindex(value.shape):
                result[index] = prepare(value[index])
            return result
        return value
    savemat(path, prepare(values), do_compression=True, oned_as="row", long_field_names=True)
