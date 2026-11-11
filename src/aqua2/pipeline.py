"""Headless stage 3 orchestration; later segmentation remains separate."""
from pathlib import Path

import numpy as np

from .act import ac_detect
from .bleach import bleach_correct, bleach_correct2
from .io import read_mat, read_tiff_hwlt, read_saved_dat_org, read_working_mask
from .pre import baseline_remove_and_noise_estimation,median_correct
from .reg import reg_cross_correlation


def scalar_options(options):
    return {key:(value.item() if isinstance(value,np.ndarray) and value.size == 1 else value)
            for key,value in options.items()}


def _read_raw(source, variable=None):
    if isinstance(source,np.ndarray):
        array = source
        depth = source.dtype.itemsize*8
    else:
        path = Path(source)
        if path.suffix.lower() in (".tif",".tiff"):
            # Honor explicit Z+time volumes; plain ZYX stacks follow MATLAB's
            # page-as-time default in read_tiff_hwlt. RGB overlays are rejected.
            import tifffile
            with tifffile.TiffFile(path) as image:
                series = image.series[0]
                if len(image.series) == 1 and set(series.axes) == set("TZYX") and len(series.axes) == 4:
                    array = series.asarray().transpose(tuple(series.axes.index(axis) for axis in "YXZT"))
                else:
                    array = read_tiff_hwlt(path)
            depth = array.dtype.itemsize*8
        elif path.suffix.lower() == ".mat":
            values = read_mat(path,[variable] if variable else None)
            array = values[variable] if variable else next(iter(values.values()))
            depth = -1
            if isinstance(array,dict):
                raise ValueError("A saved res MAT must use saved_result=True; raw MAT requires a numeric variable")
        elif path.suffix.lower() == ".avi":
            import imageio.v3 as iio
            frames = []
            for frame in iio.imiter(path,plugin="FFMPEG"):
                frame = np.asarray(frame,dtype=np.float32)
                if frame.ndim == 3:
                    frame = np.float32(.2989)*frame[:,:,0]+np.float32(.5870)*frame[:,:,1]+np.float32(.1140)*frame[:,:,2]
                frames.append(frame)
            if not frames:
                raise ValueError("Empty AVI")
            array = np.stack(frames,axis=2)[:,:,None,:]
            depth = 8
        else:
            raise ValueError(f"Unsupported input: {path.suffix}")
    if array.ndim == 2:
        array = array[:,:,None,None]
    elif array.ndim == 3:
        array = array[:,:,None,:]
    if array.ndim != 4:
        raise ValueError("Raw input must have HWT or HWLT axes")
    return np.array(array,dtype=np.float32,order="F"),depth


def prepare_movie(source, options, source2=None, *, variable=None,auto_channels=False):
    opts = scalar_options(options)
    mat_input=not isinstance(source,np.ndarray) and Path(source).suffix.lower()=='.mat'
    if auto_channels and source2 is None and isinstance(source,(str,Path)) and Path(source).suffix.lower() in ('.tif','.tiff','.mat'):
        from .input_formats import read_tiff_channels,read_mat_channels
        channels=read_mat_channels(source,variable) if mat_input else read_tiff_channels(source)
        source=channels[0];source2=channels[1] if len(channels)>1 else None
    one,depth = _read_raw(source,variable)
    if mat_input:depth=-1  # Preserve the original raw-MAT storage boundary.
    two = None if source2 is None else _read_raw(source2,variable)[0]
    gap = int(opts["regMaskGap"])
    if gap < 0 or gap*2 >= min(one.shape[:2]):
        raise ValueError("Invalid crop margin")
    crop = (slice(gap,-gap if gap else None),)*2+(slice(None),slice(None))
    one = one[crop]
    low,high = np.nanmin(one),np.nanmax(one)
    with np.errstate(divide="ignore",invalid="ignore"):
        one = np.maximum(one-low,0)/(high-low)
    opts.update({"minValueDat1":low,"maxValueDat1":high,"BitDepth":depth,"singleChannel":two is None})
    if two is not None:
        two = two[crop]
        if two.shape != one.shape:
            raise ValueError("Channel dimensions differ")
        low = np.nanmin(two)
        two = np.maximum(two-low,0)
        high = np.nanmax(two)
        with np.errstate(divide="ignore",invalid="ignore"):
            two = two/high
        # Preserve prep1's historical channel-2 maximum-after-subtraction convention.
        opts.update({"minValueDat2":low,"maxValueDat2":high})
    opts["sz"] = one.shape
    return np.array(one,order="F"),None if two is None else np.array(two,order="F"),opts


def preprocess(source, options=None, source2=None, *, variable=None, spatial_mask=None,
               saved_result=False, frames=None, prepared=False, gui_median=False, progress=None,detect_activity=True,auto_channels=False):
    if progress:progress("Loading and normalizing image")
    if saved_result:
        if source2 is not None:
            raise ValueError("Saved-result channels are taken from the same result container")
        opts = scalar_options(read_mat(source,["res/opts"])["res/opts"])
        if options:
            opts.update(scalar_options(options))
        one = read_saved_dat_org(source,frames=frames,normalized=True)
        two = None if bool(opts["singleChannel"]) else read_saved_dat_org(source,channel=2,frames=frames,normalized=True)
        opts["inputMode"] = "saved_datOrg_no_recrop_or_repeat_correction"
    elif prepared:
        opts = scalar_options(options)
        one = np.array(source,dtype=np.float32,order="F")
        two = None if source2 is None else np.array(source2,dtype=np.float32,order="F")
        opts["singleChannel"] = two is None
    else:
        if options is None:
            raise ValueError("Raw preprocessing requires explicit options")
        one,two,opts = prepare_movie(source,options,source2,variable=variable,auto_channels=auto_channels)
    if progress:progress("Input ready: " + " × ".join(str(n) for n in one.shape) + " (H×W×Z×T)")
    if not saved_result:
        registration = int(opts.get("registrateCorrect",1))
        if progress and registration != 1:progress("Registration")
        if registration == 2 or (registration == 3 and two is None):
            one,two = reg_cross_correlation(one,two)
        elif registration == 3:
            two,one = reg_cross_correlation(two,one)
        elif registration != 1:
            raise ValueError("Registration mode must be 1, 2 or 3")
        bleaching = int(opts.get("bleachCorrect",1))
        if progress and bleaching != 1:progress("Bleaching Correction")
        if bleaching in (2,3):
            correction = bleach_correct if bleaching == 2 else lambda data:bleach_correct2(data)[0]
            one = correction(one)
            if two is not None:
                two = correction(two)
        elif bleaching != 1:
            raise ValueError("Bleaching mode must be 1, 2 or 3")
        if gui_median and opts.get("medSmo",0) > 0:
            one,opts["minValueDat1"],opts["maxValueDat1"] = median_correct(
                one,opts["medSmo"],opts["minValueDat1"],opts["maxValueDat1"])
            if two is not None:
                two,opts["minValueDat2"],opts["maxValueDat2"] = median_correct(
                    two,opts["medSmo"],opts["minValueDat2"],opts["maxValueDat2"])
    if frames is not None and not saved_result:
        one = one[:,:,:,frames]
        if two is not None:
            two = two[:,:,:,frames]
    opts["sz"] = one.shape
    if isinstance(spatial_mask,(str,Path)):
        spatial_mask = read_working_mask(spatial_mask,one.shape[:3],opts.get("regMaskGap",0))
    allowed_mask_shapes = (one.shape[:3],one.shape[:2]) if one.shape[2] == 1 else (one.shape[:3],)
    if spatial_mask is not None and np.asarray(spatial_mask).shape not in allowed_mask_shapes:
        raise ValueError("Spatial mask must already match working/cropped coordinates")
    def channel_progress(channel):
        return (lambda detail,done=None,total=None:progress(f"Channel {channel} · {detail}",done,total)) if progress else None
    d1,opts,intermediate1 = baseline_remove_and_noise_estimation(one,opts,spatial_mask,1,progress=channel_progress(1))
    regions1 = ac_detect(d1,opts,spatial_mask,1,progress=channel_progress(1)) if detect_activity else []
    result = {"datOrg1":one,"dF1":d1,"activeRegions1":regions1,"intermediate1":intermediate1}
    if two is not None:
        d2,opts,intermediate2 = baseline_remove_and_noise_estimation(two,opts,spatial_mask,2,progress=channel_progress(2))
        result.update({"datOrg2":two,"dF2":d2,"activeRegions2":ac_detect(d2,opts,spatial_mask,2,progress=channel_progress(2)) if detect_activity else [],"intermediate2":intermediate2})
    result["opts"] = opts
    return result


def stage3_mat_payload(result):
    """Keep public indices zero based; convert only at the MATLAB boundary."""
    payload = {key:value for key,value in result.items() if not key.startswith("activeRegions")}
    payload["opts"] = dict(result["opts"])
    payload["opts"]["sz"] = np.asarray(result["opts"]["sz"],dtype=np.float64).reshape(1,4)
    payload["opts"]["cut"] = float(result["opts"]["cut"])
    for channel in (1,2):
        key = f"activeRegions{channel}"
        if key in result:
            regions = result[key]
            cells = np.empty((1,len(regions)),dtype=object)
            for i,indices in enumerate(regions):
                cells[0,i] = (indices+1).astype(np.float64).reshape(-1,1)
            payload[key] = cells
    return payload
