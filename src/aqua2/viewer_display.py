"""Shared, non-destructive display controls and ImageJ/Fiji Auto semantics."""
from dataclasses import dataclass,asdict
from functools import lru_cache
import numpy as np


@dataclass(frozen=True)
class DisplaySettings:
    minimum:float=0.
    maximum:float=1.
    brightness:float=.5
    contrast:float=.5
    transparency:float=0.
    overlay_brightness:float=.5

    def __post_init__(self):
        if any(not np.isfinite(v) or not 0<=v<=1 for v in asdict(self).values()):raise ValueError('Display values must be 0–1')
        if self.maximum<=self.minimum:raise ValueError('Max must exceed Min.')

    def as_dict(self):return asdict(self)


def _map(values,limits,settings):
    base,top=limits;span=max(float(top-base),1e-12)
    low=base+span*settings.minimum;high=base+span*settings.maximum
    scaled=np.clip((values-low)/(high-low),0,1)
    scaled=(scaled-.5)*2**(4*(settings.contrast-.5))+.5
    scaled*=2**(4*(settings.brightness-.5))
    return np.floor(np.clip(np.nan_to_num(scaled),0,1)*255+.5).astype(np.uint8)


@lru_cache(maxsize=16)
def _lut(dtype,low,high,settings):
    return _map(np.arange(np.iinfo(np.dtype(dtype)).max+1,dtype=float),(low,high),settings)


def display_gray(array,limits,settings,*,volume=False):
    if array.dtype in (np.dtype('uint8'),np.dtype('uint16')):result=_lut(array.dtype.str,float(limits[0]),float(limits[1]),settings)[array]
    else:result=_map(array.astype(float),limits,settings)
    if volume and settings.transparency:result=np.floor(result.astype(float)*(1-settings.transparency)+.5).astype(np.uint8)
    return result


def fiji_auto_range(array,divisor=5000):
    """ImageJ ContrastAdjuster Auto: 256 bins, dominant peaks ignored.

    Byte histograms use [0,256); shorts use (max-min+1)/256 bins;
    floating images use (max-min)/256. Nonfinite pixels are excluded.
    Returns None for no significant bins, allowing the viewer to Reset.
    """
    finite=np.asarray(array).ravel();finite=finite[np.isfinite(finite)]
    if not len(finite):return None
    low,high=float(finite.min()),float(finite.max())
    if low==high:return None
    if array.dtype==np.uint8:origin=0.;width=1.
    else:origin=low;width=(high-low+(1 if array.dtype.kind in 'ui' else 0))/256
    indices=np.clip(((finite.astype(float)-origin)/width).astype(np.int64),0,255)
    histogram=np.bincount(indices,minlength=256);n=len(finite)
    significant=np.flatnonzero((histogram<=n//10)&(histogram>n//max(1,int(divisor))))
    if not len(significant):return None
    minimum=origin+int(significant[0])*width;maximum=origin+int(significant[-1])*width
    return (low,high) if minimum==maximum else (minimum,maximum)
