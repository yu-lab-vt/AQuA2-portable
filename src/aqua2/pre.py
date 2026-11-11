"""Baseline and noise estimation translated from src/+pre (HWLT arrays)."""
from functools import lru_cache
from pathlib import Path

import numpy as np
from numba import njit
from scipy import ndimage, special

_BASELINE_BLOCK_SAMPLES = 1_000_000

def gaussian_xy(data, sigma):
    if sigma <= 0:
        return np.array(data, order="F", copy=True)
    radius = int(np.ceil(2 * sigma))
    result = np.empty_like(data, order="F")
    ndimage.gaussian_filter(data, sigma=(sigma, sigma) + (0,) * (data.ndim - 2),
        radius=(radius, radius) + (0,) * (data.ndim - 2), mode="nearest", output=result)
    return result


def median_correct(data, radius, minimum, maximum):
    """Extracted GUI median step; batch default omits it. Validation: step 9."""
    if radius < 0 or int(radius) != radius:
        raise ValueError("Median radius must be a nonnegative integer")
    size = int(radius)*2+1
    filtered = ndimage.median_filter(data,size=(size,size,1,1),mode="constant",cval=0)
    low,high = np.nanmin(filtered),np.nanmax(filtered)
    with np.errstate(invalid="ignore",divide="ignore"):
        filtered = (filtered-low)/(high-low)
    return filtered,minimum+(maximum-minimum)*low,minimum+(maximum-minimum)*high


def moving_mean(data, window, axis=-1):
    window = int(window)
    if window < 1:
        raise ValueError("Moving window must be positive")
    a = np.moveaxis(data, axis, -1)
    finite = ~np.isnan(a)
    positions = np.arange(a.shape[-1])
    lo = np.maximum(0, positions - window // 2)
    hi = np.minimum(a.shape[-1], positions + (window - 1) // 2 + 1)
    if finite.all():
        sums = np.concatenate((np.zeros(a.shape[:-1] + (1,)), np.cumsum(np.ascontiguousarray(a), axis=-1, dtype=np.float64)), axis=-1)
        denominator = hi-lo
    else:
        sums = np.concatenate((np.zeros(a.shape[:-1] + (1,)), np.cumsum(np.where(finite, a, 0), axis=-1, dtype=np.float64)), axis=-1)
        counts = np.concatenate((np.zeros(a.shape[:-1] + (1,), dtype=np.int64), np.cumsum(finite, axis=-1)), axis=-1)
        denominator = counts[..., hi] - counts[..., lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        result = (sums[..., hi] - sums[..., lo]) / denominator
    return np.array(np.moveaxis(result.astype(data.dtype), -1, axis), order="F")


@njit(cache=True)
def _baseline_fill(initial, positions, maximum):
    output = initial.copy()
    for pixel in range(initial.shape[0]):
        knots = np.unique(positions[pixel])
        valid = knots[~np.isnan(initial[pixel, knots])]
        if len(valid) == 0:
            output[pixel, :] = maximum
            continue
        output[pixel, :valid[0]+1] = initial[pixel, valid[0]]
        output[pixel, valid[-1]:] = initial[pixel, valid[-1]]
        for k in range(len(valid)-1):
            left, right = valid[k], valid[k+1]
            v0, v1 = initial[pixel, left], initial[pixel, right]
            slope = np.float32(np.float32(v1-v0) / np.float32(right-left))
            for t in range(left, right+1):
                output[pixel, t] = np.float32(v0 + np.float32(slope * np.float32(t-left)))
    return output


def baseline_linear_estimate(data, cut, mov_avg_win, *, progress=None):
    data = np.asarray(data)
    if data.ndim != 4 or data.dtype != np.float32:
        raise ValueError("Baseline input must be single precision HWLT")
    time = data.shape[3]
    cut = int(cut)
    step = int(np.floor(0.5 * cut + 0.5))
    if step < 1:
        raise ValueError("cut is too small for the MATLAB segment rule")
    curves = data.reshape((-1, time), order="F")
    count = max(1, int(np.ceil(time / step)) - 1)
    maximum = np.nanmax(data)
    result = np.empty(data.shape, dtype=data.dtype, order="F")
    output = result.reshape((-1, time), order="F")
    # Keep each complete time curve together, including segment overlap and
    # global fallback. Bound cumulative-sum/count temporaries to ~1M samples
    # instead of allocating float64/int64 movies for the entire recording.
    block_size = max(1, _BASELINE_BLOCK_SAMPLES // time)
    if progress:progress("Estimating baseline", 0, len(curves))
    for start in range(0, len(curves), block_size):
        stop = min(start + block_size, len(curves))
        source = curves[start:stop]
        initial = moving_mean(source, mov_avg_win, 1)
        invalid = np.isnan(source)
        initial[invalid] = np.nan
        positions = np.empty((len(source), count), dtype=np.int64)
        has_nan = np.isnan(initial).any()
        for k in range(count):
            lo, hi = k * step, min(time, k * step + cut + 1)
            values = initial[:, lo:hi]
            if has_nan:values = np.where(np.isnan(values), np.inf, values)
            positions[:, k] = lo + np.argmin(values, axis=1)
        filled = _baseline_fill(np.ascontiguousarray(initial), positions, maximum)
        filled[invalid] = maximum
        output[start:stop] = filled
        if progress:progress("Estimating baseline", stop, len(curves))
    return result


def truncated_kept_var(quantiles):
    quantiles = np.asarray(quantiles, dtype=np.float64)
    a = special.ndtri(quantiles)
    phi = np.exp(-a*a/2) / np.sqrt(2*np.pi)
    with np.errstate(invalid="ignore"):
        mean = a * quantiles + phi
        second = a*a*quantiles + 1 - quantiles + a*phi
        result = (second - mean*mean) * 2
    return np.where(quantiles == 0, 2.0, result)


@lru_cache(maxsize=8)
def _bias_table(path):
    from .io import read_mat
    return read_mat(path)


def obtain_bias(window, cut, table_path=None):
    if window > cut:
        return 0.0
    if table_path is None:
        table_path = Path(__file__).with_name("data") / "F0_biasMatrix.mat"
    table = _bias_table(str(table_path))
    windows, cuts = table["windowSizes"].ravel(), table["cuts"].ravel()
    ix = np.flatnonzero(window >= windows)
    iy = np.flatnonzero(cut >= cuts)
    if not len(ix) or not len(iy):
        raise ValueError("Requested window/cut precedes the MATLAB lookup table")
    i, j = ix[-1], iy[-1]
    i1 = i if i == len(windows)-1 or windows[i] == window else i+1
    j1 = j if j == len(cuts)-1 or cuts[j] == cut else j+1
    if j+1 == len(cuts):
        raise ValueError("MATLAB obtainBias accesses cuts(j+1) beyond the table at this cut")
    b0, b1 = np.nanmean(table["biasMatrix"][[i, i1], j]), np.nanmean(table["biasMatrix"][[i, i1], j1])
    return float(b1 if np.isnan(b0) else b0 + (b1-b0)/(cuts[j+1]-cuts[j])*(cut-cuts[j]))


def variance_bins(baseline, variance):
    selected = variance > 1e-8
    x0, y0 = baseline.ravel(order="F")[selected.ravel(order="F")], variance.ravel(order="F")[selected.ravel(order="F")]
    if not len(x0):
        return np.empty(0), np.empty(0)
    low, high = np.nanmin(x0), np.nanmax(x0)
    delta = np.maximum(np.float32(1e-5), np.float32((high-low)/np.float32(2000)))
    xs, ys = [], []
    for i in range(2000):
        left, right = np.float32(low + np.float32(i)*delta), np.float32(low + np.float32(i+1)*delta)
        mask = ((x0 >= left) if i == 0 else (x0 > left)) & (x0 <= right)
        if np.any(mask):
            xs.append(float(np.mean(x0[mask])))
            ys.append(float(np.mean(y0[mask])))
    return np.array(xs), np.array(ys)


@njit(cache=True)
def _fit_knots(x, y):
    n = len(x)
    cost = np.zeros((n, n))
    for j in range(n):
        for i in range(j):
            slope = (y[j]-y[i])/(x[j]-x[i])
            intercept = y[i]-slope*x[i]
            for k in range(i, j+1):
                cost[i, j] += abs(y[k] - (slope*x[k]+intercept))
    source, sink = int(np.ceil(n*.05)), max(1, int(np.floor(n*.99)))
    dist1, dist2, dist3 = np.full(n,np.inf), np.full(n,np.inf), np.full(n,np.inf)
    pre1, pre2, pre3 = np.zeros(n,np.int64), np.zeros(n,np.int64), np.zeros(n,np.int64)
    for j in range(n):
        for i in range(min(j,source)):
            horizontal = 0.0
            for k in range(i+1):
                horizontal += abs(y[k]-y[i])
            value = cost[i,j]+horizontal
            if value < dist1[j]:
                dist1[j], pre1[j] = value, i
    for j in range(n):
        for i in range(j):
            value = dist1[i] + cost[i,j]
            if value < dist2[j]:
                dist2[j], pre2[j] = value, i
    for j in range(sink-1,n):
        for i in range(j):
            horizontal = 0.0
            for k in range(j,n):
                horizontal += abs(y[k]-y[j])
            value = dist2[i]+cost[i,j]+horizontal
            if value < dist3[j]:
                dist3[j], pre3[j] = value, i
    last = np.argmin(dist3)
    if not np.isfinite(dist3[last]):
        raise ValueError("Insufficient bins for the original three-segment fit")
    middle = pre3[last]
    first = pre2[middle]
    return np.array([pre1[first],first,middle,last])


def fit_f0_var(baseline, variance, valid_map=None):
    variance = np.array(variance, copy=True, order="F")
    if valid_map is not None:
        variance[~valid_map] = np.nan
    x, y = variance_bins(baseline, variance)
    if len(x) == 0:
        return np.full(baseline.shape, 1e-8, dtype=np.float64)
    if len(x) == 1:
        return np.full(baseline.shape, y[0], dtype=np.float64)
    knots = _fit_knots(x,y)
    xx, yy = x[knots], y[knots]
    out = variance
    out[baseline <= xx[0]] = yy[0]
    for segment in range(3):
        a = (yy[segment+1]-yy[segment])/(xx[segment+1]-xx[segment])
        b = yy[segment]-a*xx[segment]
        select = (baseline >= xx[segment]) & ((baseline <= xx[segment+1]) if segment == 2 else (baseline < xx[segment+1]))
        out[select] = np.array(a,dtype=baseline.dtype)*baseline[select]+np.array(b,dtype=baseline.dtype)
    out[baseline >= xx[3]] = yy[3]
    out[np.isnan(out)] = yy[0]
    return out


def correct_boundary_std(sigma, shape):
    radius = int(np.ceil(2*sigma))
    size = radius*2+1
    kernel = np.zeros((size,size,size)); kernel[radius,radius,radius] = 1
    kernel = gaussian_xy(kernel,sigma)
    denominator = np.sum(kernel**2)
    correction = np.zeros((radius+1,)*3)
    for i in range(radius+1):
        for j in range(radius+1):
            for k in range(radius+1):
                end = (radius+i,radius+j,radius+k)
                folded = kernel.copy()
                folded[end[0],:,:] = np.sum(folded[end[0]:,:,:],axis=0)
                folded[:,end[1],:] = np.sum(folded[:,end[1]:,:],axis=1)
                folded[:,:,end[2]] = np.sum(folded[:,:,end[2]:],axis=2)
                correction[i,j,k] = np.sqrt(np.sum(folded[:end[0]+1,:end[1]+1,:end[2]+1]**2)/denominator)
    indices = [np.minimum(np.minimum(np.arange(s),s-1-np.arange(s)),radius) for s in shape[:3]]
    out = correction[np.ix_(*indices)]
    return out/out.min()


def noise_estimation(baseline_projection, original, smoothed, sigma, mask, correct_noise=True):
    temp_map = np.nanmean((original[:,:,:,1:]-original[:,:,:,:-1])**2,axis=3)
    temporal_variance = temp_map/np.float32(2)
    if correct_noise:
        counts = np.sum(original == 0,axis=3,dtype=np.float64)
        totals = np.sum(~np.isnan(original),axis=3,dtype=np.float64)
        correction = truncated_kept_var(counts/totals)
        raw_variance = temp_map/correction.astype(temp_map.dtype)
    else:
        correction = np.full(temp_map.shape,2.0)
        raw_variance = temporal_variance.copy()
    raw_variance[~mask] = np.nan
    std_raw = np.sqrt(fit_f0_var(baseline_projection,raw_variance))
    if sigma == 0:
        return std_raw,std_raw,temporal_variance,correction
    radius = int(np.ceil(2*sigma))
    kernel = np.zeros((2*radius+1,)*3); kernel[radius,radius,radius] = 1
    kernel = gaussian_xy(kernel,sigma)**2
    variance = np.nanmean((smoothed[:,:,:,1:]-smoothed[:,:,:,:-1])**2,axis=3)/np.float32(2)
    if correct_noise:
        variance = variance * ndimage.correlate(temp_map/correction.astype(temp_map.dtype),kernel,mode="constant",cval=0) / ndimage.correlate(temporal_variance,kernel,mode="constant",cval=0)
    variance[~mask] = np.nan
    boundary = correct_boundary_std(sigma,original.shape)
    std_smooth = np.sqrt(fit_f0_var(baseline_projection,variance,boundary<1+1e-5))
    std_smooth = std_smooth*boundary.astype(std_smooth.dtype)
    return std_raw,std_smooth,temporal_variance,correction


def baseline_remove_and_noise_estimation(original, options, spatial_mask=None, channel=1, *, progress=None):
    original = np.asarray(original,dtype=np.float32,order="F")
    if original.ndim != 4 or original.shape[3] < 2:
        raise ValueError("Expected HWLT input with at least two frames")
    opts = dict(options)
    mask = np.ones(original.shape[:3],bool) if spatial_mask is None else np.asarray(spatial_mask,bool).reshape(original.shape[:3],order="F")
    sigma = float(opts["smoXY"])
    if progress:progress("Spatial Smoothing")
    smoothed = gaussian_xy(original,sigma)
    opts["cut"] = min(int(opts["cut"]),original.shape[3])
    baseline = baseline_linear_estimate(smoothed,opts["cut"],int(opts["movAvgWin"]),progress=progress)
    if progress:progress("Baseline projection")
    projection = np.mean(baseline,axis=3)
    if progress:progress("Estimating noise and variance")
    std_raw,std_smooth,temporal_variance,correction = noise_estimation(projection,original,smoothed,sigma,mask)
    if progress:progress("Baseline correction and dF")
    bias = obtain_bias(int(opts["movAvgWin"]),opts["cut"])
    # MATLAB mixed single/double arithmetic preserves the single data operand.
    adjustment = (np.array(bias,dtype=std_smooth.dtype)*std_smooth).astype(np.float32)
    baseline -= adjustment[:,:,:,None]
    with np.errstate(invalid="ignore",divide="ignore"):
        delta = smoothed-baseline
        delta /= std_smooth.astype(np.float32)[:,:,:,None]
    opts.update({"tempVarOrg":temporal_variance,"correctPars":correction,
        f"stdMap{channel}":std_smooth,f"stdMapOrg{channel}":std_raw,
        f"tempVarOrg{channel}":temporal_variance,f"correctPars{channel}":correction,
        f"maxdF{channel}":np.fmin(np.float32(100),np.nanmax(delta))})
    return delta,opts,{"smoothed":smoothed,"F0":baseline,"F0Pro":projection}


def avg_dat_noise_est(curve, spatial_indices, options):
    indices = np.asarray(spatial_indices,dtype=np.int64)
    var1 = options["tempVarOrg"].ravel(order="F")[indices]
    pars = options["correctPars"].ravel(order="F")[indices].astype(var1.dtype)
    dtype = curve.dtype
    weights = np.ones(var1.size,dtype=var1.dtype)
    before = np.mean(np.diff(curve)**2)/np.array(2,dtype=dtype)
    numerator = np.dot(weights,var1*np.array(2,dtype=var1.dtype)/pars)
    return np.sqrt(before*numerator/np.dot(weights,var1))
