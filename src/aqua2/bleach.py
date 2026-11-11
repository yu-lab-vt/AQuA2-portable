"""Stretched-exponential bleaching correction; validation deferred to step 9."""
import numpy as np
from scipy.optimize import least_squares


def _fit_trend(trend):
    trend = np.asarray(trend,dtype=np.float64)
    trend = np.minimum.accumulate(trend/trend[0])
    x = np.arange(1,len(trend)+1,dtype=np.float64)
    minimum = trend.min()
    if not np.all(np.isfinite(trend)) or minimum <= 0:
        raise ValueError("Invalid trend for exponential bleaching fit")
    def evaluate(p):
        a,b,c,d = p
        return a*np.exp(-b*x**d)+c
    fit = least_squares(lambda p:evaluate(p)-trend,
        [trend.max()-minimum,0,minimum,1],bounds=([0,0,0,0],[np.inf,np.inf,minimum,1]))
    if not fit.success:
        raise RuntimeError(f"Bleaching fit did not converge: {fit.message}")
    return evaluate(fit.x)


def bleach_correct(data):
    curves = data.reshape((-1,data.shape[3]),order="F")
    trend = np.mean(curves,axis=0).astype(np.float64)
    fit = _fit_trend(trend).astype(data.dtype)
    return np.array(data/fit[None,None,None,:],order="F")


def bleach_correct2(data):
    curves = data.reshape((-1,data.shape[3]),order="F").copy()
    intensity = np.zeros(curves.shape[0])
    for t in range(curves.shape[1]):
        intensity = (intensity.astype(data.dtype)+curves[:,t]/np.mean(curves[:,t])).astype(data.dtype)
    intensity = (intensity-intensity.min())/(intensity.max()-intensity.min())
    upper,lower = 1.01,1.0
    clusters, fits = [], []
    previous = None
    while upper > 0:
        while True:
            lower -= .01
            indices = np.flatnonzero((intensity >= lower) & (intensity < upper))
            if len(indices) >= 2000 or lower < 0:
                break
        if lower >= 0:
            previous = _fit_trend(np.mean(curves[indices,:],axis=0))
        if previous is None:
            raise ValueError("Original grouped bleaching has no fitted cluster below 2000 pixels")
        clusters.append(indices); fits.append(previous.copy())
        curves[indices,:] /= previous.astype(curves.dtype)
        upper = lower
    return np.array(curves.reshape(data.shape,order="F"),order="F"),clusters,fits
