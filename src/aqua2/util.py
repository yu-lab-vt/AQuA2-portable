import numpy as np


def normalize01(data):
    result = np.asarray(data,dtype=np.float64).copy()
    finite = np.isfinite(result)
    if not finite.any():
        return np.zeros_like(result)
    low,high = result[finite].min(),result[finite].max()
    if high > low:
        result = (result-low)/(high-low)
    else:
        result.fill(0)
    result[~finite] = 0
    return result


def impute_mov(curves):
    out = np.array(curves,copy=True)
    for t in range(1,out.shape[1]):
        selected = np.isnan(out[:,t]); out[selected,t] = out[selected,t-1]
    for t in range(out.shape[1]-2,-1,-1):
        selected = np.isnan(out[:,t]); out[selected,t] = out[selected,t+1]
    return out
