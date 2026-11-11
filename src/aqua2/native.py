"""Checked C ABI for the original DTW/BILCO kernels; no MATLAB dependency."""
import ctypes
from functools import lru_cache
from pathlib import Path
import threading
import numpy as np

_lock=threading.RLock()
_pointer=np.ctypeslib.ndpointer(dtype=np.float64,flags="F_CONTIGUOUS")


@lru_cache(maxsize=2)
def _library(name):
    candidates=[p for p in Path(__file__).parent.glob("_"+name+"_core*") if p.suffix in (".pyd",".so",".dll")]
    if len(candidates)!=1:
        raise RuntimeError("Build the native kernels with pip install -e python --no-build-isolation --no-deps (C++ compiler required)")
    library=ctypes.CDLL(str(candidates[0]))
    library.aqua_run.argtypes=[ctypes.c_int]*4+[_pointer]*4+[ctypes.c_int]*2+[_pointer]*2
    library.aqua_run.restype=ctypes.c_int
    library.aqua_error.argtypes=[];library.aqua_error.restype=ctypes.c_char_p
    return library


def _matrix(value,name):
    array=np.asarray(value)
    if array.ndim!=2 or np.iscomplexobj(array) or not np.isfinite(array).all():
        raise ValueError(name+" must be a finite real matrix")
    return np.asfortranarray(array,dtype=np.float64)


def _run(name,n,t1,t2,pairs,ref,tst,initial,window=10000,absolute=False):
    # Upstream indices use signed int and allocate several dense scratch planes.
    if n*t1*t2>np.iinfo(np.int32).max//32:
        raise ValueError("Native graph exceeds the checked 32-bit workspace limit")
    output=np.empty((n if name=="bilco" else 1,t1-1),dtype=float,order="F")
    flow=np.zeros((1,1),dtype=float,order="F")
    with _lock:
        lib=_library(name)
        status=lib.aqua_run(n,t1,t2,len(pairs),ref,tst,pairs,initial,int(window),int(absolute),output,flow)
        if status:raise RuntimeError(lib.aqua_error().decode("utf-8",errors="replace"))
    return output,float(flow.item())


def dtw_edge(distance):
    """Return zero-based half-integer cuts; distinct from temporal se DTW."""
    distance=_matrix(distance,"distance")
    t2,t1=distance.shape
    if min(t1,t2)<1 or (distance<0).any() or (distance>np.finfo(np.float32).max/max(1,t1+t2)).any():raise ValueError("Distance must be nonempty, nonnegative and fit native float accumulation")
    if t1==1:return np.empty(0,float)
    empty=np.empty((0,3),order="F");dummy=np.empty((0,0),order="F")
    return _run("dtw",1,t1,t2,empty,distance,dummy,dummy)[0].ravel()


def bilco(reference,test,pairs,smoothness,initial=None,*,window=10000,metric="squared"):
    """Joint min-cut; pairs are zero-based Nx2, initial uses source's 1-based cut ABI."""
    reference=_matrix(reference,"reference");test=_matrix(test,"test")
    n,t2=test.shape;t1=reference.shape[1]
    if n<1 or min(t1,t2)<2:raise ValueError("BILCO needs >=1 curves and lengths >=2")
    if len(reference)==1 and n!=1:reference=np.asfortranarray(np.repeat(reference,n,axis=0))
    if len(reference)!=n:raise ValueError("Curve counts do not match")
    pairs=np.asarray(pairs)
    if pairs.size==0:pairs=np.empty((0,2))
    if pairs.ndim!=2 or pairs.shape[1]!=2 or not np.isfinite(pairs).all() or not np.equal(pairs,np.floor(pairs)).all() or np.any((pairs<0)|(pairs>=n)):
        raise ValueError("Neighbor pairs must contain valid zero-based integer curve indices")
    if np.any(pairs[:,0]==pairs[:,1]):raise ValueError("Self edges are invalid")
    if not np.isfinite(smoothness) or smoothness<0:raise ValueError("Smoothness must be finite and nonnegative")
    if metric not in ("squared","absolute"):raise ValueError("Unknown BILCO metric")
    bound=np.sqrt(np.finfo(np.float32).max/(4*(t1+t2))) if metric=="squared" else np.finfo(np.float32).max/(4*(t1+t2))
    if np.max(abs(reference))>bound or np.max(abs(test))>bound or smoothness>np.finfo(np.float32).max/(4*(t1+t2)):
        raise ValueError("BILCO costs exceed native float accumulation range")
    if not np.isfinite(window) or int(window)!=window or not 0<=window<=np.iinfo(np.int32).max//4:raise ValueError("Invalid BILCO window")
    initial=np.ones((n,t1-1),order="F") if initial is None else _matrix(initial,"initial cut")
    if initial.shape!=(n,t1-1) or np.any(initial<1) or np.any(initial>t2) or np.any(np.diff(initial,axis=1)<0):raise ValueError("Invalid monotone 1-based initial cuts")
    graph=np.asfortranarray(np.column_stack((pairs+1,np.full(len(pairs),smoothness))))
    return _run("bilco",n,t1,t2,graph,reference,test,initial,window,metric=="absolute")


def midpoint_to_path(midpoints,t1,t2):
    """Convert zero-based half-integer cuts to a zero-based [ref,test] path."""
    points=[];x=0
    for y,cut in enumerate(midpoints):
        points.extend((y,k) for k in range(x,int(np.floor(cut))+1))
        x=int(np.ceil(cut))
    points.extend((t1-1,k) for k in range(x,t2))
    return np.asarray(points,dtype=np.int64).reshape(-1,2)
