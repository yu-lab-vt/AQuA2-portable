"""Lossless sparse reconstruction corresponding to MATLAB overlay pix/val."""
from dataclasses import dataclass
import numpy as np

@dataclass
class Reconstruction:
    shape:tuple
    pixels:object
    values:object
    dtype:object=np.dtype('uint8')
    @property
    def size(self):return int(np.prod(self.shape))
    @property
    def ndim(self):return len(self.shape)
    def __array__(self,dtype=None,copy=None):
        result=np.zeros(self.shape,dtype=self.dtype,order='F');result.ravel(order='F')[np.asarray(self.pixels,np.int64)]=np.asarray(self.values)
        return result.astype(dtype,copy=False) if dtype is not None else result
    def __getitem__(self,key):return np.asarray(self)[key]
