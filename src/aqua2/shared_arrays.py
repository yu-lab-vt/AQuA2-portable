"""Reference-counted OS shared arrays for local, cancelable step workers.

Only descriptors cross the Pipe for arrays already shared. New results are
copied once into shared memory, never into a project/checkpoint file.
"""
from dataclasses import dataclass
from multiprocessing import shared_memory
import weakref
import numpy as np

_mappings=weakref.WeakValueDictionary()


class _Mapping:
    def __init__(self,shm,owner=False):
        self.shm=shm;self.owner=owner
        self.address=np.ndarray((1,),np.uint8,buffer=shm.buf).ctypes.data
        _mappings[shm.name]=self
    def __del__(self):
        try:
            if self.owner:self.shm.unlink()
            self.shm.close()
        except (OSError,BufferError):pass


class _ArrayOwner(np.ndarray):
    def __array_finalize__(self,source):self.mapping=getattr(source,'mapping',None)


@dataclass(frozen=True)
class SharedBuffer:
    name:str
    shape:tuple
    dtype:object
    order:str
    offset:int=0
    claim_owner:bool=False


def mapping_of(array):
    current=array
    while isinstance(current,np.ndarray):
        if isinstance(current,_ArrayOwner):return current.mapping
        current=current.base
    return None


def _array(mapping,descriptor):
    array=np.ndarray.__new__(_ArrayOwner,descriptor.shape,dtype=descriptor.dtype,
        buffer=mapping.shm.buf,offset=descriptor.offset,order=descriptor.order)
    array.mapping=mapping
    # Return a normal ndarray. Its base retains the mapping through views and
    # np.asarray, without propagating a subclass into scientific arithmetic.
    return np.asarray(array)


def describe(array):
    if not (array.flags.c_contiguous or array.flags.f_contiguous):return None
    mapping=mapping_of(array)
    if mapping is None:return None
    offset=array.ctypes.data-mapping.address
    if offset<0 or offset+array.nbytes>mapping.shm.size:return None
    return SharedBuffer(mapping.shm.name,array.shape,array.dtype,
        'F' if array.flags.f_contiguous else 'C',offset)


def publish(array):
    """Copy a new array once; keep its mapping alive until the peer ACKs."""
    order='F' if array.flags.f_contiguous else 'C'
    mapping=_Mapping(shared_memory.SharedMemory(create=True,size=array.nbytes),owner=True)
    descriptor=SharedBuffer(mapping.shm.name,array.shape,array.dtype,order,claim_owner=True)
    result=_array(mapping,descriptor);np.copyto(result,array)
    return descriptor,mapping


def attach(descriptor):
    mapping=_mappings.get(descriptor.name)
    if mapping is None:mapping=_Mapping(shared_memory.SharedMemory(name=descriptor.name))
    if descriptor.claim_owner:mapping.owner=True
    return _array(mapping,descriptor)
