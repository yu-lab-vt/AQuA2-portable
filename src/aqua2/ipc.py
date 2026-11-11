"""Bounded-memory, typed local process transport. No intermediate result files."""
from dataclasses import dataclass,fields,is_dataclass
import numpy as np
from .io import MatStructArray

CHUNK_BYTES=4*1024**2
@dataclass
class Buffer:
    index:int
    shape:tuple
    dtype:object
    order:str
@dataclass
class Cells:
    shape:tuple
    fields:object
    values:list

@dataclass
class Record:
    cls:object
    values:dict

def send_value(connection,value,*,shared=False):
    from .shared_arrays import describe,publish
    buffers=[];created=[];published={};stats={'array_bytes':0,'copied_bytes':0,'reused_bytes':0,'pipe_bytes':0}
    def encode(v):
        if isinstance(v,np.ndarray):
            if v.dtype==object:return Cells(v.shape,tuple(v.mat_fields) if isinstance(v,MatStructArray) else None,[encode(x) for x in v.flat])
            if v.nbytes>=256*1024:
                stats['array_bytes']+=v.nbytes
                if shared:
                    descriptor=describe(v)
                    if descriptor is not None:stats['reused_bytes']+=v.nbytes;return descriptor
                    if id(v) not in published:
                        descriptor,mapping=publish(v);published[id(v)]=descriptor;created.append(mapping);stats['copied_bytes']+=v.nbytes
                    return published[id(v)]
                order='F' if v.flags.f_contiguous else 'C';flat=v.ravel(order=order)
                descriptor=Buffer(len(buffers),v.shape,v.dtype,order);buffers.append(flat.view(np.uint8));return descriptor
        if isinstance(v,dict):return {k:encode(x) for k,x in v.items() if not (k=='datR' and 'ov' in v and type(x).__name__=='Reconstruction')}
        if is_dataclass(v) and not isinstance(v,type):return Record(type(v),{f.name:encode(getattr(v,f.name)) for f in fields(v)})
        if isinstance(v,list):return [encode(x) for x in v]
        if isinstance(v,tuple):return tuple(encode(x) for x in v)
        return v
    tree=encode(value)
    connection.send({'kind':'value','tree':tree,'shared':shared})
    for buffer in buffers:
        view=memoryview(buffer)
        for start in range(0,len(view),CHUNK_BYTES):
            connection.send_bytes(view[start:start+CHUNK_BYTES]);stats['pipe_bytes']+=min(CHUNK_BYTES,len(view)-start)
    if shared:
        if connection.recv()!={'kind':'shared_ack'}:raise ValueError('Shared array attachment was not acknowledged')
        for mapping in created:mapping.owner=False
    return stats

def receive_value(connection,message=None,*,read_only=False):
    from .shared_arrays import SharedBuffer,attach
    message=message or connection.recv();buffers={}
    def decode(v):
        if isinstance(v,SharedBuffer):
            array=attach(v)
            if read_only:array.flags.writeable=False
            return array
        if isinstance(v,Record):return v.cls(**{k:decode(x) for k,x in v.values.items()})
        if isinstance(v,Buffer):
            array=np.empty(v.shape,dtype=np.dtype(v.dtype),order=v.order);buffers[v.index]=array
            return array
        if isinstance(v,Cells):
            array=np.empty(v.shape,object)
            if v.fields is not None:array=array.view(MatStructArray);array.mat_fields=v.fields
            for i,x in enumerate(v.values):array.flat[i]=decode(x)
            return array
        if isinstance(v,dict):return {k:decode(x) for k,x in v.items()}
        if isinstance(v,list):return [decode(x) for x in v]
        if isinstance(v,tuple):return tuple(decode(x) for x in v)
        return v
    value=decode(message['tree'])
    for index in range(len(buffers)):
        array=buffers[index];view=memoryview(array.ravel(order='F' if array.flags.f_contiguous else 'C').view(np.uint8))
        for start in range(0,len(view),CHUNK_BYTES):
            size=connection.recv_bytes_into(view[start:start+CHUNK_BYTES])
            if size!=min(CHUNK_BYTES,len(view)-start):raise ValueError('Incomplete result transfer')
    if message.get('shared'):connection.send({'kind':'shared_ack'})
    return value
