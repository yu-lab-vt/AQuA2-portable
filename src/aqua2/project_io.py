"""Versioned, atomic HDF5 projects. No pickle, executable objects or squeezing."""
from dataclasses import dataclass
from pathlib import Path
import json,os,uuid
import h5py
import numpy as np
from .io import MatStructArray

FORMAT='AQuA2 Python project'
SCHEMA=1


@dataclass(frozen=True)
class ArrayRef:
    path: str
    node: str
    shape: tuple
    dtype: np.dtype
    matlab: bool=False

    @property
    def ndim(self):return len(self.shape)
    @property
    def size(self):return int(np.prod(self.shape))
    def __len__(self):return self.shape[0]
    def __getitem__(self,key):
        with h5py.File(self.path,'r') as file:
            if self.matlab:
                if key is Ellipsis:key=tuple(slice(None) for _ in self.shape)
                if not isinstance(key,tuple):key=(key,)
                key=key+(slice(None),)*(len(self.shape)-len(key))
                array=file[self.node][tuple(reversed(key))]
                return array.transpose(tuple(reversed(range(array.ndim)))) if array.ndim else array
            return file[self.node][key]
    def __array__(self,dtype=None,copy=None):
        result=self[...]
        if dtype is not None:result=result.astype(dtype,copy=False)
        return result.copy() if copy else result


def matlab_array(path,node):
    with h5py.File(path,'r') as file:
        data=file[node]
        if data.attrs.get('MATLAB_empty',False):return np.empty(tuple(data[()].astype(int)))
        return ArrayRef(str(Path(path).resolve()),node,tuple(reversed(data.shape)),data.dtype,True)


def _write_movie(node,value,compression):
    # Bounded C-contiguous slabs avoid h5py's implicit whole-F-movie copy.
    h,w,z,t=value.shape
    frames=max(1,min(t,8,16*1024**2//max(1,h*w*z*value.dtype.itemsize)))
    chunks=(min(h,64),min(w,64),min(z,1),frames)
    kwargs={'compression':compression,'shuffle':True}
    if compression=='gzip':kwargs['compression_opts']=1
    data=node.create_dataset('data',shape=value.shape,dtype=value.dtype,chunks=chunks,**kwargs)
    for start in range(0,t,frames):data[...,start:start+frames]=np.ascontiguousarray(value[...,start:start+frames])


def _write(node,value,*,embed_external=False,compression='gzip'):
    if isinstance(value,ArrayRef):
        if value.matlab and not embed_external:
            node.attrs['kind']='external';node.attrs['path']=value.path;node.attrs['node']=value.node
            node.attrs['shape']=value.shape;node.attrs['dtype']=value.dtype.str
            return
        node.attrs['kind']='array'
        if not value.matlab:
            with h5py.File(value.path,'r') as source:source.copy(value.node,node,name='data')
            return
        chunks=tuple(min(n,64) for n in value.shape[:-1])+(1,) if value.ndim==4 else True
        dataset=node.create_dataset('data',shape=value.shape,dtype=value.dtype,chunks=chunks,compression='gzip',compression_opts=1)
        if value.ndim==4:
            for t in range(value.shape[-1]):dataset[...,t]=value[:,:,:,t]
        else:dataset[...]=np.asarray(value)
    elif isinstance(value,dict):
        node.attrs['kind']='dict';node.attrs['keys']=json.dumps(list(value),ensure_ascii=False)
        for i,child in enumerate(value.values()):_write(node.create_group(str(i)),child,embed_external=embed_external,compression=compression)
    elif isinstance(value,(list,tuple)):
        node.attrs['kind']='tuple' if isinstance(value,tuple) else 'list'
        node.attrs['length']=len(value)
        for i,child in enumerate(value):_write(node.create_group(str(i)),child,embed_external=embed_external,compression=compression)
    elif isinstance(value,np.ndarray) and value.dtype==object:
        node.attrs['kind']='struct_array' if isinstance(value,MatStructArray) else 'cell'
        node.attrs['shape']=value.shape
        if isinstance(value,MatStructArray):node.attrs['fields']=json.dumps(list(value.mat_fields))
        for i,child in enumerate(value.flat):_write(node.create_group(str(i)),child,embed_external=embed_external,compression=compression)
    elif isinstance(value,np.ndarray):
        node.attrs['kind']='array'
        if value.dtype.kind in 'US':
            node.attrs['text_dtype']=value.dtype.str
            node.create_dataset('data',data=value.astype(object),dtype=h5py.string_dtype('utf-8'))
        else:
            if value.ndim==4 and value.nbytes>=64*1024**2:
                _write_movie(node,value,compression)
            else:
                kwargs={'compression':compression,'shuffle':True} if value.ndim and value.size>4096 else {}
                if kwargs and compression=='gzip':kwargs['compression_opts']=1
                node.create_dataset('data',data=value,**kwargs)
    elif value is None:node.attrs['kind']='none'
    elif isinstance(value,(str,Path)):
        node.attrs['kind']='str';node.create_dataset('data',data=str(value),dtype=h5py.string_dtype('utf-8'))
    elif isinstance(value,(bool,int,float,np.number,np.bool_)):
        node.attrs['kind']='scalar';node.attrs['python_scalar']=not isinstance(value,np.generic);node.create_dataset('data',data=value)
    else:raise TypeError(f'Unsupported project value: {type(value).__name__}')


def _read(node,path,lazy):
    kind=node.attrs['kind']
    if kind=='external':return ArrayRef(str(node.attrs['path']),str(node.attrs['node']),tuple(node.attrs['shape']),np.dtype(node.attrs['dtype']),True)
    if kind=='dict':return {key:_read(node[str(i)],path,lazy) for i,key in enumerate(json.loads(node.attrs['keys']))}
    if kind in ('list','tuple'):
        values=[_read(node[str(i)],path,lazy) for i in range(int(node.attrs['length']))]
        return tuple(values) if kind=='tuple' else values
    if kind in ('cell','struct_array'):
        array=np.empty(tuple(node.attrs['shape']),object)
        if kind=='struct_array':array=array.view(MatStructArray);array.mat_fields=tuple(json.loads(node.attrs['fields']))
        for i in range(array.size):array.flat[i]=_read(node[str(i)],path,lazy)
        return array
    if kind=='none':return None
    if kind=='str':return node['data'].asstr()[()]
    if kind=='scalar':
        value=node['data'][()]
        python_scalar=node.attrs.get('python_scalar',node['data'].dtype.itemsize>=8)
        return value.item() if python_scalar else value
    if kind!='array':raise ValueError(f'Unknown project node kind {kind!r}')
    dataset=node['data']
    if 'text_dtype' in node.attrs:return np.asarray(dataset.asstr()[()],dtype=np.dtype(node.attrs['text_dtype']))
    if lazy and dataset.size*dataset.dtype.itemsize>=512*1024:return ArrayRef(str(path),dataset.name,dataset.shape,dataset.dtype)
    if dataset.ndim==4 and dataset.size*dataset.dtype.itemsize>=64*1024**2:
        result=np.empty(dataset.shape,dtype=dataset.dtype,order='F')
        frames=max(1,min(dataset.shape[-1],max(8,dataset.chunks[-1] if dataset.chunks else 1),max(1,256*1024**2//max(1,int(np.prod(dataset.shape[:-1]))*dataset.dtype.itemsize))))
        for start in range(0,dataset.shape[-1],frames):result[...,start:start+frames]=dataset[...,start:start+frames]
        return result
    return np.asarray(dataset[()])


def save_tree(path,value,*,embed_external=False,compression='gzip'):
    """Replace a completed project atomically; failures leave old data intact."""
    if compression not in ('gzip','lzf'):raise ValueError('Compression must be gzip or lzf')
    path=Path(path).resolve()
    if path.suffix.lower()!='.h5':raise ValueError('Python projects use .h5 files')
    if path.exists():
        with h5py.File(path,'r') as file:
            if file.attrs.get('format')!=FORMAT:raise ValueError('Refusing to overwrite a different HDF5 format')
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    try:
        with h5py.File(temporary,'w') as file:
            file.attrs.update(format=FORMAT,schema=SCHEMA)
            _write(file.create_group('project'),value,embed_external=embed_external,compression=compression)
            file.flush()
        os.replace(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()
    return path


def load_tree(path,*,lazy=True):
    path=Path(path).resolve()
    with h5py.File(path,'r') as file:
        if file.attrs.get('format')!=FORMAT or file.attrs.get('schema')!=SCHEMA:raise ValueError('Unsupported AQuA2 Python project format/version')
        return _read(file['project'],path,lazy)


def materialize(value):
    from .reconstruction import Reconstruction
    if isinstance(value,Reconstruction):return np.asarray(value)
    if isinstance(value,ArrayRef):return np.asarray(value)
    if isinstance(value,dict):return {k:materialize(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(materialize(v) for v in value)
    if isinstance(value,np.ndarray) and value.dtype==object:
        result=value.copy()
        for i,child in enumerate(value.flat):result.flat[i]=materialize(child)
        return result
    return value
