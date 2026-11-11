"""Display-only operations; scientific arrays and labels are never modified."""
import colorsys
import numpy as np
from scipy import ndimage


def color(index):return colorsys.hsv_to_rgb((.11+int(index)*.618033988749895)%1,.72,.96)


def polygon_pixels(vertices,shape,z=0):
    """Rasterize display X/Y pixel centres into zero-based Fortran HWL indices."""
    from skimage.draw import polygon
    points=np.asarray(vertices,float)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)<3 or not np.isfinite(points).all():raise ValueError('A polygon needs at least three valid vertices.')
    x,y=points.T
    if len(np.unique(points,axis=0))<3 or abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))<1e-9:raise ValueError('A polygon must enclose a nonzero area.')
    h,w,depth=shape[:3]
    if not 0<=z<depth:raise ValueError('Zslice out of range')
    rows,columns=polygon(y,x,shape=(h,w))
    return np.sort(rows.astype(np.int64)+columns*h+z*h*w)


class EventFrameIndex:
    """Search sorted event indices instead of rescanning all voxels per frame."""
    def __init__(self,events,shape):
        self.shape=shape;self.events=[];self.start=np.empty(len(events),np.int64);self.end=self.start.copy()
        size=int(np.prod(shape[:3]))
        for i,event in enumerate(events):
            ordered=event if np.all(event[:-1]<=event[1:]) else np.sort(event)
            self.events.append(ordered);self.start[i]=ordered[0]//size if len(ordered) else shape[-1];self.end[i]=ordered[-1]//size if len(ordered) else -1
    def slices(self,t,z,visible,all_time):
        h,w,depth,_=self.shape;plane=h*w;size=plane*depth
        active=range(len(self.events)) if all_time else np.flatnonzero((self.start<=t)&(self.end>=t))
        for i in active:
            if visible is not None and i not in visible:continue
            e=self.events[i]
            if all_time:
                p=e%size;p=p[p//plane==z]%plane
            else:
                lo=t*size+z*plane;p=e[np.searchsorted(e,lo):np.searchsorted(e,lo+plane)]-lo
            if len(p):yield i,p


def event_rows(project,channel,scope):
    events=project.events(channel,scope);features=project.features(channel,scope);rows=[];size=int(np.prod(project.shape[:3]))
    fts=features['fts'] if features else {};curve=fts.get('curve',{});basic=fts.get('basic',{})
    def field(group,key,i,fallback=np.nan):
        value=group.get(key)
        return float(np.asarray(value).ravel()[i]) if value is not None and np.asarray(value).size>i else fallback
    for i,e in enumerate(events):
        start=field(curve,'tBegin',i);end=field(curve,'tEnd',i)
        if not np.isfinite(start):start=int(np.min(e//size))+1 if len(e) else 1
        if not np.isfinite(end):end=int(np.max(e//size))+1 if len(e) else start
        rows.append({'id':i,'start':start,'end':end,'peak':field(curve,'dffMaxFrame',i,start),
            'area':field(basic,'area',i),'duration':field(curve,'duration',i,(end-start+1)*project.opts['frameRate']),
            'dff':field(curve,'dffMax',i),'tau':field(curve,'decayTau',i),'voxels':len(e)})
    return rows


def label_layer(project,mode,channel,scope,t,z,*,all_time=False,visible=None):
    h,w,depth,frames=project.shape;labels=np.zeros((h,w),np.int32);alpha=np.zeros((h,w),np.float32)
    visible=None if visible is None else set(visible)
    if mode=='CFU':
        signature=(channel,z,None if visible is None else tuple(sorted(visible)))
        cached=project._cache.get('display_cfu_layer')
        if cached is not None and cached[0]==signature:labels,alpha=cached[1:]
        else:
            for i,record in enumerate(project.cfus(channel)):
                if visible is not None and i not in visible:continue
                weights=record['weightMap'];weights=np.asarray(weights[:,:,z] if len(weights.shape)==3 else weights)
                mask=weights>.1;labels[mask]=i+1;alpha[mask]=np.maximum(.2,weights[mask])
            # CFU weights are static across frames. Keep only the most recent
            # visibility/Z combination, avoiding repeated full-map disk I/O.
            project._cache['display_cfu_layer']=(signature,labels,alpha)
    else:
        index_key=('display_event_index',channel,scope);cached=project._cache.get('display_event_index')
        if cached is None or cached[0]!=index_key:
            cached=(index_key,EventFrameIndex(project.events(channel,scope),project.shape));project._cache['display_event_index']=cached
        signature=(channel,scope,z,None if visible is None else tuple(sorted(visible)))
        overlay=project._cache.get('display_all_time_layer') if all_time else None
        if overlay is not None and overlay[0]==signature:labels,alpha=overlay[1:]
        else:
            for i,pixels in cached[1].slices(t,z,visible,all_time):labels[pixels%h,pixels//h]=i+1;alpha[pixels%h,pixels//h]=1
            if all_time:project._cache['display_all_time_layer']=(signature,labels,alpha)
    maximum=int(labels.max())
    palette=project._cache.get('display_palette')
    if palette is None or len(palette)<=maximum:
        palette=np.array([(0,0,0)]+[color(i) for i in range(maximum)],np.float32);project._cache['display_palette']=palette
    return labels,alpha,palette


def layer(project,mode,channel,scope,t,z,*,selected=(),all_time=False,visible=None):
    labels,alpha,palette=label_layer(project,mode,channel,scope,t,z,all_time=all_time,visible=visible)
    rgba=np.zeros((*labels.shape,4),np.float32)
    rgba[...,:3]=palette[labels];rgba[...,3]=alpha
    for i in selected:
        mask=labels==i+1
        if mask.any():
            edge=mask&~ndimage.binary_erosion(mask,structure=np.ones((3,3),bool));rgba[edge,:3]=[1,.92,.15];rgba[edge,3]=1
    return rgba,labels


def rising_image(project,channel,scope,index,z,percent=50):
    spatial=project.spatial(channel,scope);items=spatial['riseLst'].ravel(order='F')
    if not 0<=index<len(items):raise ValueError('This event has no saved rising map')
    record=items[index];h,w,depth,_=project.shape;result=np.full((h,w),np.nan)
    rows=np.asarray(record['rgh'],np.int64).ravel()-1;columns=np.asarray(record['rgw'],np.int64).ravel()-1
    planes=np.asarray(record.get('rgl',np.array([[1]])),np.int64).ravel()-1
    if z not in planes:return result
    values=np.asarray(record[f'dlyMap{percent}']).reshape((len(rows),len(columns),len(planes)),order='F')
    result[np.ix_(rows,columns)]=(values[:,:,int(np.flatnonzero(planes==z)[0])]-1)*float(project.opts['frameRate'])
    result[~np.isfinite(result)]=np.nan
    return result


def plot_curve(ax,index,x,y,*,record=None,label=None):
    """MATLAB CFU curveRefresh styling, preserving the displayed amplitudes."""
    stroke=color(index)
    lines=ax.plot(x,y,color=stroke,linewidth=1.15,label=label)
    if record is not None:
        # NaN gaps keep separate activity windows from being connected.
        # Draw gray last, matching MATLAB when the windows overlap.
        for key,width,tint in (('timeWindow',3,np.maximum(0,np.asarray(stroke[:3])-.1)),
                               ('nonTimeWindow',1.15,(.8,.8,.8))):
            mask=np.asarray(record.get(key,[]),dtype=bool).ravel()
            if mask.size==len(y) and mask.any():
                lines.extend(ax.plot(x,np.where(mask,y,np.nan),color=tint,linewidth=width))
    return lines


def curve_series(project,mode,channel,scope,indices,kind='dF/F',cleaned=True):
    dt=float(project.opts['frameRate']);series=[]
    if mode=='CFU':
        infos=project.cfus(channel)
        for i in indices:
            record=infos[i];values=np.asarray(record['meanDff' if kind=='dF/F' else 'meanCurve']).ravel()
            series.append((i,np.arange(len(values))*dt,values,np.flatnonzero(record['occurrence'])*dt))
    else:
        features=project.features(channel,scope)
        if not features:return []
        matrix=features['dffMat' if kind=='dF/F' else 'dMat']
        for i in indices:
            if len(matrix.shape)<3 or matrix.shape[1]==0:continue
            values=np.asarray(matrix[i,:,int(cleaned)])
            series.append((i,np.arange(len(values))*dt,values,np.empty(0)))
    return series
