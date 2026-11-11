"""Event co-occurrence, keeping self membership and strict size ratios."""
import numpy as np


def event_network(events,shape):
    n=len(events);time=shape[3];spatial=int(np.prod(shape[:3]));labels=np.zeros(shape,np.uint16,order="F")
    footprints=[];windows=[];sizes=np.zeros(n)
    for i,event in enumerate(events):
        labels.ravel(order="F")[event]=min(i+1,65535)
        if len(event):
            p=np.unique(event%spatial);footprints.append(p);sizes[i]=len(p);t=event//spatial;windows.append((int(t.min()),int(t.max())))
        else:footprints.append(np.empty(0,int));windows.append(None)
    matrix=labels.reshape((spatial,time),order="F");at_time=[np.setdiff1d(matrix[:,t],0) for t in range(time)]
    counts_location=np.full((n,2),np.nan);counts_time=np.full((n,1),np.nan)
    same_location=np.empty((n,2),object);same_time=np.empty((n,1),object)
    for i,event in enumerate(events):
        same_location[i,0]=np.empty((0,0));same_location[i,1]=np.empty((0,0));same_time[i,0]=np.empty((0,0))
        if not len(event):continue
        values=np.setdiff1d(matrix[footprints[i],:],0).astype(np.uint16)
        ratio=sizes[i]/sizes[values.astype(int)-1];similar=values[(ratio<2)&(ratio>.5)]
        same_location[i,0]=(values.astype(np.int64)-1).reshape(-1,1);same_location[i,1]=(similar.astype(np.int64)-1).reshape(-1,1)
        counts_location[i]=[len(values),len(similar)]
        start,end=windows[i];values=np.unique(np.concatenate(at_time[start:end+1])).astype(np.uint16)
        same_time[i,0]=(values.astype(np.int64)-1).reshape(-1,1);counts_time[i,0]=len(values)
    return {"nOccurSameLoc":counts_location,"nOccurSameTime":counts_time,"occurSameLocList":same_location,"occurSameTimeList":same_time}


def network_mat_payload(network):
    result=dict(network)
    for key in ("occurSameLocList","occurSameTimeList"):
        output=np.empty(network[key].shape,object)
        for i,indices in enumerate(network[key].ravel()):
            output.ravel()[i]=(indices+1).astype(np.uint16) if indices.size else indices.astype(float)
        result[key]=output
    return result
