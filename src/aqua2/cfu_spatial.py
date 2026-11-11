"""CFU weighted event overlaps and spatial-basin memberships (zero based)."""
import numpy as np
from numba import njit
from scipy import ndimage,sparse
from skimage.morphology import reconstruction,local_maxima
from .act import _components
from .pre import gaussian_xy
from .se_common import impose_minima
from .se_watershed import meyer_watershed


@njit(cache=True)
def _sorted_union_int64(a,b):
    """Merge already sorted unique pixel IDs; retain union1d's sorted output."""
    output=np.empty(len(a)+len(b),np.int64);i=0;j=0;k=0
    while i<len(a) and j<len(b):
        if a[i]<b[j]:output[k]=a[i];i+=1
        elif b[j]<a[i]:output[k]=b[j];j+=1
        else:output[k]=a[i];i+=1;j+=1
        k+=1
    while i<len(a):output[k]=a[i];i+=1;k+=1
    while j<len(b):output[k]=b[j];j+=1;k+=1
    return output[:k]


def _pixel_union(a,b):
    if a.dtype==np.int64 and b.dtype==np.int64:return _sorted_union_int64(a,b)
    return np.union1d(a,b)


def prepare_events(events,shape,weighted=True):
    spatial=int(np.prod(shape[:3]));n=len(events);pixels=[];weights=[];counts=[]
    if n>65535:raise ValueError('Source uint16 event labels support at most 65535 events')
    for event in events:
        if np.any(np.asarray(event)<0) or np.any(np.asarray(event)>=spatial*shape[3]):raise ValueError('Event voxel index is outside HWLT movie')
        p,c=np.unique(np.asarray(event)%spatial,return_counts=True)
        if not len(p):raise ValueError('CFU event lists must not contain empty events')
        pixels.append(p);weights.append(c/c.max() if weighted else np.ones(len(c)));counts.append(c.max())
    # Equivalent to the source's uint16 label volume, including last-event
    # overwrite, without allocating an H*W*L*T label movie.
    if n:
        voxels=np.concatenate(events);owners=np.repeat(np.arange(n),[len(e) for e in events])
        _,reverse=np.unique(voxels[::-1],return_index=True);last=len(voxels)-1-reverse
        visible=sparse.coo_matrix((np.ones(len(last)),(voxels[last]%spatial,owners[last])),shape=(spatial,n)).tocsr()
        visible.sum_duplicates()
    edges=[];neighbors=[]
    for i,p in enumerate(pixels):
        possible=np.setdiff1d(np.unique(visible[p].indices),i).astype(np.int64);neighbors.append(possible)
        for j in possible[possible>i]:
            union=_pixel_union(p,pixels[j]);a=np.zeros(len(union));b=a.copy()
            a[np.searchsorted(union,p)]=weights[i];b[np.searchsorted(union,pixels[j])]=weights[j]
            edges.append([i,j,np.minimum(a,b).sum()/np.maximum(a,b).sum()])
    return {'s_t0':np.asarray(edges,float).reshape(-1,3),'evtIhw':pixels,'weightedIhw':weights,'maxCounts':np.asarray(counts,float),'overlapLst':neighbors}


def hierarchical_groups(linkage,n,threshold):
    """Minimum-edge single linkage; threshold comparison is strictly '<'."""
    linkage=np.asarray(linkage,float).reshape(-1,3);linkage=linkage[(linkage[:,2]<threshold)&(linkage[:,0]!=linkage[:,1])]
    pairs={(int(a),int(b)) for a,b,_ in linkage}
    if any((b,a) in pairs for a,b in pairs):
        raise ValueError('Symmetric duplicate linkage is outside the batch contract (source duplicate-edge indexing is invalid)')
    parent=np.arange(n)
    def root(i):
        while parent[i]!=i:i=parent[i]
        return i
    for a,b,_ in linkage[np.argsort(linkage[:,2],kind='stable')]:
        a,b=root(int(a)),root(int(b));parent[a]=parent[b]=min(a,b)
    groups=[np.array([i for i in range(n) if root(i)==r],np.int64) for r in range(n)]
    # label2idx includes empty labels through max label; sorting keeps those
    # after nonempty groups, relevant to hierarchy parent IDs.
    if n:groups=groups[:max(root(i) for i in range(n))+1]
    return sorted(groups,key=lambda g:-len(g))


def spatial_basins(weight,support):
    footprint=np.ones((3,)*weight.ndim,bool);components=_components(support)
    primary=np.zeros_like(support)
    if not components:return primary,np.zeros(weight.shape,np.uint16)
    maximum=max(map(len,components))
    for p in components:
        if len(p)>=maximum*.15:primary.ravel(order='F')[p]=True
    smooth=gaussian_xy(weight,1);smooth/=smooth.max();smooth[~primary]=0
    lowered=(smooth-np.float32(.12)).astype(np.float32)
    hmax=reconstruction(lowered,smooth,method='dilation',footprint=footprint).astype(np.float32)
    seeds=local_maxima(hmax,footprint=footprint)&primary&(smooth>=.3)
    labels=np.zeros(weight.shape,np.uint16,order='F');next_label=0
    for p in _components(primary):
        mask=np.zeros(weight.shape,bool,order='F');mask.ravel(order='F')[p]=True
        markers=seeds&mask
        if not markers.any():continue
        elevation=np.ones(weight.shape,np.float32);elevation[mask]=-smooth[mask]
        imposed=impose_minima(elevation,markers);imposed[~mask]=1
        component=meyer_watershed(imposed,markers);component[~mask]=0
        for label in np.setdiff1d(np.unique(component[markers]),0):
            next_label+=1;labels[component==label]=next_label
    # Merge shallow saddles, then restore only exterior-facing ridges.
    peaks=[smooth[labels==i].max() for i in range(1,next_label+1)];parent=np.arange(next_label)
    def root(i):
        while parent[i]!=i:i=parent[i]
        return i
    ridge=(labels==0)&primary
    grown=[ndimage.binary_dilation(labels==i,structure=footprint) for i in range(1,next_label+1)]
    for i in range(next_label-1):
        for j in range(i+1,next_label):
            saddle=ridge&grown[i]&grown[j]
            if saddle.any() and smooth[saddle].max()>.8*min(peaks[i],peaks[j]):parent[root(j)]=root(i)
    merged=np.zeros_like(labels);roots=[]
    for i in range(next_label):
        r=root(i)
        if r not in roots:roots.append(r)
        merged[labels==i+1]=roots.index(r)+1
    labels=merged;ridge=primary&(labels==0);neighbor=np.zeros_like(labels);several=np.zeros_like(primary)
    for i in range(1,len(roots)+1):
        touches=ridge&ndimage.binary_dilation(labels==i,structure=footprint)
        several|=touches&(neighbor>0)&(neighbor!=i);neighbor[touches&(neighbor==0)]=i
    assign=ridge&~several&(neighbor>0);labels[assign]=neighbor[assign]
    retained=np.zeros_like(primary)
    for p in _components(primary):
        if (labels.ravel(order='F')[p]>0).any():retained.ravel(order='F')[p]=True
    if not retained.any():
        p=max(_components(primary),key=len);retained.ravel(order='F')[p]=True
    return retained,labels


def refine_spatial(candidates,pre,shape,min_events,parent_ids=None):
    spatial_shape=tuple(shape[:2]) if shape[2]==1 else tuple(shape[:3]);size=int(np.prod(spatial_shape))
    output=[];parents=range(len(candidates)) if parent_ids is None else parent_ids
    for ids,parent_id in zip(candidates,parents):
        if len(ids)<min_events:continue
        iou=np.eye(len(ids))
        for i in range(len(ids)):
            for j in range(i+1,len(ids)):
                a,b=pre['evtIhw'][ids[i]],pre['evtIhw'][ids[j]];hit=len(np.intersect1d(a,b,assume_unique=True))
                iou[i,j]=iou[j,i]=hit/max(len(a)+len(b)-hit,1)
        nodes=iou.sum(axis=1);nodes/=nodes.max();vector=np.zeros(size,np.float32)
        for node,event_id in zip(nodes,ids):
            evidence=pre['weightedIhw'][event_id].astype(np.float32)*np.float32(pre['maxCounts'][event_id])
            vector[pre['evtIhw'][event_id]]+=np.float32(node)*evidence
        if vector.max()>0:vector/=vector.max()
        weight=np.array(vector.reshape(spatial_shape,order='F'),order='F');support=weight>.1
        primary,labels=spatial_basins(weight,support);n=int(labels.max());flat=labels.ravel(order='F')
        branch=[];areas=np.bincount(flat,minlength=n+1)
        for basin in range(1,n+1):
            events=[];pixels=[];scores=[]
            for event_id in ids:
                p=pre['evtIhw'][event_id];hit=np.bincount(flat[p],minlength=n+1)[1:]/np.maximum(areas[1:],1)
                if hit[basin-1]>=.5 and hit[basin-1]>=.75*hit.max():
                    events.append(event_id);pixels.append(p[flat[p]==basin]);scores.append(hit[basin-1])
            if events:branch.append((np.asarray(events,np.int64),pixels,np.asarray(scores),labels==basin))
        if n<2 or len(branch)<2:
            pixels=[];scores=[]
            for event_id in ids:
                p=pre['evtIhw'][event_id];e=pre['weightedIhw'][event_id]*pre['maxCounts'][event_id];keep=primary.ravel(order='F')[p]
                pixels.append(p[keep]);scores.append(e[keep].sum()/max(e.sum(),np.finfo(float).eps))
            branch=[(np.asarray(ids,np.int64),pixels,np.asarray(scores),primary)]
        for events,pixels,scores,mask in branch:
            region=weight.copy();region[~mask]=0
            member={'EventID':events,'SpatialPixels':pixels,'Score':scores,'VoxelIndices':[np.empty(0,np.int64) for _ in events],
                'SupportPixels':np.flatnonzero(support.ravel(order='F')),'SupportWeights':weight.ravel(order='F')[support.ravel(order='F')],
                'PrimaryPixels':np.flatnonzero(mask.ravel(order='F')),'SpatialSize':np.asarray(spatial_shape)}
            output.append({'region':region,'events':events,'parent':int(parent_id),'memberships':member})
    return output


def minimum_measure(pre,shape,threshold=.5,min_events=3):
    linkage=pre['s_t0'].copy();linkage[:,2]=1-linkage[:,2]
    groups=hierarchical_groups(linkage,len(pre['evtIhw']),1-threshold)
    parents=[i for i,g in enumerate(groups) if len(g)>=min_events]
    return refine_spatial([groups[i] for i in parents],pre,shape,min_events,parents)


def materialize_memberships(cfus,events,shape):
    size=int(np.prod(shape[:3]))
    for cfu in cfus:
        member=cfu['memberships']
        member['VoxelIndices']=[events[e][np.isin(events[e]%size,p)] for e,p in zip(member['EventID'],member['SpatialPixels'])]
    return cfus


def membership_time_windows(cfus,shape):
    n=len(cfus);size=int(np.prod(shape[:3]));own=np.zeros((n,shape[3]),bool);other=np.zeros_like(own);overlap=[]
    for i,c in enumerate(cfus):
        for voxels in c['memberships']['VoxelIndices']:own[i,np.unique(voxels//size)]=True
    for i,c in enumerate(cfus):
        target=np.flatnonzero((c['region']>.1).ravel(order='F'));neighbors=[]
        for j,source in enumerate(cfus):
            if j==i:continue
            found=False
            for v in source['memberships']['VoxelIndices']:
                selected=np.isin(v%size,target)
                if selected.any():found=True;other[i,np.unique(v[selected]//size)]=True
            if found:neighbors.append(j)
        overlap.append(np.asarray(neighbors,np.int64))
    other[own]=False
    return own,other,overlap
