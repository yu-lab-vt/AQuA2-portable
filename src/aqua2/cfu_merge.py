"""Conservative complete-linkage CFU merging and weighted trace extraction."""
import numpy as np
from scipy import ndimage
from .pre import obtain_bias
from .se_common import matlab_round
from .fea import native_sum


def moving_mean(curve,window):
    n=len(curve);k=np.arange(n);lo=np.maximum(0,k-int(window)//2);hi=np.minimum(n,k+(int(window)-1)//2+1)
    prefix=np.r_[0.,np.cumsum(curve,dtype=float)]
    return (prefix[hi]-prefix[lo])/(hi-lo)


def calculate_dff(curve,window,cut):
    smooth=moving_mean(curve,window);n=len(curve);step=int(matlab_round(.5*cut));segments=max(1,int(np.ceil(n/step))-1);baseline=np.zeros(n)
    for i in range(segments):
        first=i*step;last=min(n,first+int(cut)+1);frame=first+int(np.argmin(smooth[first:last]));value=smooth[frame]
        if i==0:baseline[:frame+1]=value
        else:
            with np.errstate(invalid='ignore',divide='ignore'):
                baseline[previous_frame:frame+1]=previous+np.divide(value-previous,frame-previous_frame)*np.arange(frame-previous_frame+1)
        if i==segments-1:baseline[frame:]=value
        previous,previous_frame=value,frame
    differences=np.diff(curve)**2
    noise=max(1e-4,np.sqrt(native_sum(differences)/len(differences)/2));baseline-=obtain_bias(window,cut)*noise
    return (curve-baseline)/(baseline+1e-4)


def compute_curves(cfus,movie,window,cut):
    """Movie may be an HWLT array or an object exposing weighted_curve.

    The source casts stored intensity to single. Mixed single/double matrix
    multiplication still produces single, as does division by sum(single).
    A saved-movie reader can select the needed pixels without a float movie.
    """
    nframes=movie.shape[-1];curves=np.zeros((len(cfus),nframes));dff=np.zeros_like(curves)
    for i,c in enumerate(cfus):
        weights=c['region'].ravel(order='F');pixels=np.flatnonzero(weights>0);w=weights[pixels]
        if hasattr(movie,'weighted_curve'):curve=movie.weighted_curve(pixels,w)
        else:
            data=movie.reshape((-1,nframes),order='F')[pixels,:].astype(np.float32,order='F')
            curve=(w@data/native_sum(w)).astype(float)
        curves[i]=curve;dff[i]=calculate_dff(curve,window,cut)
    return curves,dff


def default_merge_parameters():
    return dict(Enable=True,ExcludeSiblingBasins=True,SpatialDilationRadius=1,MinimumOverlapPixels=1,
        MinimumCurveCorrelation=.925,MinimumPositiveAUCRatio=.5,MaximumMergeRounds=5)


def curve_features(first,second):
    first=np.asarray(first,float);second=np.asarray(second,float)
    if len(first)!=len(second) or not np.isfinite(first).all() or not np.isfinite(second).all() or np.std(first)==0 or np.std(second)==0:return -np.inf,0.
    centered=np.stack((first-native_sum(first)/len(first),second-native_sum(second)/len(second)))
    covariance=(centered@centered.T)/(len(first)-1)
    deviation=np.sqrt(np.diag(covariance));correlations=covariance/deviation[:,None]/deviation[None,:]
    correlation=np.clip((correlations[0,1]+correlations[1,0])/2,-1,1)
    a=np.trapezoid(np.maximum(first,0));b=np.trapezoid(np.maximum(second,0))
    return correlation,min(a,b)/max(a,b,np.finfo(float).eps)


def complete_linkage_groups(edges,similarity):
    groups=[[i] for i in range(len(edges))]
    while True:
        best=-np.inf;pair=None
        for i in range(len(groups)-1):
            for j in range(i+1,len(groups)):
                select=np.ix_(groups[i],groups[j])
                if edges[select].all():
                    score=similarity[select].ravel(order='F').mean()
                    if score>best:best=score;pair=(i,j)
        if pair is None:return groups
        i,j=pair;groups[i]+=groups[j];del groups[j]


def _stable_unique(values):
    values=np.asarray(values);_,positions=np.unique(values,return_index=True)
    return values[np.sort(positions)]


def merge_memberships(members,region,sources,parents):
    events=_stable_unique(np.concatenate([m['EventID'] for m in members]));pixels=[];voxels=[];scores=[]
    for e in events:
        selected=[(m,i) for m in members for i in np.flatnonzero(m['EventID']==e)]
        pixels.append(np.unique(np.concatenate([m['SpatialPixels'][i] for m,i in selected])))
        voxels.append(np.unique(np.concatenate([m['VoxelIndices'][i] for m,i in selected])))
        scores.append(max(m['Score'][i] for m,i in selected))
    all_pixels=np.concatenate([m['SupportPixels'] for m in members]);weights=np.concatenate([m['SupportWeights'] for m in members]);support,location=np.unique(all_pixels,return_inverse=True)
    support_weights=np.full(len(support),-np.inf,dtype=weights.dtype);np.maximum.at(support_weights,location,weights)
    return {'EventID':events,'SpatialPixels':pixels,'VoxelIndices':voxels,'Score':np.asarray(scores),
        'SupportPixels':support,'SupportWeights':support_weights,'PrimaryPixels':np.flatnonzero((region>.1).ravel(order='F')),
        'SpatialSize':np.asarray(region.shape),'MergedFromCFUs':np.asarray(sources,np.int64),'MergedParentIDs':_stable_unique(parents)}


def merge_similar(cfus,dff,parameters=None,candidates=None):
    p=default_merge_parameters() if parameters is None else parameters;n=len(cfus)
    diagnostics=dict.fromkeys(['EvaluatedPairs','SiblingExcludedPairs','SpatialRejectedPairs','SpatialCandidatePairs','CorrelationRejectedPairs','AUCRejectedPairs','AcceptedEdges'],0)
    if not p['Enable'] or n<2:return cfus,[],[[i] for i in range(n)],diagnostics
    candidates=set(range(n) if candidates is None or not len(candidates) else candidates)
    edges=np.zeros((n,n),bool);similarity=np.full((n,n),-np.inf)
    masks=[c['region']>.1 for c in cfus];radius=int(p['SpatialDilationRadius'])
    grown=[ndimage.binary_dilation(m,structure=np.ones((2*radius+1,)*m.ndim,bool)) if radius>0 else m for m in masks]
    for i in range(n-1):
        for j in range(i+1,n):
            if i not in candidates and j not in candidates:continue
            diagnostics['EvaluatedPairs']+=1
            if p['ExcludeSiblingBasins'] and cfus[i]['parent']==cfus[j]['parent']:diagnostics['SiblingExcludedPairs']+=1;continue
            if np.count_nonzero(grown[i]&masks[j])<p['MinimumOverlapPixels']:diagnostics['SpatialRejectedPairs']+=1;continue
            diagnostics['SpatialCandidatePairs']+=1;correlation,ratio=curve_features(dff[i],dff[j])
            a=correlation>=p['MinimumCurveCorrelation'];b=ratio>=p['MinimumPositiveAUCRatio']
            if not a:diagnostics['CorrelationRejectedPairs']+=1
            if not b:diagnostics['AUCRejectedPairs']+=1
            if not a or not b:continue
            diagnostics['AcceptedEdges']+=1;edges[i,j]=edges[j,i]=True;similarity[i,j]=similarity[j,i]=correlation
    groups=complete_linkage_groups(edges,similarity);changed=[i for i,g in enumerate(groups) if len(g)>1]
    if not changed:return cfus,changed,groups,diagnostics
    output=[]
    for group in groups:
        source=[cfus[i] for i in group];region=np.maximum.reduce([c['region'] for c in source])
        member=merge_memberships([c['memberships'] for c in source],region,group,np.array([c['parent'] for c in source]))
        output.append(dict(region=region,events=_stable_unique(np.concatenate([c['events'] for c in source])),parent=source[0]['parent'],memberships=member))
    return output,changed,groups,diagnostics


def iterative_merge(cfus,curves,dff,movie,window,cut,parameters=None,progress=None):
    p=default_merge_parameters() if parameters is None else parameters;candidates=list(range(len(cfus)));rounds=[];did_merge=False
    for i in range(int(p['MaximumMergeRounds'])):
        if progress:progress('evaluate',i+1,len(cfus))
        cfus,changed,groups,diagnostics=merge_similar(cfus,dff,p,candidates);rounds.append(diagnostics)
        if not changed:break
        did_merge=True;source=[g[0] for g in groups];curves=curves[source];dff=dff[source]
        if progress:progress('recompute',i+1,len(changed))
        new_curves,new_dff=compute_curves([cfus[j] for j in changed],movie,window,cut)
        curves[changed]=new_curves;dff[changed]=new_dff;candidates=changed
    total={key:sum(r[key] for r in rounds) for key in rounds[0]} if rounds else {}
    return cfus,curves,dff,dict(didMerge=did_merge,rounds=len(rounds),diagnostics={'Rounds':rounds,'Total':total})
