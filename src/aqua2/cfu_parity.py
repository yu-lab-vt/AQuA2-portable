"""Layered CFU comparison; discrete science always uses zero tolerance.

Single SGEMM forward-error bounds are kept separate from double downstream
checks. This comparator never relaxes membership, maps, labels or delays.
"""
import numpy as np
from .io import MatStructArray
from .compare import compare,Tolerance
from .cfu_merge import calculate_dff,moving_mean
from .pre import obtain_bias
from .se_common import matlab_round

DOUBLE=Tolerance(atol=128*np.finfo(float).eps,rtol=1024*np.finfo(float).eps)


def _anchors(curve,window,cut):
    smooth=moving_mean(curve,window);step=int(matlab_round(.5*cut));n=len(curve)
    return np.array([i*step+np.argmin(smooth[i*step:min(n,i*step+int(cut)+1)]) for i in range(max(1,int(np.ceil(n/step))-1))])


def _science(payload):
    output={};aux={}
    for key,value in payload.items():
        if key in ('cfuInfo1','cfuInfo2') and isinstance(value,(MatStructArray,dict)):
            records=[value] if isinstance(value,dict) else list(value.ravel(order='F'))
            output[key]={};aux[key]={}
            for i,record in enumerate(records):
                selected={k:v for k,v in record.items() if k not in ('meanCurve','meanDff','frequencyStats')}
                if isinstance(record.get('frequencyStats'),dict):
                    freq=record['frequencyStats'];selected['frequencyCount']=freq['count'];selected['frequencyMethod']=freq['method']
                    aux[key][str(i)]={k:v for k,v in freq.items() if k not in ('count','method')}
                output[key][str(i)]=selected
            output[key+'_shape']=np.array([[1,1] if isinstance(value,dict) else list(value.shape)],np.int64)
        elif key=='cfuRelation':
            array=np.asarray(value).reshape(-1,4);output[key]=array[:,[0,1,3]];aux[key]=array[:,2]
        elif key=='cfuGroupInfo':
            output[key]=value[:,:3];aux[key]=value[:,3:]
        else:output[key]=value
    return output,aux


def compare_cfu(reference,actual,window,cut):
    ref_science,ref_aux=_science(reference);got_science,got_aux=_science(actual)
    science=compare(ref_science,got_science);auxiliary=compare(ref_aux,got_aux,tolerance=DOUBLE);curves=[]
    for channel in ('cfuInfo1','cfuInfo2'):
        a=reference[channel];b=actual[channel]
        expected=[a] if isinstance(a,dict) else list(a.ravel(order='F'))
        observed=[b] if isinstance(b,dict) else list(b.ravel(order='F'))
        if len(expected)!=len(observed):continue
        for i,(r,p) in enumerate(zip(expected,observed)):
            x=r['meanCurve'];y=p['meanCurve'];df=r['meanDff'];dy=p['meanDff']
            compatible=x.shape==y.shape==df.shape==dy.shape and x.dtype==y.dtype==df.dtype==dy.dtype and np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(df).all() and np.isfinite(dy).all()
            if not compatible:curves.append(dict(channel=channel,index=i,passed=False,reason='curve shape/type/nonfinite mismatch'));continue
            x=x.ravel();y=y.ravel();df=df.ravel();dy=dy.ravel();n=int(np.count_nonzero(r['weightMap']>0));u=np.finfo(np.float32).eps/2
            gamma=n*u/(1-n*u)
            # Each single GEMM has <=gamma_n weighted-positive-sum error;
            # two kernels and their final single divisions contribute here.
            raw_limit=(2*gamma+2*u)*np.maximum(abs(x),abs(y))/(1-gamma)
            delta=float(np.max(abs(x-y),initial=0));raw_pass=bool(np.all(abs(x-y)<=raw_limit))
            fixed=compare(r['meanDff'],calculate_dff(x,window,cut).reshape(r['meanDff'].shape))
            same_anchors=np.array_equal(_anchors(x,window,cut),_anchors(y,window,cut))
            e=1e-4;bx=(x-df*e)/(1+df);by=(y-dy*e)/(1+dy)
            # With identical knots: min/movmean/interpolation are 1-Lipschitz;
            # the RMS first-difference noise changes by at most sqrt(2)*delta.
            baseline_limit=delta*(1+abs(obtain_bias(window,cut))*np.sqrt(2))+512*np.finfo(float).eps*np.maximum(abs(bx),abs(by))
            dff_limit=delta/abs(by+e)+abs(x+e)*baseline_limit/(abs(by+e)*abs(bx+e))+128*np.finfo(float).eps*np.maximum(1,abs(df))
            propagated=bool(np.all(abs(df-dy)<=dff_limit));curve_pass=raw_pass and fixed['passed'] and same_anchors and propagated
            curves.append(dict(channel=channel,index=i,passed=curve_pass,pixels=n,raw_single_forward_bound=raw_pass,
                fixed_raw_dff_exact=fixed['passed'],baseline_anchor_frames_exact=bool(same_anchors),dff_propagated_bound=propagated,
                max_raw_abs_error=delta,max_raw_relative_error=float(np.max(abs(x-y)/np.maximum(abs(x),np.finfo(float).tiny))),
                max_raw_ulp_error=float(np.max(abs(x-y)/np.spacing(x.astype(np.float32)))),max_dff_abs_error=float(np.max(abs(df-dy),initial=0))))
    return dict(passed=science['passed'] and auxiliary['passed'] and all(c['passed'] for c in curves),
        discrete_maps_memberships_occurrence_delays_exact=science,double_frequency_pvalues=auxiliary,curves=curves,
        raw_budget='Two single positive SGEMM gamma_n bounds plus final single rounding; n=positive weight pixels, u=eps32/2',
        dff_budget='Exact fixed-raw DFF, identical minimum-anchor frames, propagated raw/noise bound; no discrete tolerance')
