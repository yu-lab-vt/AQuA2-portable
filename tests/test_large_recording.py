"""Large-recording optimizations preserve the prior numerical and index rules."""
import ast,importlib.util,tempfile,unittest
from pathlib import Path
import numpy as np
from scipy import ndimage
from aqua2.act import _components,bw2reg
from aqua2.components import _sparse_roots
from aqua2.pre import moving_mean
from aqua2.project_io import save_tree,load_tree
from aqua2.se_common import my_resize,_resize_axis


def old_components(mask):
    labels,_=ndimage.label(mask,structure=np.ones((3,)*mask.ndim,bool));flat=labels.ravel(order='F');indices=np.flatnonzero(flat)
    if not len(indices):return []
    indices=indices[np.argsort(flat[indices],kind='stable')]
    return sorted(np.split(indices,np.flatnonzero(np.diff(flat[indices]))+1),key=lambda g:g[0])


class LargeRecordingOptimizations(unittest.TestCase):
    def test_sparse_dense_and_filtered_components(self):
        rng=np.random.default_rng(71)
        for shape in [(1,),(1,1),(8,9),(8,9,1,12),(8,9,3,12),(0,3),(31,32,1,35),(5,6,3,8)]:
            for density in [0,.001,.02,.25,1.]:
                for order in ['C','F']:
                    mask=np.array(rng.random(shape)<density,order=order);gold=old_components(mask)
                    for minimum in [1,2,5,20]:
                        expected=[g for g in gold if len(g)>=minimum];actual=_components(mask,minimum)
                        self.assertEqual(len(expected),len(actual))
                        for a,b in zip(expected,actual):np.testing.assert_array_equal(a,b)
    def test_hash_sparse_path(self):
        from unittest.mock import patch
        import aqua2.components as components
        rng=np.random.default_rng(451)
        mask=rng.random((31,32,1,35))<.002
        mask[0:4,0:4,0,0:4]=True
        with patch.object(components,'_SPARSE_BINARY_LIMIT',0):actual=_components(mask,3)
        expected=[g for g in old_components(mask) if len(g)>=3]
        self.assertEqual(len(expected),len(actual))
        for a,b in zip(expected,actual):np.testing.assert_array_equal(a,b)
    def test_dilated_original_pixel_filter(self):
        rng=np.random.default_rng(35);mask=rng.random((17,18,1,15))<.04
        for radius in [0,1,2]:
            opts={'spaMergeDist':radius};expected=bw2reg(mask,opts)
            for minimum in [2,10,20]:
                actual=bw2reg(mask,opts,min_voxels=minimum);gold=[g for g in expected if len(g)>=minimum]
                self.assertEqual(len(actual),len(gold))
                for a,b in zip(actual,gold):np.testing.assert_array_equal(a,b)
    def test_moving_mean_finite_and_nan_exact(self):
        rng=np.random.default_rng(15)
        for dtype in [np.float32,np.float64]:
            for order in ['C','F']:
                for invalid in [False,True]:
                    value=np.array(rng.normal(size=(13,97)),dtype=dtype,order=order)
                    if invalid:value[2,15]=np.nan;value[5,22]=np.nan
                    for window in [1,2,9,25,150]:
                        finite=~np.isnan(value);sums=np.concatenate((np.zeros((13,1)),np.cumsum(np.where(finite,value,0),axis=-1,dtype=np.float64)),axis=-1)
                        counts=np.concatenate((np.zeros((13,1),np.int64),np.cumsum(finite,axis=-1)),axis=-1);p=np.arange(97);lo=np.maximum(0,p-window//2);hi=np.minimum(97,p+(window-1)//2+1)
                        with np.errstate(invalid='ignore',divide='ignore'):expected=((sums[:,hi]-sums[:,lo])/(counts[:,hi]-counts[:,lo])).astype(dtype)
                        np.testing.assert_array_equal(expected,moving_mean(value,window))
    def test_movie_resize_matches_original_axis_kernel(self):
        rng=np.random.default_rng(19)
        for dtype in [np.float32,np.float64,np.uint16,np.bool_]:
            value=(rng.random((17,19,2,23))*200).astype(dtype)
            if dtype==np.bool_:value=rng.random(value.shape)>.5
            for order in ['C','F']:
                value=np.array(value,order=order)
                for scale in [.5,.25,.125,.63]:
                    # Same original kernel/layout, with only the 4-D fast path disabled.
                    import aqua2.se_common as common
                    from unittest.mock import patch
                    def legacy_axis(data,indices,weights,axis,integer):
                        moved=np.moveaxis(data,axis,0);flat=np.ascontiguousarray(moved.reshape(data.shape[axis],-1));out=_resize_axis(flat,indices,weights,integer)
                        return np.moveaxis(out.reshape((len(indices),)+moved.shape[1:]),0,axis)
                    with patch.object(common,'_resize_movie_axis',side_effect=legacy_axis):expected=my_resize(value,scale)
                    np.testing.assert_array_equal(expected,my_resize(value,scale))
    def test_streamed_movie_formats_roundtrip_and_atomic_rejection(self):
        # Cross the streaming boundary with a singleton Z and unusual last slab.
        value=np.arange(136*137*1*917,dtype=np.float32).reshape((136,137,1,917),order='F')
        with tempfile.TemporaryDirectory() as folder:
            for compression in ['gzip','lzf']:
                path=Path(folder)/(compression+'.h5');save_tree(path,{'movie':value},compression=compression)
                actual=load_tree(path,lazy=False)['movie'];self.assertTrue(actual.flags.f_contiguous);np.testing.assert_array_equal(value,actual)
                size=path.stat().st_size
                with self.assertRaises(ValueError):save_tree(path,{'movie':value},compression='unknown')
                self.assertEqual(size,path.stat().st_size)
