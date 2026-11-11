import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from aqua2.io import read_mat,write_mat
from aqua2.compare import compare,Tolerance
from aqua2.pipeline import scalar_options
from aqua2.fea import get_features,features_mat_payload,curve_statistics,indexed_spatial_mean,curve_baseline
from aqua2.fea_geometry import boundaries,close_mask,perimeter,bicubic_resize
from aqua2.fea_propagation import propagation_for_event
from aqua2.fea_region import _chessboard_geodesic,mask_pixels
from aqua2.features import compute_features
from aqua2.export import export_features

LOCAL=Path(__file__).resolve().parents[1]/"reference/local"


class FeatureUnitTests(unittest.TestCase):
    def test_mask_coordinate_validation(self):
        for value in ([np.array([-1])],[np.ones((2,3),bool)],[np.array([.5])]):
            with self.assertRaises(ValueError):mask_pixels(value,(4,4,1))
        np.testing.assert_array_equal(mask_pixels(np.eye(4,dtype=bool),(4,4,1))[0],[0,5,10,15])

    def test_default_landmark_geodesic_is_chessboard(self):
        mask=np.ones((5,5),bool);seed=np.zeros_like(mask);seed[0,0]=True
        distance=_chessboard_geodesic(mask,seed);self.assertEqual(distance[4,4],4)
        mask[:,2]=False;distance=_chessboard_geodesic(mask,seed)
        self.assertTrue(np.isnan(distance[:,2]).all());self.assertTrue(np.isinf(distance[:,3:]).all())


@unittest.skipUnless((LOCAL/"stage6-primitives.mat").exists(),"Generate R2024b feature primitives")
class FeaturePrimitiveMatlabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.f=read_mat(LOCAL/"stage6-primitives.mat")

    def test_exact_geometry_holes_and_disconnected_regions(self):
        for fixture in self.f["geometry"].ravel():
            np.testing.assert_array_equal(close_mask(fixture["mask"]),fixture["closed"])
            self.assertEqual(perimeter(fixture["mask"]),fixture["perimeter"].item())
            for connectivity in (4,8):
                actual=boundaries(fixture["mask"],connectivity);expected=fixture["boundary"+str(connectivity)].ravel()
                self.assertEqual(len(actual),len(expected))
                for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b-1)

    def test_standard_bicubic_is_not_custom_block_resize(self):
        actual=bicubic_resize(self.f["resizeProbe"],self.f["resizeScale"].item())
        np.testing.assert_allclose(actual,self.f["resized"],atol=8*np.finfo(float).eps,rtol=8*np.finfo(float).eps)

    def test_curve_threshold_statistics(self):
        for curve,expected in zip(self.f["curves"].ravel(),self.f["curveStats"].ravel()):
            actual=curve_statistics(curve.ravel(),.5,True)
            for key in ("rise19","fall91","width55","width11"):self.assertEqual(actual[key],expected[key].item())
            np.testing.assert_array_equal(actual["pp"],expected["pp"]-1)
            self.assertAlmostEqual(actual["riseTime"],expected["riseTime"].item()-1,places=14)
            self.assertTrue(np.isnan(actual["decayTau"]))


@unittest.skipUnless((LOCAL/"stage6-fit.mat").exists(),"Generate R2024b exp1 fit cases")
class ExponentialModelMatlabTests(unittest.TestCase):
    def test_exp1_matches_reference_objective_tolerance(self):
        fixtures=read_mat(LOCAL/"stage6-fit.mat");primitive=read_mat(LOCAL/"stage6-primitives.mat")
        tolerance=fixtures["settings"]["TolFun"].item()
        for curve,case in zip(primitive["curves"].ravel(),fixtures["cases"].ravel()):
            tau=curve_statistics(curve.ravel(),.5,False)["decayTau"];b=-.5/tau
            x,y=case["x"].ravel(),case["y"].ravel();e=np.exp(b*x);a=e@y/(e@e)
            actual=np.sum((a*e-y)**2);coef=case["coefficients"].ravel();gold=np.sum((coef[0]*np.exp(coef[1]*x)-y)**2)
            self.assertLessEqual(abs(actual-gold),tolerance)


@unittest.skipUnless((LOCAL/"stage6-small.mat").exists(),"Generate R2024b small feature fixtures")
class SmallFeatureMatlabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.f=read_mat(LOCAL/"stage6-small.mat")

    def test_small_spatial_mean_and_aligned_curve_exact(self):
        f=self.f;events=[p.astype(int).ravel()-1 for p in f["events"].ravel()]
        result,_=get_features(f["data"],events,f["options"])
        with tempfile.TemporaryDirectory(dir=LOCAL) as directory:
            path=Path(directory)/"features.mat";write_mat(path,features_mat_payload(result));check=compare(f["features"],read_mat(path))
            self.assertTrue(check["passed"],check["issues"])

    def test_multiple_propagation_thresholds_and_rotated_north_exact(self):
        f=self.f;actual=propagation_for_event(f["volume"],.5,scalar_options(f["propOptions"]),np.array([.6,.8]))
        for key,value in actual.items():
            expected=f["propagation"][key]
            expected=expected[0,0] if expected.dtype==object else expected.ravel()
            np.testing.assert_array_equal(value,expected)

    def test_logical_resize_rounds_each_axis(self):
        f=self.f;np.testing.assert_array_equal(bicubic_resize(f["logicalProbe"],.63),f["logicalResized"])

    def test_csv_and_curve_npz_roundtrip(self):
        f=self.f;events=[p.astype(int).ravel()-1 for p in f["events"].ravel()]
        options=scalar_options(f["options"]);spatial={"evt":events,"datR":np.zeros(f["data"].shape,np.uint8),"riseLst":[]}
        result=compute_features(f["data"],spatial,options,channel=1)
        with tempfile.TemporaryDirectory(dir=LOCAL) as directory:
            path=export_features(directory,result)
            with path.open(encoding="utf-8-sig",newline="") as stream:rows=list(csv.reader(stream))
            self.assertEqual(rows[0],["Feature","Event 1"]);self.assertIn(["Channel","1.0"],rows)
            with np.load(Path(directory)/"events_curves.npz",allow_pickle=False) as curves:
                for key in ("dMat","dffMat","dffAlignedMat"):np.testing.assert_array_equal(curves[key],result[key])
            metadata=json.loads((Path(directory)/"events_metadata.json").read_text(encoding="utf-8"));self.assertEqual(metadata["csv_event_and_frame_origin"],1)

    def test_no_events_has_stable_empty_schema(self):
        f=self.f;shape=f["data"].shape;spatial={"evt":[],"datR":np.zeros(shape,np.uint8),"riseLst":[]}
        result=compute_features(f["data"],spatial,f["options"])
        self.assertEqual(result["dMat"].shape,(0,shape[3],2));self.assertEqual(result["dffAlignedMat"].shape,(0,111))
        with tempfile.TemporaryDirectory(dir=LOCAL) as directory:self.assertTrue(export_features(directory,result).is_file())


@unittest.skipUnless((LOCAL/"stage6-moving.mat").exists(),"Generate R2024b curve mean/moving reference")
class CurvePrecisionMatlabTests(unittest.TestCase):
    def test_indexed_mean_4096_blocks_on_real_large_regions(self):
        f=read_mat(LOCAL/"preprocessing.mat");gold=read_mat(LOCAL/"pipeline.mat");trace=read_mat(LOCAL/"stage6-numeric.mat")["trace"].ravel()
        vectors=f["roi"].reshape((-1,f["roi"].shape[3]),order="F")
        for pixels,expected in zip(gold["spatial"]["evt"].ravel(),trace):
            footprint=np.unique((pixels.astype(int).ravel()-1)%vectors.shape[0])
            np.testing.assert_array_equal(indexed_spatial_mean(vectors[footprint,:]),expected["meanRaw"].ravel())
