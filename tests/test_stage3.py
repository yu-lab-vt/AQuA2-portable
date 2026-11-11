import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from aqua2.act import ac_detect
from aqua2.io import read_mat
from aqua2.pre import baseline_linear_estimate,obtain_bias,truncated_kept_var,avg_dat_noise_est
from aqua2.util import impute_mov,normalize01

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "reference/local"


class Stage3UnitTests(unittest.TestCase):
    def test_baseline_blocks_preserve_curves_global_fallback_and_progress(self):
        import aqua2.pre as pre
        rng = np.random.default_rng(42)
        data = np.array(rng.normal(size=(9, 7, 1, 41)), dtype=np.float32, order="F")
        data[0, 0, 0, :] = np.nan
        data[2, 3, 0, 17] = np.nan
        data[-1, -1, 0, -1] = 100
        before = data.copy()
        expected = baseline_linear_estimate(data, 20, 8)
        events = []; blocks = []; original = pre.moving_mean
        def measured(values, window, axis):
            blocks.append(values.size)
            return original(values, window, axis)
        with patch.object(pre, "_BASELINE_BLOCK_SAMPLES", 41 * 5), patch.object(pre, "moving_mean", side_effect=measured):
            actual = baseline_linear_estimate(data, 20, 8, progress=lambda *args: events.append(args))
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(data, before)
        np.testing.assert_array_equal(actual[0, 0, 0], 100)
        self.assertGreater(len(blocks), 1)
        self.assertLessEqual(max(blocks), 41 * 5)
        self.assertEqual(events[0][1:], (0, 63))
        self.assertEqual(events[-1][1:], (63, 63))

    def test_preprocessing_progress_reaches_activity_without_changing_result(self):
        from aqua2.pipeline import preprocess
        from aqua2.config import read_preset
        from aqua2.compare import compare
        rng = np.random.default_rng(15)
        movie = rng.integers(10, 100, size=(14, 13, 1, 30), dtype=np.uint16)
        opts = read_preset(ROOT / "src/aqua2/data/parameters_for_batch.csv")
        opts.update(regMaskGap=0, thrARScl=1000)
        events = []
        before = movie.copy()
        actual = preprocess(movie, opts, progress=lambda *args: events.append(args))
        expected = preprocess(movie, opts)
        self.assertTrue(compare(expected, actual)["passed"])
        np.testing.assert_array_equal(movie, before)
        self.assertTrue(any("Estimating baseline" in item[0] and len(item) == 3 for item in events))
        self.assertTrue(any("Labeling active regions" in item[0] for item in events))
        self.assertIn("Active regions detected", events[-1][0])

    def test_normalize_and_imputation(self):
        np.testing.assert_array_equal(normalize01([np.nan,2,4,np.inf]),[0,0,1,0])
        np.testing.assert_array_equal(normalize01([5,5]),[0,0])
        np.testing.assert_array_equal(impute_mov(np.array([[np.nan,1,np.nan,3]])),[[1,1,1,3]])

    def test_truncation_endpoints_and_bias_asset(self):
        pars = truncated_kept_var(np.array([0,.01,.5]))
        self.assertEqual(pars[0],2)
        self.assertTrue(np.all(np.isfinite(pars)))
        self.assertEqual(obtain_bias(25,20),0)
        self.assertTrue(np.isfinite(obtain_bias(25,100)))

    def test_empty_activity(self):
        data = np.zeros((4,5,1,10),dtype=np.float32)
        opts = dict(thrARScl=3,maxdF1=0,maxSize=np.inf,circularityThr=0,compress=0,minDur=5,minSize=20)
        self.assertEqual(ac_detect(data,opts),[])

    @unittest.skipUnless((LOCAL / "stage3-internals.mat").exists(),"Generate stage 3 internals in R2024b")
    def test_baseline_multiple_segments_even_window_and_nan(self):
        fixture = read_mat(LOCAL / "stage3-internals.mat",["small","smallBaseline"])
        result = baseline_linear_estimate(fixture["small"],30,8)
        np.testing.assert_allclose(result,fixture["smallBaseline"],rtol=4*np.finfo(np.float32).eps,atol=0)

    @unittest.skipUnless((LOCAL / "stage3-internals.mat").exists(),"Generate R2024b noise oracle")
    def test_average_curve_noise(self):
        fixture = read_mat(LOCAL / "stage3-internals.mat",["avgCurve","noisePixels","avgNoise"])
        opts = read_mat(LOCAL / "preprocessing.mat",["optsOut"])["optsOut"]
        noise = avg_dat_noise_est(fixture["avgCurve"].ravel(),fixture["noisePixels"].astype(np.int64).ravel()-1,opts)
        np.testing.assert_allclose(noise,fixture["avgNoise"].item(),rtol=16*np.finfo(np.float32).eps,atol=0)
