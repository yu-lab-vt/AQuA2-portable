import concurrent.futures
from pathlib import Path
import unittest
import numpy as np

from aqua2.io import read_mat
from aqua2.native import dtw_edge,bilco,midpoint_to_path
from aqua2.evt import delay_map_to_events,se_to_events_top
from aqua2.glo import remove_detected
from aqua2.pipeline import scalar_options

LOCAL=Path(__file__).resolve().parents[1]/"reference/local"


class SpatialUnitTests(unittest.TestCase):
    def test_native_bad_inputs(self):
        for distance in ([[np.nan]],[[1+1j]],[[1,-1]]):
            with self.assertRaises(ValueError):dtw_edge(distance)
        for edges in ([[0,2]],[[0,0]],[[0,.5]]):
            with self.assertRaises(ValueError):bilco([[0,1]],[[0,1]],edges,.2)

    def test_empty_events(self):
        temporal=dict(seLst=[],subEvtLst=[],seLabel=np.empty(0,int),majorInfo=[],opts={})
        result=se_to_events_top(np.zeros((4,4,1,5),np.float32),temporal)
        self.assertEqual(result["evt"],[]);self.assertEqual(result["riseLst"],[])
        self.assertFalse(result["datR"].any())


@unittest.skipUnless((LOCAL/"stage5-native.mat").exists(),"Generate R2024b native oracles")
class NativeMatlabTests(unittest.TestCase):
    def test_multiple_sizes_and_ties(self):
        for f in read_mat(LOCAL/"stage5-native.mat")["native"].ravel():
            cut=dtw_edge(f["distance"])
            np.testing.assert_array_equal(cut,f["cut"].ravel())
            np.testing.assert_array_equal(midpoint_to_path(cut,f["ref"].shape[1],f["tst"].shape[1]),f["path"]-1)
            cut,flow=bilco(f["ref"],f["tst"],f["gij"]-1,.3,f["initial"])
            np.testing.assert_array_equal(cut,f["bilcoCut"]);self.assertEqual(flow,f["flow"].item())

    def test_repeated_and_threaded_calls(self):
        f=read_mat(LOCAL/"stage5-native.mat")["native"].ravel()[-1]
        def run(_):
            cut,flow=bilco(f["ref"],f["tst"],f["gij"]-1,.3,f["initial"])
            return np.array_equal(cut,f["bilcoCut"]) and flow==f["flow"].item()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            self.assertTrue(all(executor.map(run,range(24))))


@unittest.skipUnless((LOCAL/"stage5-small.mat").exists(),"Generate R2024b source clustering/interpolation oracle")
class SpatialSmallMatlabTests(unittest.TestCase):
    def test_two_source_delay_watershed(self):
        f=read_mat(LOCAL/"stage5-small.mat")
        major=[{"ihw":m["ihw"].astype(int).ravel()-1} for m in f["major"].ravel()]
        actual=delay_map_to_events(f["delay"][...,None],major,scalar_options(f["options"]))
        np.testing.assert_array_equal(actual,f["sourceLabels"].ravel()-1)

    def test_remove_local_events_for_global_input(self):
        f=read_mat(LOCAL/"stage5-small.mat");events=[p.astype(int).ravel()-1 for p in f["events"].ravel()]
        np.testing.assert_array_equal(remove_detected(f["raw"],events),f["removed"])
