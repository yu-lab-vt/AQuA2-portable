import unittest
from pathlib import Path

import numpy as np

from aqua2.io import read_mat
from aqua2.se import se_detection,temporal_mat_payload,temporal_from_stage3
from aqua2.se_common import my_resize,impose_minima,marker_watershed,mean_spatial
from aqua2.se_majority import dtw_path,create_merging_info,merge_super_events
from aqua2.se_partition import grow_watershed_result
from aqua2.se_refine import split_events
from aqua2.se_stats import order_statistics

LOCAL = Path(__file__).resolve().parents[1] / "reference/local"


def cells(value):
    return [np.asarray(v).astype(np.int64).ravel()-1 for v in value.ravel(order="F")]


def majors(value):
    return [{"ihw":m["ihw"].astype(np.int64).ravel()-1,
             "TW":m["TW"].astype(np.int64).ravel()-1,
             "delays":m["delays"].ravel(),"curve":m["curve"].ravel()}
            for m in value.ravel(order="F")]


class TemporalUnitTests(unittest.TestCase):
    def test_resize_partial_block_integer_and_logical(self):
        values=np.arange(1,14,dtype=np.uint16)[:,None]
        np.testing.assert_array_equal(my_resize(values,(.25,1)).ravel(),[3,7,11,13])
        mask=np.ones((4,4),bool);mask[0,0]=False
        np.testing.assert_array_equal(my_resize(mask,.5),[[False,True],[True,True]])

    def test_dtw_single_precision_and_ties(self):
        x=np.array([0,1,1,0],np.float32);y=np.array([0,1,0],np.float32)
        a,b=dtw_path(x,y)
        np.testing.assert_array_equal(a,[0,1,2,3])
        np.testing.assert_array_equal(b,[0,1,1,2])

    def test_empty_activity_and_bypass(self):
        t=np.arange(24,dtype=np.float32)
        raw=np.broadcast_to(.2+.01*np.sin(t),(8,8,1,24)).copy(order="F")
        options=dict(maxValueDat1=1,BitDepth=16,maxdF1=0,thrARScl=3,
            tempVarOrg=np.ones((8,8,1),np.float32),correctPars=np.ones((8,8,1))*2,
            minSize=3,minDur=3,seedSzRatio=.01,TPatch=20,sigThr=3,spaMergeDist=0,maxDelay=.6)
        result=se_detection(np.zeros_like(raw),raw,[],options)
        self.assertEqual(result["seLst"],[])
        self.assertEqual(temporal_mat_payload(result)["seLabel"].shape,(1,0))
        bypass=temporal_from_stage3(dict(dF1=raw,datOrg1=raw,activeRegions1=[],opts={**options,"needTemp":False}))
        self.assertEqual(bypass["channel1"]["subEvtLst"],[])


@unittest.skipUnless((LOCAL/"stage4-primitives.mat").exists(),"Generate R2024b stage-4 primitives")
class TemporalMatlabPrimitiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f=read_mat(LOCAL/"stage4-primitives.mat")

    def test_exact_custom_resize(self):
        f=self.f
        for name,source,scale in (("resize2",f["probe"],.5),("resize4",f["probe"],.25),
            ("intResize",f["intProbe"],.25),("rowResize",f["row"],1/3),("colResize",f["row"].T,1/3)):
            with self.subTest(name=name):np.testing.assert_array_equal(my_resize(source,scale),f[name])

    def test_order_statistics_original_table(self):
        f=self.f;mu,sigma=order_statistics(f["fg"],f["bg"],f["omit"])
        np.testing.assert_allclose([mu,sigma],[f["orderMu"].item(),f["orderSigma"].item()],rtol=8*np.finfo(float).eps,atol=0)

    def test_exact_watershed_and_single_precision_minima(self):
        f=self.f
        np.testing.assert_array_equal(impose_minima(f["surface"],f["markers"]),f["imposed"])
        np.testing.assert_array_equal(marker_watershed(f["surface"],f["markers"]),f["water"])

    def test_dtw_matlab_paths(self):
        for x,y,px,py in (([0,.5,1,0],[0,.4,.8,1,0],"dtwX","dtwY"),([0,1,1,0],[0,1,0],"tieX","tieY")):
            a,b=dtw_path(np.array(x,float),np.array(y,float))
            np.testing.assert_array_equal(a,self.f[px].ravel()-1)
            np.testing.assert_array_equal(b,self.f[py].ravel()-1)


@unittest.skipUnless((LOCAL/"stage4-split-small.mat").exists(),"Generate R2024b optional-branch fixtures")
class TemporalOptionalTests(unittest.TestCase):
    def test_two_real_peaks_split_including_majority(self):
        f=read_mat(LOCAL/"stage4-split-small.mat")
        from aqua2.pipeline import scalar_options
        events,major,seeds=split_events(f["df"],f["raw"],cells(f["event"]),cells(f["seeds"]),majors(f["major"]),scalar_options(f["options"]))
        self.assertEqual(len(events),2)
        for actual,expected in ((events,cells(f["splitEvents"])),(seeds,cells(f["splitSeeds"]))):
            for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
        for a,b in zip(major,majors(f["splitMajor"])):
            for key in a:np.testing.assert_array_equal(a[key],b[key])

    def test_correlated_footprint_actually_grows(self):
        f=read_mat(LOCAL/"stage4-grow-small.mat")
        from aqua2.pipeline import scalar_options
        events,regions=grow_watershed_result(cells(f["growEvents"]),majors(f["growMajor"]),f["growData"],scalar_options(f["options"]))
        self.assertGreater(len(events[0]),len(cells(f["growEvents"])[0]))
        for actual,expected in ((events,cells(f["grownEvents"])),(regions,cells(f["grownRegions"]))):
            self.assertEqual(len(actual),len(expected))
            for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)


@unittest.skipUnless((LOCAL/"stage4-boundaries.mat").exists(),"Generate R2024b threshold-boundary fixture")
class TemporalBoundaryTests(unittest.TestCase):
    def test_native_single_precision_spatial_mean(self):
        f=read_mat(LOCAL/"stage4-boundaries.mat")
        np.testing.assert_array_equal(mean_spatial(f["meanProbe"]),f["meanResult"].ravel())
        for a,b in zip(f["meanRows"].ravel(),f["meanByRows"].ravel()):
            np.testing.assert_array_equal(mean_spatial(a),b.ravel())

    def test_merge_at_threshold_and_one_ulp_below(self):
        from aqua2.pipeline import scalar_options
        f=read_mat(LOCAL/"stage4-boundaries.mat")
        events,major,regions=cells(f["events"]),majors(f["major"]),cells(f["regions"])
        shape=tuple(f["shape"].astype(int).ravel());opts=scalar_options(f["options"])
        for prefix,threshold in (("at",f["distance"].item()),("below",opts["maxDelay"])):
            with self.subTest(prefix=prefix):
                options={**opts,"maxDelay":threshold}
                info=create_merging_info(events,major,regions,shape,options)
                actual,labels,_=merge_super_events(events,major,info,shape,regions,options)
                self.assertEqual(len(actual),len(cells(f[prefix+"Events"])))
                for a,b in zip(actual,cells(f[prefix+"Events"])):np.testing.assert_array_equal(a,b)
                np.testing.assert_array_equal(labels,f[prefix+"Labels"].ravel()-1)

if __name__=="__main__":unittest.main()
