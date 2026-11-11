import unittest
import tempfile
from pathlib import Path
import numpy as np
from aqua2.io import read_mat,write_mat
from aqua2.compare import compare,Tolerance
from aqua2.cfu import (frequency_statistics,rising_time,normalize_info,info_mat_payload,
    membership_mat_payload,cfu_mat_payload,detect_channel,default_options,run_cfu,read_cfu_result)
from aqua2.cfu_spatial import prepare_events,minimum_measure,materialize_memberships,membership_time_windows,hierarchical_groups
from aqua2.cfu_merge import merge_similar,complete_linkage_groups
from aqua2.cfu_dependency import dependency,dependency_simple,binomial_tail,group_cfus
from aqua2.cfu_parity import compare_cfu

LOCAL=Path(__file__).resolve().parents[1]/'reference/local'
DOUBLE=Tolerance(atol=128*np.finfo(float).eps,rtol=1024*np.finfo(float).eps)


class CFUUnitTests(unittest.TestCase):
    def test_strict_overlap_threshold_and_label_order(self):
        edges=np.array([[0,1,.5],[1,2,.25]])
        groups=hierarchical_groups(edges,3,.5)
        np.testing.assert_array_equal(groups[0],[1,2]);np.testing.assert_array_equal(groups[1],[0])
        self.assertEqual(len(hierarchical_groups(np.empty((0,3)),0,.5)),0)

    def test_complete_linkage_rejects_chain(self):
        edges=np.array([[0,1,0],[1,0,1],[0,1,0]],bool);scores=np.array([[0,.96,0],[.96,0,.95],[0,.95,0]])
        self.assertEqual(complete_linkage_groups(edges,scores),[[0,1],[2]])

    def test_empty_rise_window_is_unmarked(self):
        self.assertIsNone(rising_time(np.arange(5.,0,-1),0,3,np.ones(5,bool)))

    def test_legacy_records_and_empty_schema(self):
        cells=np.empty((1,8),object)
        for j in range(8):cells[0,j]=np.empty((0,0))
        cells[0,1]=np.array([[2.,4.]])
        info=normalize_info(cells);self.assertEqual(info[0]['id'],0);np.testing.assert_array_equal(info[0]['eventIds'],[1,3])
        self.assertEqual(len(info[0]),14);self.assertEqual(info_mat_payload([]).shape,(0,1))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'legacy.mat'
            write_mat(path,dict(cfuInfo1=cells,cfuInfo2=cells,cfuRelation=np.empty((0,0)),cfuGroupInfo=np.empty((0,4),object),cfuOpts=default_options()))
            self.assertEqual(read_cfu_result(path)['channel_count'],2)

    def test_frequency_selection_and_percentile(self):
        self.assertEqual(frequency_statistics([1],.4)['method'],'N/A')
        self.assertEqual(frequency_statistics([1,3,5,7,9,11],.5)['peakFreq80'],1)
        self.assertEqual(frequency_statistics([0,1,2,3,100],1)['method'],'Med')


@unittest.skipUnless((LOCAL/'stage7-primitives.mat').exists(),'Generate R2024b stage-7 primitives')
class CFUMatlabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.f=read_mat(LOCAL/'stage7-primitives.mat')

    def test_shared_spatial_basins_and_membership_windows(self):
        g=self.f['spatial'];shape=tuple(g['shape'].astype(int).ravel());events=[p.astype(np.int64).ravel()-1 for p in g['events'].ravel()]
        pre=prepare_events(events,shape);cfus=materialize_memberships(minimum_measure(pre,shape,.4,3),events,shape)
        self.assertEqual(len(cfus),len(g['regions']));self.assertGreater(len(np.intersect1d(cfus[0]['events'],cfus[1]['events'])),0)
        for c,r,ids,m in zip(cfus,g['regions'].ravel(),g['lists'].ravel(),g['memberships'].ravel()):
            np.testing.assert_array_equal(c['region'],r);np.testing.assert_array_equal(c['events'],ids.ravel()-1)
            check=compare(m,membership_mat_payload(c['memberships']));self.assertTrue(check['passed'],check['issues'])
        own,other,overlap=membership_time_windows(cfus,shape)
        np.testing.assert_array_equal(own,g['own']);np.testing.assert_array_equal(other,g['other'])
        for a,b in zip(overlap,g['overlapping'].ravel()):np.testing.assert_array_equal(a,b.ravel()-1)

    def test_full_small_single_dual_and_empty(self):
        g=self.f['spatial'];movie=g['movie'];events=[p.ravel().astype(np.int64)-1 for p in g['events'].ravel()];peaks=g['peaks'].ravel()-1
        for name,channels in (('smallSingle',1),('smallDual',2),('smallDualEmpty',2),('smallEmpty',1)):
            gold=self.f[name];actual=run_cfu([movie]*channels,[events]*channels,self.f['smallOpts'],[peaks]*channels,gold['cfuOpts'])
            payload=cfu_mat_payload(actual);check=compare_cfu(gold,payload,5,40)
            self.assertTrue(check['passed'],check)

    def test_merge_maps_memberships_and_diagnostics(self):
        for g in self.f['mergeCases'].ravel():
            cfus=[]
            records=[]
            for i,(r,e,p,m) in enumerate(zip(g['regions'].ravel(),g['lists'].ravel(),g['parents'].ravel(),g['memberships'].ravel())):
                # Reuse the public schema adapter for one-based memberships.
                raw=dict(id=np.array([[i+1]]),eventIds=e,weightMap=r,memberships=m)
                member=normalize_info(raw)[0]['memberships'];cfus.append(dict(region=r,events=e.ravel().astype(int)-1,parent=int(p)-1,memberships=member))
            p={k:np.asarray(v).item() for k,v in g['parameters'].items() if k!='ProgressCallback'}
            output,changed,groups,diagnostics=merge_similar(cfus,g['dff'],p)
            self.assertEqual(bool(changed),bool(g['didMerge'].item()));self.assertEqual(diagnostics,{k:int(v.item()) for k,v in g['diagnostics'].items()})
            np.testing.assert_array_equal(changed,g['changed'].ravel()-1)
            for c,r,e,m in zip(output,g['outRegions'].ravel(),g['outLists'].ravel(),g['outMembers'].ravel()):
                np.testing.assert_array_equal(c['region'],r);np.testing.assert_array_equal(c['events'],e.ravel()-1)
                check=compare(m,membership_mat_payload(c['memberships']));self.assertTrue(check['passed'],check['issues'])

    def test_dependency_split_windows_shift_and_ties(self):
        for g in self.f['dependencies'].ravel():
            p,d,distribution=dependency(g['first'],g['second'],g['shift'].item(),g['distances'].ravel())
            self.assertEqual(d,g['d'].item());np.testing.assert_array_equal(distribution,g['distribution'])
            np.testing.assert_allclose(p,g['p'].item(),atol=DOUBLE.atol,rtol=DOUBLE.rtol)
            simple=dependency_simple(g['first'].ravel(),g['second'].ravel(),g['distances'].ravel())
            np.testing.assert_allclose(simple,g['simple'].ravel(),atol=DOUBLE.atol,rtol=DOUBLE.rtol)
            np.testing.assert_allclose(dependency_simple(g['first'],g['second'],g['distances'].ravel()),g['simple'].ravel(),atol=DOUBLE.atol,rtol=DOUBLE.rtol)

    def test_saddlepoint_branches(self):
        for g in self.f['tails'].ravel():np.testing.assert_allclose(binomial_tail(g['p'].ravel(),g['m'].item()),g['tail'].item(),atol=DOUBLE.atol,rtol=DOUBLE.rtol)

    def test_group_delay_direction_and_order(self):
        g=self.f['groupCase'];relation=g['relation'].copy();relation[:,:2]-=1
        groups=group_cfus([{}]*5,[],relation,1e-5,2);self.assertEqual(len(groups),len(g['groups']))
        for actual,row in zip(groups,g['groups']):
            np.testing.assert_array_equal(actual['cfuIds'],row[1].ravel()-1);np.testing.assert_array_equal(actual['delays'],row[2].ravel());np.testing.assert_array_equal(actual['pvalues'],row[3].ravel())


@unittest.skipUnless((LOCAL/'stage7-input.mat').exists(),'Register the user-selected v2 CFU result')
class CFUHistoricalReadTests(unittest.TestCase):
    def test_v2_named_info_read_write_preserves_scientific_fields(self):
        f=read_mat(LOCAL/'stage7-input.mat');path=str(f['source'].item()).replace('_AQuA2.mat','_AQuA2_v2_res_cfu.mat')
        expected=read_mat(path,['cfuInfo1'])['cfuInfo1'];restored=read_cfu_result(path)
        actual=cfu_mat_payload(restored)['cfuInfo1'];check=compare(expected,actual)
        self.assertTrue(check['passed'],check['issues'])
