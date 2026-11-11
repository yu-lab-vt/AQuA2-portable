import copy,queue,tempfile,unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
import tifffile
from aqua2.viewer_display import DisplaySettings,display_gray,fiji_auto_range
from aqua2.viewer_render import FrameRequest,FrameReader,prepare_frame,grayscale_u8
from aqua2.workflow import refresh_annotation_features,execute_job
from aqua2.compare import compare
from test_result_export import sample


class SharedDisplay(unittest.TestCase):
    def test_neutral_and_brightness_direction_for_integer_and_float(self):
        for dtype,top in ((np.uint8,255),(np.uint16,65535),(np.float32,1)):
            image=np.linspace(0,top,100).astype(dtype).reshape(10,10)
            default=DisplaySettings();neutral=display_gray(image,(0,top),default)
            np.testing.assert_array_equal(neutral,grayscale_u8(image,(0,top)))
            darker=display_gray(image,(0,top),replace(default,brightness=.25))
            brighter=display_gray(image,(0,top),replace(default,brightness=.75))
            self.assertTrue(np.all(darker<=neutral));self.assertTrue(np.all(brighter>=neutral))
            self.assertTrue(np.any(brighter>neutral));self.assertTrue(np.any(darker<neutral))

    def test_window_contrast_and_volume_only_transparency(self):
        image=np.array([[0.,.25,.5,.75,1.]])
        settings=DisplaySettings(minimum=.25,maximum=.75)
        np.testing.assert_array_equal(display_gray(image,(0,1),settings),[[0,0,128,255,255]])
        strong=display_gray(image,(0,1),DisplaySettings(contrast=.75))
        np.testing.assert_array_equal(strong,[[0,0,128,255,255]])
        transparent=DisplaySettings(transparency=1)
        np.testing.assert_array_equal(display_gray(image,(0,1),transparent),grayscale_u8(image,(0,1)))
        self.assertFalse(display_gray(image,(0,1),transparent,volume=True).any())
        for kwargs in (dict(minimum=1),dict(maximum=0),dict(brightness=float('nan')),dict(contrast=2)):
            with self.assertRaises(ValueError):DisplaySettings(**kwargs)

    def test_same_controls_apply_to_both_panels(self):
        project=sample();settings=DisplaySettings(brightness=.7,contrast=.6)
        request=FrameRequest(project,1,0,'channel1','local','Event','Original',(),(0,),False,1.,.5,True,True,False,0,display=settings)
        main,raw,_,_=prepare_frame(request)
        np.testing.assert_array_equal(main,raw)
        self.assertFalse(np.array_equal(main,prepare_frame(replace(request,display=DisplaySettings()))[0]))

    def test_fiji_histogram_ignores_dominant_peak_and_handles_degenerate_data(self):
        image=np.r_[np.zeros(60000,np.uint16),np.tile(np.arange(1,1201,dtype=np.uint16),5)]
        lo,hi=fiji_auto_range(image);self.assertGreater(lo,0);self.assertGreater(hi,1100)
        self.assertIsNone(fiji_auto_range(np.full(300,13,np.uint8)))
        self.assertIsNone(fiji_auto_range(np.array([np.nan,np.inf])))
        ramp=np.tile(np.arange(256,dtype=np.uint8),100)
        self.assertEqual(fiji_auto_range(ramp),(0,255))
        values=np.linspace(0,1,10000).astype(np.float32)
        self.assertEqual(fiji_auto_range(np.r_[values,np.nan]),fiji_auto_range(values))

    def test_export_matches_display_bytes_with_annotations_and_orientation(self):
        from aqua2.project import export_project
        project=sample();display=DisplaySettings(minimum=.05,maximum=.9,brightness=.7,contrast=.6,transparency=.15,overlay_brightness=.8)
        settings=dict(display=display.as_dict(),y_up=True,annotate=True,label_font=12,mode='Event',channel='channel1',scope='local',view='dF / σ',selected=[0],all_time=True)
        reader=FrameReader()
        try:
            with tempfile.TemporaryDirectory() as folder:
                out=export_project(project,folder,selection={'masked_movie':True},display_settings=settings)
                movie=tifffile.imread(out/'channel1_local_events_masked.tif')
                for t in range(3):
                    for z in range(2):
                        req=FrameRequest(project,t,z,'channel1','local','Event','dF / σ',(0,),(0,),True,1.,.5,True,False,False,0,annotate=True,label_font=12,display=display)
                        np.testing.assert_array_equal(movie[t,z],prepare_frame(req,reader)[0][::-1])
                cfu=tifffile.imread(out/'channel1_cfu_mean_masked.tif')
                self.assertEqual(cfu.shape,(2,12,13,3))
                for z in range(2):
                    req=replace(req,t=0,z=z,mode='CFU',view='Mean Projection',selected=())
                    np.testing.assert_array_equal(cfu[z],prepare_frame(req,reader)[0][::-1])
        finally:reader.close()


class AutomaticAnnotations(unittest.TestCase):
    def annotated(self):
        p=sample();p.data['stages']['detection']['opts'].update(spatialRes=.5,minShow1=.2,networkFeatures=False)
        p.data['annotations'].update(regions=[np.arange(50)],regions_dirty=True)
        return p

    def test_refresh_matches_original_kernels_and_preserves_curve_inputs(self):
        from aqua2.fea_region import region_features,region_mat_payload
        from aqua2.fea_network import event_network,network_mat_payload
        p=self.annotated();old=p.features();result=refresh_annotation_features(p,{'networkFeatures':True});fts=result.features()['fts']
        expected=region_mat_payload(region_features(p.events(),p.spatial()['datR'],[np.arange(50)],[],.5,.2,['1'],[]))
        self.assertTrue(compare(expected,fts['region'])['passed'])
        self.assertTrue(compare(network_mat_payload(event_network(p.events(),p.shape)),fts['networkAll'])['passed'])
        for key in ('curve','basic','loc'):self.assertIs(fts[key],old['fts'][key])
        for key in ('dMat','dffMat'):self.assertIs(result.features()[key],old[key])
        self.assertTrue(p.data['annotations']['regions_dirty']);self.assertFalse(result.data['annotations']['regions_dirty'])
        self.assertNotIn('region',old['fts']);self.assertTrue(result.opts['networkFeatures'])
        self.assertIs(refresh_annotation_features(result),result)

    def test_missing_reconstruction_allows_regions_but_preserves_old_project_on_landmark_failure(self):
        p=self.annotated()
        for scope in ('local','global'):p.spatial(scope=scope)['datR']=np.empty((0,0))
        self.assertFalse(refresh_annotation_features(p).data['annotations']['regions_dirty'])
        p.data['annotations']['landmarks']=[np.arange(3)]
        with self.assertRaisesRegex(ValueError,'datR'):refresh_annotation_features(p)
        self.assertTrue(p.data['annotations']['regions_dirty']);self.assertNotIn('region',p.features()['fts'])

    def test_network_toggle_and_cfu_dispatch_refresh_without_rebuilding_features(self):
        p=self.annotated();p.data['annotations']['regions_dirty']=False
        result=refresh_annotation_features(p,{'networkFeatures':True});self.assertIn('network',result.features()['fts'])
        result=refresh_annotation_features(result,{'networkFeatures':False});self.assertNotIn('network',result.features()['fts'])
        p.data['annotations']['regions_dirty']=True
        with patch('aqua2.workflow.recompute_cfu',side_effect=lambda project,*a,**kw:project) as cfu:
            answer=execute_job({'operation':'cfu','project':p,'feature_options':{'networkFeatures':True}})
        cfu.assert_called_once();self.assertFalse(cfu.call_args.args[0].data['annotations']['regions_dirty'])

    def test_preview_mask_import_is_cropped_once_in_preprocessing(self):
        from scipy.io import savemat
        from aqua2.workflow_steps import load_input,preprocess_project
        from test_workflow_steps import prepared
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'raw.tif';tifffile.imwrite(path,np.zeros((10,8,9),np.uint8),photometric='minisblack')
            masks=Path(folder)/'mask.mat';cell=np.empty((1,1),object)
            cell[0,0]=np.array([1,np.ravel_multi_index((2,3,0),(8,9,1),order='F')+1]);savemat(masks,{'regions':cell})
            p=load_input(path,{'regions':str(masks),'preview_mask_coordinates':{'regions':(8,9,1)}})
            fixture=prepared();opts=dict(fixture.opts);opts['regMaskGap']=1
            stage=dict(opts=opts,datOrg1=fixture.movie(kind='working'),dF1=fixture.movie(kind='dF'))
            with patch('aqua2.workflow_steps.preprocess',return_value=stage):result=preprocess_project(p)
            np.testing.assert_array_equal(result.data['annotations']['regions'][0],[np.ravel_multi_index((1,2,0),(6,7,1),order='F')])


class CompactDisplayControls(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        from aqua2.tk_runtime import initialize
        self.root=tk.Tk.__new__(tk.Tk);initialize(self.root);self.root.withdraw()
    def tearDown(self):self.root.destroy()

    def test_advanced_widgets_are_reduced_without_losing_loaded_scientific_options(self):
        from aqua2.viewer_step_forms import StepForms
        from tkinter import ttk
        forms=StepForms(ttk.Frame(self.root),Mock(),Mock())
        labels=lambda panel:[child.cget('text') for child in panel.winfo_children() if isinstance(child,ttk.Label)]
        self.assertEqual(labels(forms.advanced['detect']),['Temporal Threshold','Merge Delay','Seed Size Ratio'])
        self.assertFalse(any('North Vector' in label for label in labels(forms.advanced['preprocess'])))
        forms.values['northx'].set('.3');self.assertEqual(forms.get('preprocess')['northx'],.3)
        forms.values['gtwSmo'].set('3');self.assertEqual(forms.get('detect')['gtwSmo'],3)

    def test_import_one_kind_does_not_modify_other_mask_or_mutate_on_missing_field(self):
        from aqua2.viewer import Workbench
        from scipy.io import savemat
        p=sample();other=[np.arange(3)];p.data['annotations']['landmarks']=other
        window=SimpleNamespace(project=p,guarded=lambda:True,region_text=Mock(),request_render=Mock(),mask_revision=0)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'region.mat';cells=np.empty((1,1),object);cells[0,0]=np.arange(1,6);savemat(path,{'regions':cells})
            with patch('aqua2.viewer.filedialog.askopenfilename',return_value=str(path)):
                Workbench.import_masks(window,'regions')
                np.testing.assert_array_equal(p.data['annotations']['regions'][0],np.arange(5))
                self.assertIs(p.data['annotations']['landmarks'],other)
                with patch('aqua2.viewer.messagebox.showerror') as error:Workbench.import_masks(window,'landmarks')
                error.assert_called_once();self.assertIs(p.data['annotations']['landmarks'],other)

    def test_reset_and_crossed_window_remain_valid(self):
        import tkinter as tk
        from aqua2.viewer import Workbench
        window=SimpleNamespace(display_vars={key:tk.DoubleVar(self.root,value=value) for key,value in DisplaySettings().as_dict().items()},display_revision=0,auto_divisor=2500,request_render=Mock())
        window.display_vars['minimum'].set(1);Workbench.change_display(window,'minimum')
        settings=Workbench.display_settings(window);self.assertGreater(settings.maximum,settings.minimum)
        Workbench.reset_display(window);self.assertEqual(Workbench.display_settings(window),DisplaySettings());self.assertEqual(window.auto_divisor,0)

    def test_pending_auto_does_not_overwrite_manual_changes(self):
        import tkinter as tk
        from aqua2.viewer import Workbench
        p=sample();key=(id(p),'channel1','Original',0,0)
        var=lambda value:SimpleNamespace(get=lambda:value)
        for changed in (False,True):
            window=SimpleNamespace(project=p,channel=var('channel1'),view=var('Original'),frame=var(1),z=var(1),
                display_vars={key:tk.DoubleVar(self.root,value=value) for key,value in DisplaySettings().as_dict().items()},
                display_revision=int(changed),auto_pending=True,auto_button=Mock(),status=Mock(),request_render=Mock(),
                ui_events=queue.Queue(),job=SimpleNamespace(events=queue.Queue()),after=Mock(),poll=Mock(),busy=False,loading=False)
            window.ui_events.put(('auto_display',p,key,0,(.1,.8),(0,1),None));Workbench.poll(window)
            self.assertFalse(window.auto_pending)
            self.assertEqual(window.display_vars['minimum'].get(),0 if changed else .1)
            self.assertEqual(window.request_render.call_count,0 if changed else 1)

    def test_event_curves_always_use_uncleaned_data(self):
        from aqua2.viewer import Workbench
        var=lambda value:SimpleNamespace(get=lambda:value)
        window=SimpleNamespace(curve_ax=Mock(),project=sample(),mode=var('Event'),channel=var('channel1'),scope=var('local'),
            selected=[0],curve_kind=var('dF/F'),cleaned=var(True),frame=var(1),curve_figure=Mock(),curve_blit=Mock(),curve_canvas=Mock())
        with patch('aqua2.viewer.curve_series',return_value=[]) as curves:Workbench.plot_curves(window)
        self.assertIs(curves.call_args.args[-1],False)


if __name__=='__main__':unittest.main()
