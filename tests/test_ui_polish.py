import tempfile,unittest,copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
import tifffile
from aqua2.config import read_preset,read_parameter_bundle
from aqua2.input_formats import read_tiff_channels
from aqua2.workflow_steps import load_input
from aqua2.pipeline import prepare_movie
from aqua2.viewer_steps import PREPROCESS_FIELDS,DETECTION_FIELDS,FIXED_DETECTION_FLAGS
from aqua2.viewer_render import annotate_labels

PARAMETERS=Path(__file__).parents[1]/'src/aqua2/data/parameters_for_batch.csv'


class FormatAndParameters(unittest.TestCase):
    def test_parameter_bundle_keeps_original_options_and_three_cfu_defaults(self):
        for preset in (1,2,3):
            options,settings=read_parameter_bundle(PARAMETERS,preset)
            self.assertEqual(options,read_preset(PARAMETERS,preset))
            self.assertNotIn('overlapThr',options)
            self.assertEqual(settings['cfuDetect'],dict(overlapThr1=.5,overlapThr2=.5,minNumEvt1=3,minNumEvt2=3,minimumCurveCorrelation=.925,minimumCurveCorrelation2=.925))
        self.assertTrue(PARAMETERS.is_file())

    def test_two_channel_tiff_preview_and_preprocessing_use_identical_axis_order(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'channels.tif';data=np.arange(6*2*8*9,dtype=np.uint16).reshape(6,2,8,9)
            tifffile.imwrite(path,data,photometric='minisblack',metadata={'axes':'TCYX'})
            p=load_input(path,{'auto_channels':True});self.assertEqual(p.channels,['channel1','channel2'])
            expected=[data[:,c].transpose(1,2,0)[:,:,None,:] for c in (0,1)]
            for c in (0,1):np.testing.assert_array_equal(p.movie('channel'+str(c+1)),expected[c])
            options=read_preset(PARAMETERS);options['regMaskGap']=0
            one,two,opts=prepare_movie(path,options,auto_channels=True)
            baseline=prepare_movie(expected[0],options,expected[1])
            np.testing.assert_array_equal(one,baseline[0]);np.testing.assert_array_equal(two,baseline[1]);self.assertFalse(opts['singleChannel'])

    def test_explicit_volume_channels_and_two_frame_mono_do_not_confuse_axes(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'volume.tif';data=np.arange(4*3*2*8*9,dtype=np.uint16).reshape(4,3,2,8,9)
            tifffile.imwrite(path,data,photometric='minisblack',metadata={'axes':'TZCYX'})
            channels=read_tiff_channels(path)
            np.testing.assert_array_equal(channels[1],data[:,:,1].transpose(2,3,1,0))
            self.assertEqual(channels[0].shape,(8,9,3,4))
            path=Path(folder)/'two-frames.tif';data=data[:2,0,0]
            tifffile.imwrite(path,data,photometric='minisblack',metadata={'axes':'TYX'})
            self.assertEqual(len(read_tiff_channels(path)),1);self.assertEqual(read_tiff_channels(path)[0].shape,(8,9,1,2))

    def test_raw_mat_size_two_channel_and_saved_result_are_detected(self):
        from scipy.io import savemat
        from aqua2.input_formats import read_mat_channels,is_saved_input
        import h5py
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'raw.mat';data=np.arange(8*9*6*2,dtype=np.uint16).reshape(8,9,6,2)
            savemat(path,{'movie':data});channels=read_mat_channels(path)
            self.assertFalse(is_saved_input(path));self.assertEqual(len(channels),2)
            np.testing.assert_array_equal(channels[1],data[:,:,:,1][:,:,None,:])
            project=load_input(path,{'auto_channels':True});self.assertEqual(project.channels,['channel1','channel2'])
            opts=read_preset(PARAMETERS);opts['regMaskGap']=0
            one,two,actual=prepare_movie(path,opts,auto_channels=True);self.assertEqual(actual['BitDepth'],-1)
            path=Path(folder)/'saved.mat'
            with h5py.File(path,'w') as root:root.create_dataset('res/datOrg1',data=np.zeros((4,1,8,9),np.uint16));root.create_group('res/opts')
            self.assertTrue(is_saved_input(path))

    def test_rgb_overlay_rejected_instead_of_becoming_channel_data(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'rgb.tif';tifffile.imwrite(path,np.zeros((10,11,3),np.uint8),photometric='rgb')
            with self.assertRaisesRegex(ValueError,'RGB'):read_tiff_channels(path)

    def test_unadjusted_side_stays_fixed_when_right_gamma_changes(self):
        from aqua2.project import Project,new_project
        from aqua2.viewer_render import FrameRequest,prepare_frame
        from dataclasses import replace
        data=new_project('raw comparison');movie=np.arange(30*40,dtype=np.uint16).reshape(30,40,1,1)
        data.update(opts=dict(sz=movie.shape,frameRate=.5,minValueDat1=0.,maxValueDat1=1200.,singleChannel=True),movies={'channel1':{'stored':movie}})
        p=Project(data);req=FrameRequest(p,0,0,'channel1','local','Event','Original',(),(),False,1.,.5,False,True,False,0,annotate=True)
        first=prepare_frame(req);second=prepare_frame(replace(req,gamma=2.,auto_contrast=True))
        np.testing.assert_array_equal(first[1],second[1]);self.assertFalse(np.array_equal(first[0],second[0]))

    def test_overlapping_cfus_keep_their_own_id_and_selected_boundary(self):
        from aqua2.project import Project,new_project
        p=SimpleNamespace(_cache={},cfus=lambda channel:[{'weightMap':np.pad(np.ones((10,10)),((5,35),(5,35)))},{'weightMap':np.ones((50,50))}])
        hits=np.full((50,50),2,np.int32);p._cache['display_cfu_layer']=(('channel1',0,(0,1)),hits,np.ones_like(hits))
        actual=annotate_labels(np.zeros((50,50,3),np.uint8),hits,p,True,selected=(0,))
        self.assertEqual([item[0] for item in p._cache['display_id_layout'][1]],[1,2])
        self.assertTrue(np.any(np.all(actual==[140,217,255],axis=-1)))

    def test_numbers_and_light_blue_cfu_boundary_do_not_change_hits(self):
        hits=np.zeros((100,120),np.int32);hits[20:80,20:100]=12;original=hits.copy()
        image=np.full((100,120,3),30,np.uint8);project=SimpleNamespace(_cache={})
        actual=annotate_labels(image.copy(),hits,project,True,selected=(11,))
        np.testing.assert_array_equal(actual[20,25],[140,217,255]);np.testing.assert_array_equal(actual[0,0],[30]*3)
        self.assertTrue(np.any(actual[40:65,40:80]!=30));np.testing.assert_array_equal(hits,original)
        np.testing.assert_array_equal(actual,annotate_labels(image.copy(),hits,project,True,selected=(11,)))


class CompactControls(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        from aqua2.tk_runtime import initialize
        self.root=tk.Tk.__new__(tk.Tk);initialize(self.root);self.root.withdraw()
    def tearDown(self):self.root.destroy()

    def test_stage_forms_start_at_preset1_and_advanced_sections_are_collapsed(self):
        from aqua2.viewer_step_forms import StepForms
        from tkinter import ttk
        forms=StepForms(ttk.Frame(self.root),Mock(),Mock());options=read_preset(PARAMETERS)
        for key in forms.values:self.assertEqual(float(forms.values[key].get()),options[key])
        self.assertEqual(set(forms.flags),{'detectGlo','propMetric','networkFeatures'})
        self.assertFalse(forms.advanced['preprocess'].grid_info());self.assertFalse(forms.advanced['detect'].grid_info())
        for key,value in FIXED_DETECTION_FLAGS.items():self.assertEqual(forms.get('detect')[key],value)
        options,settings=read_parameter_bundle(PARAMETERS,2);settings['cfuDetect']['minimumCurveCorrelation']=.97
        forms.apply_parameters(options,settings)
        self.assertEqual(forms.values['thrARScl'].get(),'2.5');self.assertEqual(forms.cfu.values['minimumCurveCorrelation'].get(),'0.8')
        self.assertEqual(forms.get('cfu')['cfuDetect']['minimumCurveCorrelation'],.97)

    def test_parameter_dialog_loads_all_stages_with_one_action(self):
        from aqua2.viewer_dialogs import ParameterDialog
        calls=[];dialog=ParameterDialog(self.root,PARAMETERS,3,lambda *args:calls.append(args));dialog.withdraw();dialog.submit()
        self.assertEqual(len(calls),1);self.assertEqual(calls[0][2]['thrARScl'],2.)
        self.assertEqual(calls[0][3]['cfuDetect']['minimumCurveCorrelation'],.925)

    def test_layout_raw_is_physically_left_and_processed_right(self):
        from aqua2.viewer import Workbench
        from aqua2.viewer_render import BlitCanvas
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        figure=Figure();canvas=FigureCanvasAgg(figure)
        window=SimpleNamespace(image_figure=figure,image_canvas=canvas,image_blit=BlitCanvas(canvas),display_key=None,draw_masks=lambda ax,z:[])
        request=SimpleNamespace(project=SimpleNamespace(),side=True,y_up=False,view='Overlay',z=0,mask_revision=0,overlay=True)
        processed=np.full((20,30,3),100,np.uint8);raw=np.full_like(processed,20)
        Workbench.draw_image(window,request,processed,raw,None,np.zeros((20,30),int))
        self.assertGreater(window.image_axes[0].get_position().x0,window.image_axes[1].get_position().x0)
        np.testing.assert_array_equal(window.image_artists[0].get_array(),processed);np.testing.assert_array_equal(window.image_artists[1].get_array(),raw)


if __name__=='__main__':unittest.main()
