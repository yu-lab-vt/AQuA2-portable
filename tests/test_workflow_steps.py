import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock,patch
from types import SimpleNamespace
import numpy as np
import tifffile
from aqua2.workflow_steps import load_input,preprocess_project,detect_project,phase,can_detect,display_for
from aqua2.project import Project,new_project
from aqua2.viewer_steps import validated,PREPROCESS_FIELDS,DETECTION_FIELDS


def prepared():
    from aqua2.config import read_preset
    opts=read_preset(Path(__file__).parents[1]/'src/aqua2/data/parameters_for_batch.csv',1)
    shape=(6,7,1,10);opts.update(sz=shape,cut=10.,minValueDat1=0.,maxValueDat1=255.,singleChannel=True,BitDepth=8)
    working=np.linspace(0,1,np.prod(shape),dtype=np.float32).reshape(shape,order='F');stored=np.floor(working*255+.5).astype(np.uint8)
    data=new_project('prepared');data.update(opts=opts,movies={'channel1':dict(working=working,stored=stored,dF=working.copy())},runtime={'preprocessed':True},analysis_state={'preprocessed':True})
    data['stages']={'preprocessing':{'opts':opts}}
    return Project(data)


class WorkflowSteps(unittest.TestCase):
    def test_load_only_has_uncropped_original_pixels_and_no_analysis(self):
        movie=np.arange(6*8*9,dtype=np.uint16).reshape(6,8,9)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'raw.tif';tifffile.imwrite(path,movie,photometric='minisblack',metadata={'axes':'TYX'})
            with patch('aqua2.workflow_steps.preprocess',side_effect=AssertionError('Must not compute')):
                p=load_input(path,{'frames':[1,5,2]})
            np.testing.assert_array_equal(p.movie(),movie[[1,3]].transpose(1,2,0)[:,:,None,:])
            self.assertEqual(phase(p),'input');self.assertEqual(display_for(p),('Event','Original'));self.assertNotIn('dF',p.data['movies']['channel1']);self.assertFalse(can_detect(p))
            self.assertEqual(sorted(x.name for x in Path(folder).iterdir()),['raw.tif'])

    def test_explicit_volume_preview_retains_z_and_time(self):
        data=np.arange(3*2*5*6,dtype=np.uint16).reshape(3,2,5,6)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'volume.tif';tifffile.imwrite(path,data,photometric='minisblack',metadata={'axes':'TZYX'})
            p=load_input(path);np.testing.assert_array_equal(p.movie(),data.transpose(2,3,1,0))

    def test_saved_datOrg_preview_does_not_read_rgb_or_recrop(self):
        import h5py
        from aqua2.io import write_mat
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'saved.mat';raw=np.arange(4*7*8,dtype=np.uint16).reshape(4,1,7,8)
            with h5py.File(path,'w') as file:
                file['res/datOrg1']=raw
                values=dict(singleChannel=1,minValueDat1=0.,maxValueDat1=65535.,frameRate=.5,spatialRes=.5,sz=np.array([8,7,1,4]))
                for key,value in values.items():
                    ds=file.create_dataset('res/opts/'+key,data=np.atleast_2d(value));ds.attrs['MATLAB_class']=np.bytes_('double')
                file['res/opts'].attrs['MATLAB_class']=np.bytes_('struct')
            p=load_input(path,{'saved_result':True});np.testing.assert_array_equal(p.movie(),raw.transpose(3,2,1,0));self.assertEqual(p.shape,(8,7,1,4))

    def test_detection_reuses_unquantized_movie_and_invalidates_downstream(self):
        p=prepared();p.data['stages']['cfu']={'stale':True};p.data['annotations']['favorites']={'Event:channel1:local':[2]}
        before=p.movie(kind='working').copy()
        def detect(stage,**kwargs):
            np.testing.assert_array_equal(stage['datOrg1'],before)
            self.assertFalse(np.array_equal(stage['datOrg1'],p.movie().astype(np.float32)/255))
            self.assertEqual(stage['opts']['thrARScl'],4.)
            return {'opts':stage['opts']}
        with patch('aqua2.act.ac_detect',return_value=[]) as activity,patch('aqua2.detection.detect_from_stage3',side_effect=detect),patch('aqua2.detection.detection_mat_payload',side_effect=lambda x:x),patch('aqua2.workflow_steps.preprocess',side_effect=AssertionError('Do not recompute')):
            result=detect_project(p,{'thrARScl':4.})
        activity.assert_called_once();self.assertNotIn('cfu',result.data['stages']);self.assertEqual(result.data['annotations']['favorites'],{})
        self.assertIn('cfu',p.data['stages']);self.assertEqual(p.data['annotations']['favorites'],{'Event:channel1:local':[2]})
        self.assertEqual(display_for(result),('Event','Overlay'))

    def test_failed_detection_preserves_previous_result_and_parameters(self):
        p=prepared();p.data['stages']['detection']={'opts':p.opts,'old':True};p.data['stages']['cfu']={'old':True};value=p.opts['thrARScl']
        with patch('aqua2.act.ac_detect',side_effect=RuntimeError('cancelled')):
            with self.assertRaisesRegex(RuntimeError,'cancelled'):detect_project(p,{'thrARScl':4.})
        self.assertEqual(p.opts['thrARScl'],value);self.assertTrue(p.data['stages']['cfu']['old']);self.assertTrue(p.data['stages']['detection']['old'])

    def test_partial_manual_save_never_silently_uses_quantized_input(self):
        p=prepared()
        with tempfile.TemporaryDirectory() as folder:
            saved=p.save(Path(folder)/'partial.aqua.h5')
            self.assertEqual(phase(saved),'preprocess');self.assertEqual(display_for(saved),('Event','dF / σ'))
            self.assertNotIn('runtime',saved.data);self.assertNotIn('working',saved.data['movies']['channel1'])
            self.assertFalse(can_detect(saved));self.assertTrue(can_detect(p))
            with self.assertRaisesRegex(ValueError,'preprocessing'):detect_project(saved)

    def test_preprocessing_parameter_change_requires_new_preprocessing(self):
        p=prepared()
        with patch('aqua2.act.ac_detect') as activity:
            with self.assertRaisesRegex(ValueError,'Parameters changed'):detect_project(p,{'smoXY':2.})
        activity.assert_not_called()

    def test_parameter_validation_rejects_fractional_frames_and_invalid_modes(self):
        p=prepared();values={key:p.opts[key] for key,*_ in PREPROCESS_FIELDS}
        for key,value in [('movAvgWin',2.5),('frameRate',0),('registrateCorrect',4),('regMaskGap',-1)]:
            with self.subTest(key=key):
                with self.assertRaises(ValueError):validated({**values,key:value},PREPROCESS_FIELDS)

    def test_preprocessing_rerun_transport_omits_old_movies(self):
        from aqua2.workflow_steps import preprocessing_input
        p=prepared();p.data['stages']['cfu']={'large_old_result':np.ones((10,20))}
        value=preprocessing_input(p)
        self.assertEqual(value['movies'],{});self.assertEqual(value['stages'],{});self.assertNotIn('runtime',value)
        self.assertTrue(value['analysis_state']['preprocessed']);self.assertIn('working',p.data['movies']['channel1'])

    def test_gui_manual_save_keeps_exact_working_in_current_session_only(self):
        import queue
        from aqua2.viewer import Workbench
        p=prepared();expected=p.movie(kind='working').copy()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'manual.aqua.h5';saved=p.save(path);before=path.read_bytes()
            window=SimpleNamespace(project=p,loading=True,dirty=True,busy=False,ui_events=queue.Queue(),job=SimpleNamespace(events=queue.Queue()),
                finish_activity=Mock(),status=SimpleNamespace(set=Mock()),step_forms=Mock(),after=Mock(),poll=Mock())
            window.ui_events.put(('saved',saved,None));Workbench.poll(window)
            self.assertTrue(can_detect(window.project));self.assertFalse(window.dirty)
            np.testing.assert_array_equal(window.project.movie(kind='working'),expected)
            self.assertFalse(can_detect(Project.open(path)));self.assertEqual(path.read_bytes(),before)

    def test_one_click_advances_only_after_result_is_ready(self):
        from collections import deque
        from aqua2.viewer import Workbench
        window=SimpleNamespace(chain_ready=False,chain_steps=deque([{'operation':'detect'}]),busy=False,loading=False,start_job=Mock())
        Workbench.advance_chain(window);window.start_job.assert_not_called()
        window.chain_ready=True;window.busy=True;Workbench.advance_chain(window);window.start_job.assert_not_called()
        window.busy=False;Workbench.advance_chain(window);window.start_job.assert_called_once_with({'operation':'detect'});self.assertFalse(window.chain_ready)
