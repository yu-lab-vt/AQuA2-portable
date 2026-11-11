import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import h5py
import numpy as np
from aqua2.io import MatStructArray,read_mat
from aqua2.compare import compare
from aqua2.project_io import save_tree,load_tree,ArrayRef,matlab_array
from aqua2.project import Project,new_project,validate_cfu_binding,relink_saved_source
from aqua2.workflow import stored_intensities,run_workflow,run_batch
from aqua2.fea import features_from_mat_payload,features_mat_payload
from aqua2.viewer_model import layer,rising_image,curve_series

LOCAL=Path(__file__).resolve().parents[1]/'reference/local'


class ProjectTests(unittest.TestCase):
    def test_close_requires_confirmation_before_stopping_tasks(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from aqua2.viewer import Workbench
        for busy,dirty in ((False,False),(True,False),(False,True),(True,True)):
            for confirmed in (False,True):
                with self.subTest(busy=busy,dirty=dirty,confirmed=confirmed):
                    window=SimpleNamespace(loading=False,busy=busy,dirty=dirty,playing=True,
                        job=Mock(),io=Mock(),destroy=Mock())
                    with patch('aqua2.viewer.messagebox.askokcancel',return_value=confirmed) as prompt:
                        Workbench.close(window)
                    prompt.assert_called_once()
                    self.assertEqual(prompt.call_args.kwargs['default'],'cancel')
                    self.assertEqual('cancels the current task' in prompt.call_args.args[1],busy)
                    self.assertEqual('edits will be discarded' in prompt.call_args.args[1],dirty)
                    if confirmed:
                        window.job.cancel.assert_called_once()
                        window.io.shutdown.assert_called_once_with(wait=False,cancel_futures=True)
                        window.destroy.assert_called_once()
                    else:
                        window.job.cancel.assert_not_called()
                        window.io.shutdown.assert_not_called()
                        window.destroy.assert_not_called()
                        self.assertTrue(window.playing)

    def test_close_waits_for_project_io(self):
        from types import SimpleNamespace
        from aqua2.viewer import Workbench
        with patch('aqua2.viewer.messagebox.showinfo') as info,patch('aqua2.viewer.messagebox.askokcancel') as prompt:
            Workbench.close(SimpleNamespace(loading=True))
        info.assert_called_once();prompt.assert_not_called()

    def test_preprocessing_movies_are_released_without_checkpoint(self):
        import gc,weakref
        refs=[];shape=(3,4,1,6)
        def prepare(*args,**kwargs):
            smooth=np.ones(shape,np.float32);baseline=np.zeros(shape,np.float32)
            refs.extend([weakref.ref(smooth),weakref.ref(baseline)])
            return {'opts':{'sz':shape,'cut':4},'datOrg1':np.ones(shape,np.float32),
                    'dF1':np.zeros(shape,np.float32),'activeRegions1':[],
                    'intermediate1':{'smoothed':smooth,'F0':baseline}}
        def detect(stage,**kwargs):
            self.assertNotIn('intermediate1',stage)
            gc.collect()
            self.assertTrue(all(ref() is None for ref in refs))
            raise RuntimeError('captured detection boundary')
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'input.tif';source.write_bytes(b'test');output=Path(folder)/'result.aqua.h5'
            with patch('aqua2.workflow.preprocess',side_effect=prepare),patch('aqua2.workflow.detect_from_stage3',side_effect=detect):
                with self.assertRaisesRegex(RuntimeError,'captured detection boundary'):run_workflow(source,output,{'with_cfu':False})
            self.assertEqual(list(Path(folder).iterdir()),[source])

    def test_progress_counts_unknown_steps_and_elapsed_time(self):
        from aqua2.worker_client import ProgressState
        state = ProgressState(now=10)
        self.assertIsNone(state.fraction)
        state.update({'done': 30, 'total': 120}, now=11)
        self.assertEqual(state.fraction, .25)
        self.assertEqual(state.counts, '30 / 120')
        state.update({'done': .6}, now=12)
        self.assertEqual(state.fraction, .6)
        state.update({'detail': 'Labeling connected components'}, now=13)
        self.assertIsNone(state.fraction)
        self.assertEqual(state.counts, '')
        state.cpu_seconds = 2.5
        self.assertIn('Elapsed 00:00:10', state.timing(now=20))
        self.assertIn('Last update 00:00:07', state.timing(now=20))
        self.assertIn('CPU time 2 s', state.timing(now=20))

    def test_typed_atomic_roundtrip_and_failed_write(self):
        cell=np.empty((1,2),object);cell[0]=[np.array([[np.nan,np.inf]],np.float32),np.empty((0,3),bool)]
        records=np.empty((0,1),object).view(MatStructArray);records.mat_fields=('id','events')
        value={'cell':cell,'struct':records,'text':np.array([['Region α']]),'singleton':np.zeros((3,4,1,2),np.uint16),'empty':np.empty((0,1),float),'scalar32':np.float32(.1),'unsigned_scalar':np.uint64(1),'zero_d':np.array(3,np.float32)}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'project.h5';save_tree(path,value);self.assertTrue(compare(value,load_tree(path,lazy=False))['passed'])
            before=path.read_bytes()
            with self.assertRaises(TypeError):save_tree(path,{'bad':object()})
            self.assertEqual(path.read_bytes(),before);self.assertEqual(list(Path(folder).glob('*.partial')),[])

    def test_lazy_mat_axes_and_portable_embedding(self):
        movie=np.arange(5*7*2*11,dtype=np.uint16).reshape((5,7,2,11),order='F')
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'input.mat';path=Path(folder)/'project.h5'
            with h5py.File(source,'w') as file:file.create_dataset('res/datOrg1',data=movie.transpose(3,2,1,0))
            ref=matlab_array(source,'res/datOrg1');np.testing.assert_array_equal(ref[:,:,1,3],movie[:,:,1,3])
            save_tree(path,{'movie':ref});loaded=load_tree(path);self.assertIsInstance(loaded['movie'],ArrayRef)
            save_tree(path,loaded,embed_external=True);source.unlink();np.testing.assert_array_equal(load_tree(path,lazy=False)['movie'],movie)

    def test_refuse_foreign_hdf_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'input.h5'
            with h5py.File(path,'w') as file:file['raw']=np.arange(8)
            with self.assertRaises(ValueError):save_tree(path,{'x':1})
            with h5py.File(path) as file:np.testing.assert_array_equal(file['raw'],np.arange(8))

    def test_relink_requires_same_content_and_retains_coordinates(self):
        import hashlib
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'source.mat';moved=Path(folder)/'moved.mat';wrong=Path(folder)/'wrong.mat'
            movie=np.arange(24,dtype=np.uint16).reshape((3,4,1,2),order='F')
            with h5py.File(source,'w') as file:file['res/datOrg1']=movie.transpose(3,2,1,0)
            moved.write_bytes(source.read_bytes());wrong.write_bytes(b'different')
            data=new_project('linked',{'path':str(source.resolve()),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
            data['opts']={'sz':movie.shape};data['movies']={'channel1':{'stored':matlab_array(source,'res/datOrg1')}}
            project=Project(data).save(Path(folder)/'project.h5')
            with self.assertRaisesRegex(ValueError,'different'):relink_saved_source(project,wrong)
            self.assertEqual(project.data['source']['path'],str(source.resolve()))
            source.unlink();relinked=relink_saved_source(project,moved)
            np.testing.assert_array_equal(relinked.frame(1),movie[:,:,0,1])

    def test_lazy_noise_map_reaches_feature_kernel_as_exact_array(self):
        from aqua2.workflow import recompute_features
        shape=(256,256,1,2);noise=np.arange(256*256,dtype=np.float64).reshape(256,256,1)
        data=new_project('lazy correction');events=np.empty((1,1),object);events[0,0]=np.array([[1.]])
        data['movies']={'channel1':{'working':np.zeros(shape,np.float32),'stored':np.zeros(shape,np.uint16)}}
        data['stages']={'detection':{'opts':{'sz':shape,'frameRate':1.,'correctPars':noise,'minValueDat1':0.,'maxValueDat1':1.},'channel1':{'local':{
            'evt':events,'datR':np.zeros(shape,np.uint8),'riseLst':np.empty((1,0),object)}}}}
        with tempfile.TemporaryDirectory() as folder:
            project=Project(data).save(Path(folder)/'project.h5');before=project.path.read_bytes()
            self.assertIsInstance(project.opts['correctPars'],ArrayRef)
            def kernel(movie,spatial,options,**kwargs):
                self.assertIsInstance(options['correctPars'],np.ndarray)
                np.testing.assert_array_equal(options['correctPars'],noise)
                raise RuntimeError('boundary captured')
            with patch('aqua2.features.compute_features',side_effect=kernel):
                with self.assertRaisesRegex(RuntimeError,'boundary captured'):recompute_features(project.path)
            self.assertEqual(project.path.read_bytes(),before)

    def test_v73_scalar_struct_with_typed_cells_is_not_struct_array(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'network.mat'
            with h5py.File(path,'w') as file:
                group=file.create_group('network');group.attrs['MATLAB_class']=np.bytes_('struct')
                counts=group.create_dataset('count',data=np.ones((2,3)));counts.attrs['MATLAB_class']=np.bytes_('double')
                value=file.create_dataset('value',data=np.array([[1.]]));value.attrs['MATLAB_class']=np.bytes_('double')
                for name,shape in (('location',(2,3)),('time',(1,3))):
                    cell=group.create_dataset(name,shape=shape,dtype=h5py.ref_dtype);cell.attrs['MATLAB_class']=np.bytes_('cell');cell[...]=value.ref
            result=read_mat(path)['network'];self.assertIsInstance(result,dict);self.assertEqual(result['location'].shape,(3,2));self.assertEqual(result['time'].shape,(3,1))

    def test_matlab_saved_quantization_boundary(self):
        x=np.array([0,.05,.15,.25,.5,1,-2],np.float32)
        out=stored_intensities(x,{'minValueDat1':10.,'maxValueDat1':20.,'BitDepth':16})
        np.testing.assert_array_equal(out,[10,11,12,13,15,20,0]);self.assertEqual(out.dtype,np.uint16)

    def test_display_axes_hits_curves_and_no_mutation(self):
        data=new_project('tiny');shape=(4,5,1,6);movie=np.arange(np.prod(shape),dtype=np.float32).reshape(shape,order='F')
        event=np.array([1+4*2+20*2,2+4*2+20*2]);cells=np.empty((1,1),object);cells[0,0]=(event+1.).reshape(-1,1)
        rises=np.empty((1,1),object);rises[0,0]={'rgh':np.array([[2,3]]),'rgw':np.array([[3]]),'rgl':np.array([[1]]),'dlyMap50':np.array([[3.],[4.]])}
        f={'fts':{},'dffMat':np.ones((1,6,2),np.float32),'dMat':np.ones((1,6,2),np.float32)}
        data['movies']={'channel1':{'stored':movie}};data['stages']={'detection':{'opts':{'sz':shape,'frameRate':.5},'channel1':{'local':{'evt':cells,'riseLst':rises,'datR':movie,'features':f}}}}
        project=Project(data);before=movie.copy();rgba,hits=layer(project,'Event','channel1','local',2,0)
        self.assertEqual(hits[1,2],1);self.assertEqual(hits[2,2],1);self.assertEqual(np.count_nonzero(hits),2)
        np.testing.assert_array_equal(project.frame(2),movie[:,:,0,2]);np.testing.assert_array_equal(movie,before)
        rising=rising_image(project,'channel1','local',0,0);self.assertEqual(rising[1,2],1);self.assertEqual(rising[2,2],1.5)
        self.assertEqual(curve_series(project,'Event','channel1','local',[0])[0][1][-1],2.5)
        with self.assertRaises(ValueError):validate_cfu_binding(project,{'cfuInfo1':[{'eventIds':np.array([0]),'weightMap':np.zeros((5,4)),'occurrence':np.zeros(6)}],'cfuInfo2':[]})

    def test_analysis_memory_default_and_explicit_manual_save(self):
        from aqua2.features import compute_features
        shape=(3,4,1,6);opts={'sz':shape,'cut':4,'frameRate':.5,'minValueDat1':10,'maxValueDat1':20,'BitDepth':16,'singleChannel':True,'networkFeatures':False,'propMetric':False}
        stage={'opts':opts,'datOrg1':np.ones(shape,np.float32),'dF1':np.zeros(shape,np.float32),'activeRegions1':[]}
        spatial={'evt':[],'datR':np.zeros(shape,np.uint8),'riseLst':[],'opts':opts};spatial['features']=compute_features(stage['datOrg1'],spatial,opts)
        detected={'opts':opts,'channel1':{'local':spatial,'global':{'evt':[],'datR':np.empty((0,0)),'riseLst':[]}}}
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'Original_é.tif';source.write_bytes(b'fixed source');output=Path(folder)/'analysis.aqua.h5';config={'with_cfu':False,'options':{'maxSize':float('inf')}}
            with patch('aqua2.workflow.preprocess',return_value=stage) as prep,patch('aqua2.workflow.detect_from_stage3',return_value=detected) as detect:
                project=run_workflow(source,output,config);self.assertEqual(project.summary()['events']['channel1'],0)
                self.assertIsNone(project.path);self.assertFalse(output.exists());self.assertFalse(output.with_name(output.name+'.work').exists())
                project.data['annotations']['favorites']={'channel1:local':[0]};saved=project.save(output)
                self.assertEqual(saved.data['annotations']['favorites'],{'channel1:local':[0]})
                with self.assertRaisesRegex(ValueError,'removed'):run_workflow(source,output,config,resume=True)

    def test_failed_analysis_does_not_replace_existing_manual_project(self):
        from aqua2.features import compute_features
        shape=(3,4,1,6);opts={'sz':shape,'cut':4,'frameRate':.5,'minValueDat1':10,'maxValueDat1':20,'singleChannel':True,'networkFeatures':False,'propMetric':False}
        stage={'opts':opts,'datOrg1':np.ones(shape,np.float32),'dF1':np.zeros(shape,np.float32),'activeRegions1':[]}
        spatial={'evt':[],'datR':np.zeros(shape,np.uint8),'riseLst':[]};spatial['features']=compute_features(stage['datOrg1'],spatial,opts)
        detected={'opts':opts,'channel1':{'local':spatial,'global':{'evt':[],'datR':np.empty((0,0)),'riseLst':[]}}}
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'raw.tif';source.write_bytes(b'unchanged');output=Path(folder)/'analysis.aqua.h5'
            with patch('aqua2.workflow.preprocess',return_value=stage),patch('aqua2.workflow.detect_from_stage3',return_value=detected):
                run_workflow(source,output,{'with_cfu':False},save_project=True)
            before=output.read_bytes();config={'with_cfu':False,'options':{'thrARScl':4.}}
            with patch('aqua2.workflow.preprocess',return_value=stage),patch('aqua2.workflow.detect_from_stage3',side_effect=RuntimeError('test interruption')):
                with self.assertRaisesRegex(RuntimeError,'interruption'):run_workflow(source,output,config,overwrite=True,save_project=True)
            self.assertEqual(output.read_bytes(),before);self.assertFalse(output.with_name(output.name+'.work').exists())


@unittest.skipUnless((LOCAL/'stage6-regions.mat').exists(),'Generate stage-6 feature references')
class PersistedFeatureTests(unittest.TestCase):
    def test_imported_features_without_saved_curves_export_tables(self):
        from aqua2.project import export_project
        from aqua2.project_io import materialize
        import json
        path=LOCAL/'stage8-cibr.aqua.h5'
        if not path.exists():self.skipTest('Generate stage-8 integration project')
        project=Project.open(path);stored=project.features();count=len(project.events())
        stored.update({key:np.empty((count,0),np.float32) for key in ('dMat','dffMat','dffAlignedMat')})
        with tempfile.TemporaryDirectory() as folder:
            export_project(project,folder,selection={'event_table':True});folder=Path(folder)
            self.assertTrue((folder/'channel1_local_features.csv').exists())
            self.assertFalse((folder/'channel1_local_curves.npz').exists())
            self.assertTrue(json.loads((folder/'export_metadata.json').read_text())['complete']);self.assertFalse(list(folder.glob('*.mat')))

    def test_inverse_feature_boundary_with_regions_is_exact(self):
        base=read_mat(LOCAL/'pipeline.mat')['features'];regions=read_mat(LOCAL/'stage6-regions.mat')
        payload={**base,'fts':regions['features']};roundtrip=features_mat_payload(features_from_mat_payload(payload))
        check=compare(payload,roundtrip);self.assertTrue(check['passed'],check['issues'])
