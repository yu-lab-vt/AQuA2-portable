import csv
import multiprocessing
import tempfile
import threading
import unittest
from pathlib import Path
import numpy as np
import tifffile
from aqua2.compare import compare
from aqua2.io import MatStructArray
from aqua2.project import Project,new_project,export_project
from aqua2.project_io import materialize


def sample():
    shape=(12,13,2,3);data=new_project('export fixture')
    movie=np.linspace(0,.5,np.prod(shape),dtype=np.float32).reshape(shape,order='F')
    data['movies']={'channel1':dict(stored=movie,working=movie,dF=movie)}
    events=np.empty((1,1),object);events[0,0]=np.array([[2.],[158.]])
    rises=np.empty((1,1),object);rises[0,0]=dict(rgh=np.array([[2.]]),rgw=np.array([[1.]]),rgl=np.array([[1.,2.]]),dlyMap50=np.array([[[1.,3.]]]))
    curves=dict(tBegin=np.array([[1.]]),tEnd=np.array([[2.]]))
    features=dict(fts=dict(basic={},loc={},curve=curves,notes={'propDirectionOrder':np.empty((1,0),object)},bds=np.empty((0,0))),dMat=np.ones((1,3,2),np.float32),dffMat=np.ones((1,3,2),np.float32),dffAlignedMat=np.ones((1,111)),opts={'duplicate':True})
    reconstructed=np.zeros(shape,np.uint8);reconstructed[1,0,0,0]=17;reconstructed[8,9,1,2]=23
    spatial=dict(evt=events,riseLst=rises,datR=reconstructed,seLst=events,features=features)
    data['stages']={'preprocessing':{'cache':movie},'detection':{'opts':dict(sz=shape,frameRate=.5,minValueDat1=0.,maxValueDat1=1.),'channel1':{'local':spatial,'global':dict(spatial)}}}
    data['annotations']['favorites']={'Event:channel1:local':[0],'CFU:channel1:global':[0]}
    project=Project(data)
    project._cache[('cfus','channel1')]=[dict(eventIds=np.array([0]),weightMap=np.ones(shape[:3]),occurrence=np.array([True,False,True]),parentId=0,frequencyStats=dict(mainFreq=1.,method='Mean',peakFreq80=np.nan,dt=np.array([1.]))) ]
    return project


class ResultExports(unittest.TestCase):
    def test_pure_masks_are_independent_of_background_and_include_actual_channels(self):
        from aqua2.viewer_model import label_layer
        project=sample()
        project.data['movies']['channel2']=project.data['movies']['channel1']
        project.data['stages']['detection']['channel2']=project.data['stages']['detection']['channel1']
        project._cache[('cfus','channel2')]=project.cfus('channel1')
        with tempfile.TemporaryDirectory() as folder:
            output=export_project(project,folder,selection={'mask_only':True})
            self.assertEqual(len(list(output.glob('*.tif'))),6)
            self.assertFalse(list(output.glob('*masked.tif')));self.assertFalse(list(output.glob('*.csv')))
            for channel in project.channels:
                movie=tifffile.imread(output/f'{channel}_local_events_mask_only.tif')
                self.assertEqual(movie.shape,(3,2,12,13,3))
                for t in range(3):
                    for z in range(2):
                        labels,_,palette=label_layer(project,'Event',channel,'local',t,z)
                        np.testing.assert_array_equal(movie[t,z],np.floor(palette[labels]*255+.5).astype(np.uint8))
                        self.assertTrue(np.all(movie[t,z][labels==0]==0))
                cfu=tifffile.imread(output/f'{channel}_cfu_mask_only.tif')
                self.assertEqual(cfu.shape,(2,12,13,3))

    def test_export_inherits_gui_intensity_settings_and_reconstruction(self):
        from aqua2.viewer_render import FrameRequest,FrameReader,prepare_frame
        project=sample();settings={'gamma':1.7,'opacity':.8,'auto_contrast':True}
        reader=FrameReader()
        try:
            with tempfile.TemporaryDirectory() as folder:
                output=export_project(project,folder,selection={'masked_movie':True},display_settings=settings)
                movie=tifffile.imread(output/'channel1_local_events_masked.tif')
                for t in range(3):
                    for z in range(2):
                        request=FrameRequest(project,t,z,'channel1','local','Event','Overlay',(),(0,),False,1.7,.8,True,False,False,0,True)
                        np.testing.assert_array_equal(movie[t,z],prepare_frame(request,reader)[0])
                cfu=tifffile.imread(output/'channel1_cfu_mean_masked.tif')
                for z in range(2):
                    request=FrameRequest(project,0,z,'channel1','local','CFU','Mean Projection',(),(0,),False,1.7,.8,True,False,False,0,True)
                    np.testing.assert_array_equal(cfu[z],prepare_frame(request,reader)[0])
        finally:reader.close()

    def test_old_cfu_favorites_can_be_unstarred_and_exported_consistently(self):
        from aqua2.viewer import Workbench
        from aqua2.result_export import favorites
        from types import SimpleNamespace
        from unittest.mock import Mock
        project=sample()
        window=SimpleNamespace(project=project,guarded=lambda:True,selected=[0],
            channel=SimpleNamespace(get=lambda:'channel1'),scope=SimpleNamespace(get=lambda:'global'),
            mode=SimpleNamespace(get=lambda:'CFU'),favorites_key=lambda:'CFU:channel1:local',refresh_table=Mock())
        Workbench.favorite(window)
        self.assertEqual(favorites(project,'channel1','local','CFU'),set());self.assertTrue(window.dirty)

    def test_default_all_categories_coordinates_and_favorites(self):
        project=sample()
        with tempfile.TemporaryDirectory() as folder:
            directory=export_project(project,folder)
            with (directory/'channel1_local_features.csv').open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.reader(stream))
            self.assertEqual(rows[1],['Favorite','1'])
            with (directory/'channel1_global_features.csv').open(encoding='utf-8-sig',newline='') as stream:self.assertEqual(list(csv.reader(stream))[1],['Favorite','0'])
            with (directory/'channel1_cfu_features.csv').open(encoding='utf-8-sig',newline='') as stream:record=next(csv.DictReader(stream))
            self.assertEqual(record['Favorite'],'1');self.assertEqual(record['Event IDs (1-based)'],'1')
            movie=tifffile.imread(directory/'channel1_local_events_masked.tif')
            self.assertEqual(movie.shape,(3,2,12,13,3));self.assertEqual(movie.dtype,np.uint8)
            np.testing.assert_array_equal(movie[0,0,0,0],[0,0,0])
            from aqua2.viewer_model import color
            expected=int(np.floor(project.frame(0)[1,0]*255+color(0)[0]*.5*17+.5))
            self.assertEqual(int(movie[0,0,1,0,0]),expected)
            self.assertEqual(int(movie[0,0,-1,-1,0]),int(np.floor(project.frame(0)[-1,-1]*255+.5)))
            np.testing.assert_array_equal(tifffile.imread(directory/'rising_map/channel1_local_event_1_rising50.tif').ravel(),[0.,1.])
            self.assertEqual(len(list((directory/'rising_map').glob('*.png'))),4);self.assertEqual(list(directory.glob('*rising50*')),[])
            cfu=tifffile.imread(directory/'channel1_cfu_mean_masked.tif');self.assertEqual(cfu.shape,(2,12,13,3))
            self.assertFalse((directory/'channel1_cfu_masked.tif').exists())
            self.assertEqual(list(directory.glob('*.mat')),[])
            import json
            metadata=json.loads((directory/'export_metadata.json').read_text(encoding='utf-8'));self.assertTrue(metadata['complete']);self.assertTrue(all(metadata['selection'].values()))
            self.assertTrue(all((directory/name).is_file() for name in metadata['files']))

    def test_selection_empty_selection_and_no_overwrite(self):
        project=sample()
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'empty'
            with self.assertRaises(ValueError):export_project(project,target,selection={})
            self.assertFalse(target.exists())
            target.mkdir();old=target/'old.mat';old.write_bytes(b'preserve')
            result=export_project(project,target,selection={'cfu_table':True})
            self.assertNotEqual(result,target);self.assertEqual(old.read_bytes(),b'preserve')
            self.assertEqual(sorted(p.name for p in result.iterdir()),['channel1_cfu_features.csv','export_metadata.json'])

    def test_cfu_image_is_mean_of_all_frames_not_first_frame(self):
        from aqua2.result_export import cfu_mean_image
        from aqua2.viewer_render import grayscale_rgb
        project=sample();project.cfus()[0]['weightMap'][:]=0
        expected=grayscale_rgb(project.movie().mean(axis=-1,dtype=np.float64),(0,1))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'mean.tif';cfu_mean_image(project,path,'channel1',lambda *args:None)
            actual=tifffile.imread(path)
            np.testing.assert_array_equal(actual,expected.transpose(2,0,1,3))
            self.assertFalse(np.array_equal(actual[0],grayscale_rgb(project.frame(0),(0,1))))
            with tifffile.TiffFile(path) as file:self.assertEqual(len(file.pages),project.shape[2])

    def test_manual_save_compacts_without_mutation_and_reconstructs_exactly(self):
        project=sample();original=project.spatial();expected=original['datR'].copy()
        with tempfile.TemporaryDirectory() as folder:
            saved=project.save(Path(folder)/'manual.aqua.h5');spatial=saved.spatial()
            self.assertNotIn('preprocessing',saved.data['stages']);self.assertEqual(set(saved.data['movies']['channel1']),{'stored','dF'})
            self.assertNotIn('seLst',spatial);self.assertNotIn('opts',spatial['features']);self.assertNotIn('dffAlignedMat',spatial['features'])
            self.assertIn('ov',spatial);np.testing.assert_array_equal(materialize(spatial)['datR'],expected)
            np.testing.assert_array_equal(np.asarray(saved.movie()),project.movie())
            self.assertIn('working',project.data['movies']['channel1']);self.assertIn('seLst',original);self.assertIn('dffAlignedMat',original['features'])
            second=saved.save();np.testing.assert_array_equal(materialize(second.spatial())['datR'],expected)

    def test_chunked_transport_keeps_fortran_cells_and_types(self):
        from aqua2.ipc import send_value,receive_value
        send,receive=multiprocessing.Pipe();errors=[]
        records=np.empty((1,1),object).view(MatStructArray);records.mat_fields=('pixels','empty');records[0,0]={'pixels':np.arange(1600000,dtype=np.float32).reshape(1000,1600,order='F'),'empty':np.empty((0,1),bool)}
        value={'records':records,'C':np.arange(80000).reshape(200,400),'scalar':np.float32(.1),'NaN':np.nan}
        def write():
            try:send_value(send,value)
            except BaseException as error:errors.append(error)
            finally:send.close()
        thread=threading.Thread(target=write);thread.start()
        try:received=receive_value(receive)
        finally:receive.close();thread.join(10)
        self.assertFalse(thread.is_alive());self.assertFalse(errors)
        self.assertTrue(compare(value,received)['passed']);self.assertEqual(received['records'].mat_fields,records.mat_fields)
        self.assertTrue(received['records'][0,0]['pixels'].flags.f_contiguous)
