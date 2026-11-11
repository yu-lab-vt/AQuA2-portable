import tempfile,unittest
from collections import deque
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import h5py
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from aqua2.project import Project,new_project
from aqua2.project_io import matlab_array,materialize
from aqua2.viewer_render import FrameRequest,FrameReader,LatestRenderer,prepare_frame,video_pixels,grayscale_rgb,BlitCanvas
from aqua2.viewer_pixels import pixel_imshow


def fixture():
    data=new_project('pixel fidelity');shape=(8,9,1,3)
    movie=np.arange(np.prod(shape),dtype=np.uint8).reshape(shape,order='F')
    cells=np.empty((1,1),object);cells[0,0]=np.array([[2.],[3.],[10.],[11.]])
    reconstruction=np.zeros(shape,np.uint8);reconstruction[1,0,0,0]=128
    data['movies']={'channel1':{'stored':movie,'dF':movie.astype(np.float32)}}
    data['stages']={'detection':{'opts':dict(sz=shape,frameRate=.2,minValueDat1=0.,maxValueDat1=255.,maxdF1=20.),'channel1':{'local':{'evt':cells,'datR':reconstruction}}}}
    return Project(data)


class VideoPixels(unittest.TestCase):
    def test_default_raw_uint8_is_identical_and_contrast_is_explicit(self):
        p=fixture();request=FrameRequest(p,0,0,'channel1','local','Event','Original',(),(0,),False,1.,.5,False,True,False,0)
        main,raw,_,hits=prepare_frame(request)
        for component in range(3):np.testing.assert_array_equal(main[...,component],p.frame(0));np.testing.assert_array_equal(raw[...,component],p.frame(0))
        self.assertFalse(np.array_equal(main,prepare_frame(replace(request,auto_contrast=True))[0]))
        values=np.array([[10,20],[100,200]],np.uint8)
        np.testing.assert_array_equal(grayscale_rgb(values,(0,255))[...,0],values)
        self.assertEqual(int(grayscale_rgb(np.array([[1000]],np.uint16),(0,2000))[0,0,0]),128)

    def test_additive_overlay_preserves_background_and_reconstruction_strength(self):
        p=fixture();reader=FrameReader();request=FrameRequest(p,0,0,'channel1','local','Event','Overlay',(),(0,),False,1.,.5,True,True,False,0)
        from aqua2.viewer_model import color
        try:main,raw,_,hits=prepare_frame(request,reader)
        finally:reader.close()
        expected=np.floor(np.clip(p.frame(0)[1,0]+np.asarray(color(0))*.5*128,0,255)+.5).astype(np.uint8)
        np.testing.assert_array_equal(main[1,0],expected)
        np.testing.assert_array_equal(main[2,0],raw[2,0])  # zero reconstruction stays grey
        self.assertTrue(np.all(main[1,0]>=raw[1,0]))

    def test_fused_selection_edges_and_colour_match_reference(self):
        from scipy.ndimage import binary_erosion
        rng=np.random.default_rng(78);gray=rng.integers(0,256,(11,13),dtype=np.uint8);labels=rng.integers(0,4,gray.shape,dtype=np.int32)
        alpha=rng.random(gray.shape).astype(np.float32);alpha[labels==0]=0
        palette=np.array([[0,0,0],[.3,.6,.8],[.9,.4,.2],[.4,.2,.7]],np.float32);flags=np.array([False,False,True,True])
        colours=palette[labels].copy();strength=alpha.astype(float)
        for label in (2,3):
            mask=labels==label;edge=mask&~binary_erosion(mask,structure=np.ones((3,3),bool));colours[edge]=[1,.92,.15];strength[edge]=1
        expected=np.floor(np.clip(gray[...,None]+colours.astype(float)*strength[...,None]*.5*255,0,255)+.5).astype(np.uint8)
        actual,raw=video_pixels(gray,gray,labels,alpha,palette,flags,.5,True,True)
        np.testing.assert_allclose(actual,expected,atol=1,rtol=0);np.testing.assert_array_equal(raw[...,1],gray)

    def test_pixel_artist_orientation_aspect_and_zoom(self):
        rng=np.random.default_rng(12);pixels=rng.integers(0,256,(8,11,3),dtype=np.uint8)
        for origin in ('upper','lower'):
            for zoom in (False,True):
                buffers=[];samples=[]
                for fast in (False,True):
                    fig=Figure(figsize=(2.2,1.6),dpi=100);canvas=FigureCanvasAgg(fig);ax=fig.add_axes([0,0,1,1]);ax.set_axis_off()
                    (pixel_imshow if fast else lambda axis,image,**kw:axis.imshow(image,**kw))(ax,pixels,origin=origin,interpolation='nearest')
                    if zoom:ax.set_xlim(1.5,7.5)
                    canvas.draw();buffer=np.asarray(canvas.buffer_rgba()).copy();buffers.append(buffer)
                    visible=[]
                    for row in range(8):
                        for column in range(11):
                            if min(ax.get_xlim())<column<max(ax.get_xlim()):
                                x,y=ax.transData.transform((column,row));visible.append(buffer[buffer.shape[0]-1-int(y),int(x),:3]);np.testing.assert_array_equal(visible[-1],pixels[row,column])
                    samples.append(visible)
                # Backends differ at single-pixel nearest-neighbour cell
                # boundaries; every source pixel centre must remain exact.
                np.testing.assert_array_equal(*samples)

    def test_reader_keeps_axes_sparse_reconstruction_and_can_release_for_save(self):
        p=fixture()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'saved.aqua.h5';saved=p.save(path);reader=FrameReader()
            try:
                np.testing.assert_array_equal(reader.frame(saved,1,0,'channel1'),p.frame(1))
                np.testing.assert_array_equal(reader.reconstruction(saved,saved.spatial(),0,0),p.spatial()['datR'][:,:,0,0])
                native=Path(directory)/'mat.h5'
                with h5py.File(native,'w') as file:file['movie']=p.movie().transpose(3,2,1,0)
                other=fixture();other.data['movies']['channel1']['stored']=matlab_array(native,'movie')
                np.testing.assert_array_equal(reader.frame(other,2,0,'channel1'),p.frame(2))
                self.assertEqual(len(reader.files),1)
            finally:reader.close()
            renderer=LatestRenderer()
            try:
                renderer.reader.frame(saved,0,0,'channel1');renderer.release().result(timeout=3)
                self.assertFalse(renderer.reader.files);saved.save()
            finally:renderer.close()

    def test_cursor_strip_refresh_matches_full_refresh(self):
        figure=Figure(figsize=(5,2),dpi=100);canvas=FigureCanvasAgg(figure);ax=figure.subplots();ax.plot([0,1,2],[0,1,0]);cursor=ax.axvline(.5,linestyle='--')
        blit=BlitCanvas(canvas);blit.set_artists([cursor]);blit.paint()
        cursor.set_xdata([1.5]*2);blit.paint_cursor(cursor);partial=np.asarray(canvas.buffer_rgba()).copy()
        blit.invalidate();blit.paint();np.testing.assert_array_equal(partial,np.asarray(canvas.buffer_rgba()))

    def test_prepared_frame_waits_for_deadline_then_prefetches_without_60_cap(self):
        from aqua2.viewer import Workbench
        p=SimpleNamespace(opts={'frameRate':.2},shape=(3,3,1,100));request=SimpleNamespace(project=p,t=12)
        result=(3,request,('image','raw','rgba','hits'),None,.9)
        window=SimpleNamespace(renderer=SimpleNamespace(poll=Mock(side_effect=[result,None]),generation=3),project=p,draw_kind=None,playing=True,play_deadline=1.,
            draw_image=Mock(),frame=SimpleNamespace(set=Mock()),slider=SimpleNamespace(set=Mock()),frame_label=SimpleNamespace(configure=Mock()),cursor=None,
            present_times=deque(),rate_updated=0.,actual_fps=SimpleNamespace(set=Mock()),fps=SimpleNamespace(get=lambda:100),play_tick=Mock(),after=Mock(),poll_render=Mock())
        with patch('aqua2.viewer.time.perf_counter',return_value=.95):Workbench.poll_render(window)
        window.draw_image.assert_not_called();window.frame.set.assert_not_called()
        with patch('aqua2.viewer.time.perf_counter',return_value=1.):Workbench.poll_render(window)
        window.draw_image.assert_called_once();window.play_tick.assert_called_once();self.assertAlmostEqual(window.play_deadline,1.01)
