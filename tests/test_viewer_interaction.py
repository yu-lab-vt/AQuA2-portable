import threading,time,unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backend_bases import MouseEvent
from matplotlib.widgets import PolygonSelector
from aqua2.project import Project,new_project
from aqua2.viewer_model import polygon_pixels,layer
from aqua2.viewer_render import LatestRenderer,BlitCanvas


class ViewerInteraction(unittest.TestCase):
    def test_cfu_curve_windows_style_members_and_other_events_without_changing_values(self):
        from aqua2.viewer_model import plot_curve,color
        figure=Figure();ax=figure.subplots();x=np.arange(9)*.5;y=np.arange(9,dtype=float)
        own=np.array([0,1,1,0,0,1,1,0,0],bool)
        other=np.array([0,0,0,1,1,1,0,0,0],bool)
        lines=plot_curve(ax,2,x,y,record={'timeWindow':own[None,:],'nonTimeWindow':other[:,None]},label='CFU 3')
        self.assertEqual(len(lines),3);self.assertEqual(lines[0].get_label(),'CFU 3')
        np.testing.assert_array_equal(lines[0].get_ydata(),y)
        np.testing.assert_array_equal(lines[1].get_ydata(),np.where(own,y,np.nan))
        np.testing.assert_array_equal(lines[2].get_ydata(),np.where(other,y,np.nan))
        self.assertGreater(lines[1].get_linewidth(),lines[0].get_linewidth())
        np.testing.assert_array_equal(lines[2].get_color(),(.8,.8,.8))
        np.testing.assert_array_equal(y,np.arange(9,dtype=float))
        self.assertEqual(len(plot_curve(ax,0,x,y,record={})),1)
        self.assertEqual(len(plot_curve(ax,0,x,y)),1)

    def test_playback_clock_corrects_timer_rounding_without_unseen_catchup(self):
        from aqua2.viewer_render import playback_delay
        deadline,delay=playback_delay(0.,.06,10)
        self.assertAlmostEqual(deadline,.1);self.assertEqual(delay,40)
        deadline,delay=playback_delay(deadline,.168,10)
        self.assertAlmostEqual(deadline,.2);self.assertEqual(delay,32)
        deadline,delay=playback_delay(deadline,.6,10)
        self.assertEqual(deadline,.6);self.assertEqual(delay,1)

    def test_space_does_not_double_invoke_a_focused_play_button(self):
        from aqua2.viewer import Workbench,ttk
        for focus in (None,Mock(spec=ttk.Button),Mock(spec=ttk.Entry)):
            window=SimpleNamespace(focus_get=lambda:focus,toggle_play=Mock())
            result=Workbench.space_key(window,None)
            self.assertEqual(window.toggle_play.call_count,int(focus is None))
            self.assertEqual(result,'break' if focus is None else None)

    def test_polygon_is_not_bounding_box_and_respects_z_orientation(self):
        vertices=[(1,1),(5,1),(1,5)];shape=(8,9,3,2)
        pixels=polygon_pixels(vertices,shape,2)
        self.assertIn(1+1*8+2*72,pixels);self.assertNotIn(5+5*8+2*72,pixels)
        np.testing.assert_array_equal(pixels,polygon_pixels(vertices[::-1],shape,2))
        self.assertTrue(np.all(pixels//72==2))
        concave=polygon_pixels([(1,1),(5,1),(5,2),(2,2),(2,5),(1,5)],shape)
        self.assertIn(1+4*8,concave);self.assertNotIn(4+4*8,concave)
        clipped=polygon_pixels([(-4,-4),(3,0),(0,3)],shape)
        self.assertTrue(np.all((clipped>=0)&(clipped<72)))
        for invalid in ([(1,1),(2,2)],[(1,1),(2,2),(3,3)],[(0,0),(1,1),(np.nan,0)]):
            with self.assertRaises(ValueError):polygon_pixels(invalid,shape)

    def test_selector_finishes_only_when_three_vertices_close_to_start(self):
        figure=Figure(figsize=(3,3),dpi=100);canvas=FigureCanvasAgg(figure);ax=figure.subplots();ax.set(xlim=(0,10),ylim=(0,10));canvas.draw();done=[]
        selector=PolygonSelector(ax,lambda vertices:done.append(vertices),useblit=True)
        def click(x,y):
            px,py=ax.transData.transform((x,y))
            for name in ('motion_notify_event','button_press_event','button_release_event'):
                canvas.callbacks.process(name,MouseEvent(name,canvas,px,py,button=1))
        for point in ((2,2),(8,2),(2,8)):click(*point)
        self.assertEqual(done,[]);click(2,2);self.assertEqual(len(done),1);self.assertEqual(len(done[0]),3)
        selector.disconnect_events()

    def test_polygon_blit_keeps_the_other_comparison_image_visible(self):
        figure=Figure(figsize=(6,3),dpi=100);canvas=FigureCanvasAgg(figure);axes=figure.subplots(1,2)
        images=[ax.imshow(np.ones((10,10)),vmin=0,vmax=1,cmap='gray') for ax in axes]
        blit=BlitCanvas(canvas);blit.set_artists(images);blit.paint();blit.suspend()
        selector=PolygonSelector(axes[0],lambda vertices:None,useblit=True)
        px,py=axes[0].transData.transform((3,3))
        for name in ('motion_notify_event','button_press_event','button_release_event'):
            canvas.callbacks.process(name,MouseEvent(name,canvas,px,py,button=1))
        px,py=axes[1].transData.transform((5,5));screen=np.asarray(canvas.buffer_rgba())
        self.assertGreater(int(screen[int(figure.bbox.height-py),int(px),0]),250)
        selector.disconnect_events();blit.resume();self.assertTrue(all(i.get_animated() for i in images))

    def test_slow_renderer_coalesces_and_never_queues_every_playback_tick(self):
        entered=threading.Event();release=threading.Event();calls=[]
        def calculate(value):
            calls.append(value)
            if value==0:entered.set();release.wait(5)
            return value
        renderer=LatestRenderer(calculate)
        try:
            renderer.submit(0);self.assertTrue(entered.wait(2))
            for value in range(1,50):renderer.submit(value)
            self.assertEqual(calls,[0]);release.set()
            deadline=time.monotonic()+5;received=[]
            while time.monotonic()<deadline:
                result=renderer.poll()
                if result is not None:received.append(result);break
                time.sleep(.002)
            self.assertEqual(calls,[0,49]);self.assertEqual([r[2] for r in received],[49]);self.assertIsNone(renderer.pending)
            renderer.submit(50);renderer.invalidate()
            time.sleep(.01);self.assertIsNone(renderer.poll())
        finally:release.set();renderer.close()

    def test_frame_controls_commit_only_after_actual_image_paint(self):
        from aqua2.viewer import Workbench
        p=SimpleNamespace(opts={'frameRate':.2},shape=(3,3,1,100));request=SimpleNamespace(project=p,t=12)
        result=(1,request,('image','raw','rgba','hits'),None,time.perf_counter())
        state={};order=[]
        def variable(name):return SimpleNamespace(set=lambda value:(state.__setitem__(name,value),order.append(name)))
        window=SimpleNamespace(renderer=SimpleNamespace(poll=lambda:result),project=p,draw_kind=None,playing=False,
            draw_image=lambda *args:order.append('paint'),frame=variable('frame'),slider=variable('slider'),
            frame_label=SimpleNamespace(configure=Mock()),cursor=None,after=Mock(),poll_render=Mock())
        Workbench.poll_render(window)
        self.assertEqual(order,['paint','frame','slider']);self.assertEqual(state,{'frame':13,'slider':13});self.assertEqual(window.displayed_frame,12)

    def test_event_index_matches_brute_force_unsorted_overlaps_and_filters(self):
        from scipy import ndimage
        from aqua2.viewer_model import color
        rng=np.random.default_rng(62);shape=(11,13,2,5);data=new_project('indexed')
        data['opts']={'sz':shape};data['movies']={'channel1':{'stored':np.zeros(shape,np.uint8)}};events=[]
        for i in range(15):events.append(rng.choice(np.prod(shape),size=150,replace=False))
        cells=np.empty((len(events),1),object)
        for i,e in enumerate(events):cells[i,0]=(e+1.).reshape(-1,1)
        data['stages']={'detection':{'opts':{'sz':shape},'channel1':{'local':{'evt':cells}}}};p=Project(data)
        for all_time in (False,True):
            for visible in (None,[1,3,6,12]):
                for t in (0,2,4):
                    for z in (0,1):
                        labels=np.zeros(shape[:2],np.int32);rgba=np.zeros((*shape[:2],4),np.float32)
                        for i,e in enumerate(events):
                            if visible is not None and i not in visible:continue
                            indices=e%286 if all_time else e[e//286==t]%286;indices=indices[indices//143==z]%143
                            labels[indices%11,indices//11]=i+1
                        for i in np.unique(labels):
                            if i<=0:continue
                            mask=labels==i;rgba[mask,:3]=color(i-1);rgba[mask,3]=1
                            if i-1 in (3,6):
                                edge=mask&~ndimage.binary_erosion(mask,structure=np.ones((3,3),bool));rgba[edge,:3]=[1,.92,.15]
                        actual,hits=layer(p,'Event','channel1','local',t,z,selected=[3,6],all_time=all_time,visible=visible)
                        np.testing.assert_array_equal(hits,labels);np.testing.assert_array_equal(actual,rgba)

    def test_reused_image_axes_and_cursor_blit_handle_full_redraw(self):
        from aqua2.viewer import Workbench
        figure=Figure(figsize=(6,4),dpi=100);canvas=FigureCanvasAgg(figure)
        window=SimpleNamespace(image_figure=figure,image_canvas=canvas,image_blit=BlitCanvas(canvas),display_key=None,draw_masks=lambda ax,z:[])
        project=SimpleNamespace();request=SimpleNamespace(project=project,side=True,y_up=True,view='Overlay',z=0,mask_revision=0,overlay=True)
        images=np.arange(10*12).reshape(10,12)/120;rgba=np.zeros((10,12,4));hits=np.zeros((10,12),int)
        Workbench.draw_image(window,request,images,images,rgba,hits);axes=window.image_axes.copy();artist=window.image_artists[0]
        before=np.asarray(canvas.buffer_rgba()).copy();Workbench.draw_image(window,request,1-images,1-images,rgba,hits)
        self.assertIs(window.image_axes[0],axes[0]);self.assertIs(window.image_artists[0],artist);self.assertFalse(np.array_equal(before,canvas.buffer_rgba()))
        axes[0].set_xlim(2,8);canvas.draw();Workbench.draw_image(window,request,images,images,rgba,hits);self.assertEqual(axes[0].get_xlim(),(2,8))
        self.assertTrue(np.any(np.asarray(canvas.buffer_rgba())[:,:,:3]<40))
        import io
        buffer=io.BytesIO();figure.savefig(buffer,format='png',dpi=150)
        self.assertGreater(len(buffer.getvalue()),1000);self.assertIsNone(window.image_blit.background)
        window.image_blit.paint();self.assertEqual(np.asarray(canvas.buffer_rgba()).shape,(400,600,4))
