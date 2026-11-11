"""GPU geometry and opt-in failure handling, without requiring a GPU driver."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
from aqua2.viewer_gpu import GpuCanvas,image_quad


class GpuDisplay(unittest.TestCase):
    def test_quad_preserves_pixel_edges_orientation_and_viewport(self):
        # 20x10 image on a smaller axis within a 200x100 canvas.
        down=image_quad((20,10,180,90),((-.5,19.5),(9.5,-.5)),(10,20),(200,100))
        np.testing.assert_allclose(down[[0,1,2,5],:2],[[-.8,-.8],[.8,-.8],[.8,.8],[-.8,.8]])
        np.testing.assert_array_equal(down[[0,1,2,5],2:],[[0,1],[1,1],[1,0],[0,0]])
        up=image_quad((20,10,180,90),((-.5,19.5),(-.5,9.5)),(10,20),(200,100))
        np.testing.assert_array_equal(up[[0,2],2:],[[0,0],[1,1]])
        zoom=image_quad((0,0,200,100),((3.5,11.5),(7.5,1.5)),(10,20),(200,100))
        np.testing.assert_allclose(zoom[[0,2],2:],[[.2,.8],[.6,.2]])

    def test_failed_optional_backend_leaves_legacy_renderer_available(self):
        from aqua2.viewer import Workbench
        enabled=Mock();enabled.get.return_value=True
        owner=SimpleNamespace(pause_play=Mock(),gpu_canvas=None,use_gpu=enabled,status=Mock(),
                              image_canvas=Mock(),image_blit=Mock(),request_render=Mock())
        with patch('aqua2.viewer_gpu.GpuCanvas',side_effect=RuntimeError('driver unavailable')):
            Workbench.toggle_gpu(owner)
        self.assertIsNone(owner.gpu_canvas);enabled.set.assert_called_once_with(False)
        owner.image_blit.invalidate.assert_called_once();owner.request_render.assert_called_once()

    def test_partial_initialization_is_cleaned_and_original_error_preserved(self):
        with patch.object(GpuCanvas,'initialize',side_effect=RuntimeError('context unavailable')),patch.object(GpuCanvas,'close') as close:
            with self.assertRaisesRegex(RuntimeError,'context unavailable'):GpuCanvas(Mock(),Mock())
        close.assert_called_once()

    def test_toolbar_home_invalidates_surface_without_advancing_frame(self):
        from matplotlib.figure import Figure
        figure=Figure(figsize=(8,4));axis=figure.subplots()
        owner=SimpleNamespace(display_key=('same-frame',),image_axes=[axis],draw_kind=None,
            navtoolbar=SimpleNamespace(mode=''),view=Mock(),focus_get=Mock(return_value=None),
            request_render=Mock(),playing=False,after=Mock())
        owner.view.get.return_value='Original'
        gpu=GpuCanvas.__new__(GpuCanvas);gpu.owner=owner;gpu.canvas=SimpleNamespace(figure=figure)
        gpu.closed=False;gpu.blocked=False;gpu.window=Mock();gpu.stamp=gpu.layout_stamp()
        gpu.process=Mock();gpu.process.is_alive.return_value=True
        gpu.connection=Mock();gpu.connection.poll.return_value=False;gpu.inputs=[]
        axis.set_xlim(10,20);gpu.pump()
        self.assertIsNone(gpu.stamp);owner.request_render.assert_called_once()
        self.assertEqual(owner.display_key,('same-frame',))

    def test_worker_exit_repaints_compatible_canvas(self):
        gpu=GpuCanvas.__new__(GpuCanvas);gpu.closed=False;gpu.close=Mock()
        gpu.process=Mock();gpu.process.is_alive.return_value=False;gpu.process.exitcode=7
        gpu.owner=SimpleNamespace(use_gpu=Mock(),status=Mock(),image_blit=Mock(),request_render=Mock())
        gpu.pump()
        gpu.close.assert_called_once();gpu.owner.use_gpu.set.assert_called_once_with(False)
        gpu.owner.image_blit.paint.assert_called_once();gpu.owner.request_render.assert_called_once()
        self.assertIn('7',gpu.error)

    def test_shared_display_buffer_is_bounded_and_never_accepts_scientific_floats(self):
        gpu=GpuCanvas.__new__(GpuCanvas);gpu.buffers={};gpu.closed=False
        gpu.process=None;gpu.connection=None;gpu.after_id=None
        source=np.arange(120,dtype=np.uint8).reshape(5,8,3)[:,::2]
        try:
            first=gpu._pixels(1,source);second=gpu._pixels(1,source+1)
            self.assertEqual(first['name'],second['name']);self.assertEqual(len(gpu.buffers),1)
            np.testing.assert_array_equal(gpu.buffers[1][1],source+1)
            with self.assertRaises(ValueError):gpu._pixels(1,source.astype(np.float32))
        finally:gpu.close()

    def test_forwarded_input_uses_parent_canvas_coordinates(self):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        figure=Figure(figsize=(4,3),dpi=100);canvas=FigureCanvasAgg(figure);axis=figure.subplots();canvas.draw()
        events=[];canvas.mpl_connect('button_press_event',events.append)
        gpu=GpuCanvas.__new__(GpuCanvas);gpu.canvas=canvas
        gpu.owner=SimpleNamespace(step=Mock(),toggle_play=Mock(),cancel_draw=Mock())
        x,y=axis.transData.transform((.25,.5))
        gpu.inputs=[dict(kind='mouse',name='button_press_event',x=x,y=300-y,button=1),dict(kind='key',key='right')]
        gpu._dispatch_inputs()
        self.assertAlmostEqual(events[0].xdata,.25);self.assertAlmostEqual(events[0].ydata,.5)
        gpu.owner.step.assert_called_once_with(1);self.assertEqual(gpu.inputs,[])


if __name__=='__main__':unittest.main()
