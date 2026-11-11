import unittest
from dataclasses import replace
import numpy as np
from aqua2.viewer_display import DisplaySettings
from aqua2.viewer_render import FrameRequest,prepare_frame
from aqua2.viewer_volume import display_volume,prepare_volume,VolumeScene
from test_result_export import sample


class VolumeTests(unittest.TestCase):
    def request(self,p):
        return FrameRequest(p,0,0,'channel1','local','Event','Overlay',(),(0,),False,1,.5,True,True,False,0,display=DisplaySettings(),volume=True,volume_style='gradient')

    def test_float_mapping_retains_sub_byte_structure(self):
        a=np.array([0.,.0001,.0002,1.],np.float32)
        np.testing.assert_array_equal(display_volume(a,(0,1),DisplaySettings()),a)
        self.assertEqual(display_volume(a,(0,1),DisplaySettings()).dtype,np.float32)
        self.assertGreater(display_volume(a,(0,1),DisplaySettings(brightness=.8))[1],a[1])

    def test_range_and_nonfinite_data(self):
        a=np.array([0,.25,.5,.75,1,np.nan],np.float32)
        np.testing.assert_array_equal(display_volume(a,(0,1),DisplaySettings(minimum=.25,maximum=.75)),[0,0,.5,1,1,0])

    def test_transparency_changes_alpha_without_changing_intensity(self):
        p=sample();req=self.request(p);one=prepare_volume(req)[0]
        two=prepare_volume(replace(req,display=DisplaySettings(transparency=.8)))[0]
        np.testing.assert_array_equal(one.volumes[0],two.volumes[0]);self.assertAlmostEqual(two.opacity,.2)

    def test_overlay_labels_selection_and_original_view(self):
        p=sample();req=replace(self.request(p),selected=(0,));scene=prepare_volume(req)[0]
        self.assertTrue(np.any(scene.overlay[:,:,:,3]>0));self.assertTrue(np.all(scene.overlay[:,:,:,3]<=1))
        coords=np.where(scene.overlay[:,:,:,3]>0);np.testing.assert_allclose(scene.overlay[coords][:,:3],np.tile([1,.92,.15],(len(coords[0]),1)))
        raw=prepare_volume(replace(req,view='Original'))[0];self.assertFalse(raw.overlay.any())

    def test_scene_routing_raw_mean_and_anisotropy_preserve_input(self):
        p=sample();before=p.movie().copy();req=replace(self.request(p),view='Mean Projection',z_scale=3,gradient_opacity=.6)
        scene=prepare_frame(req)[0];self.assertIsInstance(scene,VolumeScene)
        self.assertFalse(np.array_equal(*scene.volumes));self.assertEqual(scene.z_scale,3);self.assertEqual(scene.gradient,.6)
        np.testing.assert_array_equal(p.movie(),before)

    def test_region_and_landmark_overlay_coordinates(self):
        p=sample();p.data['annotations'].update(regions=[np.array([0])],landmarks=[np.array([1])])
        scene=prepare_volume(self.request(p))[0]
        np.testing.assert_allclose(scene.overlay[0,0,0],[0,.85,.9,.25]);np.testing.assert_allclose(scene.overlay[1,0,0],[1,.8,.2,.25])
