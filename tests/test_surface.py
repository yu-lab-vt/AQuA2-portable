import unittest
from dataclasses import replace
import numpy as np
from aqua2.viewer_surface import surface_triangles,prepare_surface
from aqua2.viewer_render import FrameRequest
from aqua2.viewer_display import DisplaySettings
from test_result_export import sample


class SurfaceTests(unittest.TestCase):
    def test_closed_boundary_object_xyz_and_spacing(self):
        a=np.ones((4,6,3),np.float32);before=a.copy()
        mesh=surface_triangles(a,.5,spacing=(1,1,2))
        self.assertEqual(mesh.shape[1],11);self.assertEqual(len(mesh)%3,0)
        np.testing.assert_allclose(mesh[:,:3].min(axis=0),[-.5,-.5,-1])
        np.testing.assert_allclose(mesh[:,:3].max(axis=0),[5.5,3.5,5])
        np.testing.assert_array_equal(a,before);self.assertTrue(np.isfinite(mesh).all())

    def test_empty_or_nonfinite_volume(self):
        for a in (np.zeros((3,4,2)),np.full((3,4,2),np.nan)):
            self.assertEqual(surface_triangles(a,.5).shape,(0,11))

    def test_overlay_mesh_and_transparency_leave_scientific_movie_unchanged(self):
        p=sample();before=p.movie().copy()
        req=FrameRequest(p,0,0,'channel1','local','Event','Overlay',(),(0,),False,1,.5,True,True,False,0,display=DisplaySettings(transparency=.7),volume=True)
        scene=prepare_surface(req)[0];self.assertTrue(scene.side)
        self.assertTrue(np.any(scene.vertices[:,10]==1));self.assertTrue(np.any(scene.vertices[:,10]==0))
        np.testing.assert_allclose(scene.vertices[scene.vertices[:,10]==0,9],.3)
        raw=prepare_surface(replace(req,view='Original'))[0];self.assertFalse(np.any(raw.vertices[:,10]==1))
        np.testing.assert_array_equal(before,p.movie())

    def test_comparison_keeps_raw_and_processed_surfaces_separate(self):
        p=sample();req=FrameRequest(p,0,0,'channel1','local','Event','Mean Projection',(),(),False,1,.5,False,True,False,0,display=DisplaySettings(),volume=True)
        scene=prepare_surface(req)[0]
        self.assertEqual(set(np.unique(scene.vertices[:,10])),{-1.,1.})
