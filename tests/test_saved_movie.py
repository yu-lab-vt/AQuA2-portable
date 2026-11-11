import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from aqua2.io import read_saved_dat_org


class SavedMovieTests(unittest.TestCase):
    def test_axes_slice_saved_scale_and_no_second_crop(self):
        movie = (np.arange(60).reshape(3, 4, 1, 5) + 200).astype("uint16")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "saved.mat"
            with h5py.File(path, "w") as root:
                root.create_dataset("res/datOrg1", data=movie.transpose(3, 2, 1, 0))
                root.create_dataset("res/opts/sz", data=np.array(movie.shape).reshape(4, 1))
                root.create_dataset("res/opts/minValueDat1", data=[[109]])
                root.create_dataset("res/opts/maxValueDat1", data=[[860]])
                root.create_dataset("res/opts/regMaskGap", data=[[1]])
            stored = read_saved_dat_org(path, frames=slice(1, 4))
            np.testing.assert_array_equal(stored, movie[:, :, :, 1:4])
            self.assertEqual(stored.shape, (3, 4, 1, 3))
            normalized = read_saved_dat_org(path, frames=slice(1, 4), normalized=True)
            np.testing.assert_array_equal(normalized, (stored.astype("float32") - 109) / 751)
            self.assertEqual(normalized.dtype, np.float32)
