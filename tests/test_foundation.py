import tempfile
import unittest
from pathlib import Path

import numpy as np

from aqua2.compat import from_matlab_indices, to_matlab_indices, ind2sub, sub2ind
from aqua2.config import read_preset
from aqua2.compare import Tolerance, compare, compare_stages
from aqua2.io import read_mat, write_mat, read_tiff_hwlt, MatStructArray
from aqua2.model import StageResult

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "reference/local"


class FoundationTests(unittest.TestCase):
    def test_column_major_indexing(self):
        indices = np.array([1, 3, 4, 12, 48])
        shape = (3, 4, 1, 4)
        subs = ind2sub(shape, indices)
        np.testing.assert_array_equal(subs[0], [0, 2, 0, 2, 2])
        np.testing.assert_array_equal(subs[1], [0, 0, 1, 3, 3])
        np.testing.assert_array_equal(sub2ind(shape, *subs), indices)
        np.testing.assert_array_equal(to_matlab_indices(from_matlab_indices(indices, 48), 48), indices)
        np.testing.assert_array_equal(from_matlab_indices(indices.astype(np.float64), 48), indices - 1)

    def test_invalid_indices(self):
        for indices in ([0], [49]):
            with self.assertRaises(IndexError):
                from_matlab_indices(indices, 48)
        with self.assertRaises(TypeError):
            from_matlab_indices([1.5], 48)

    def test_preset(self):
        opts = read_preset(ROOT / "src/aqua2/data/parameters_for_batch.csv")
        self.assertEqual(opts["movAvgWin"], 25)
        self.assertTrue(np.isposinf(opts["maxSize"]))
        with self.assertRaises(ValueError):
            read_preset(ROOT / "src/aqua2/data/parameters_for_batch.csv", 999)

    def test_shape_contract(self):
        StageResult("input", (3, 4, 1, 20))
        with self.assertRaises(ValueError):
            StageResult("input", (3, 4, 20))

    def test_exact_and_special_values(self):
        data = np.array([1, np.nan, np.inf, -np.inf], dtype=np.float32)
        self.assertTrue(compare(data, data.copy())["passed"])
        changed = data.copy()
        changed[-1] = np.inf
        self.assertEqual(compare(data, changed)["first_difference"]["reason"], "nan_or_inf_distribution")

    def test_error_metrics_and_explicit_tolerance(self):
        a = np.ones((2, 3), dtype=np.float64)
        b = a.copy()
        b[1, 2] += 0.125
        report = compare({"dF": a}, {"dF": b})
        self.assertFalse(report["passed"])
        self.assertEqual(report["first_difference"]["field"], "root/dF")
        self.assertEqual(report["first_difference"]["first_index"], [1, 2])
        self.assertEqual(report["first_difference"]["max_abs_error"], 0.125)
        self.assertTrue(compare(a, b, tolerance=Tolerance(atol=0.125))["passed"])

    def test_discrete_values_never_use_tolerance(self):
        self.assertFalse(compare(np.array([1]), np.array([2]), tolerance=Tolerance(atol=100))["passed"])

    def test_shape_dtype_and_missing(self):
        self.assertFalse(compare(np.ones((2, 1)), np.ones((2,)))["passed"])
        self.assertFalse(compare(np.ones(2, dtype="float32"), np.ones(2, dtype="float64"))["passed"])
        self.assertFalse(compare({"dF": np.ones(2)}, {})["passed"])

    def test_first_differing_stage(self):
        reference = {"input": [1.0], "noise": [2.0], "events": [3]}
        actual = {"input": [1.0], "noise": [2.1], "events": [4]}
        self.assertEqual(compare_stages(reference, actual)["first_differing_stage"], "noise")

    def test_invalid_tolerance(self):
        for value in (-1, np.inf, np.nan):
            with self.assertRaises(ValueError):
                Tolerance(atol=value)

    def test_v7_roundtrip(self):
        data = {"movie": np.arange(48, dtype="float32").reshape(3, 4, 1, 4),
                "mask": np.array([[True, False]]),
                "record": {"id": np.array([[1]], dtype="uint32")}}
        with tempfile.TemporaryDirectory() as temp:
            first, second = Path(temp) / "first.mat", Path(temp) / "second.mat"
            write_mat(first, data)
            loaded = read_mat(first)
            self.assertTrue(compare(data, loaded)["passed"])
            write_mat(second, loaded)
            self.assertTrue(compare(loaded, read_mat(second))["passed"])

    def test_cells_containing_structs_stay_cells(self):
        cells = np.empty((1, 2), dtype=object)
        cells[0, 0], cells[0, 1] = {"id": np.array([[1]])}, {"id": np.array([[2]])}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cell.mat"
            write_mat(path, {"cells": cells})
            loaded = read_mat(path)["cells"]
            self.assertNotIsInstance(loaded, MatStructArray)
            self.assertIsInstance(loaded[0, 0], dict)
            self.assertTrue(compare(cells, loaded)["passed"])

    def test_empty_struct_array_preserves_fields(self):
        records = np.empty((0, 1), dtype=object).view(MatStructArray)
        records.mat_fields = ("id", "curve")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "empty-struct.mat"
            write_mat(path, {"records": records})
            loaded = read_mat(path)["records"]
            self.assertIsInstance(loaded, MatStructArray)
            self.assertEqual(loaded.mat_fields, records.mat_fields)
            self.assertTrue(compare(records, loaded)["passed"])

    def test_cli_detects_error(self):
        import subprocess
        import sys
        import json
        with tempfile.TemporaryDirectory() as temp:
            a, b, report = (Path(temp) / name for name in ("a.mat", "b.mat", "report.json"))
            write_mat(a, {"dF": np.array([[1.0, 2.0]])})
            write_mat(b, {"dF": np.array([[1.0, 2.1]])})
            run = subprocess.run([sys.executable, "-m", "aqua2", "compare", str(a), str(b),
                                  "--output", str(report)], capture_output=True)
            self.assertEqual(run.returncode, 1, run.stderr)
            self.assertEqual(json.loads(report.read_text())["first_difference"]["field"], "root/dF")

    def test_tiff_axes(self):
        import tifffile
        stack = np.arange(3 * 4 * 5, dtype="uint16").reshape(3, 4, 5)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "movie.tif"
            tifffile.imwrite(path, stack, metadata={"axes": "TYX"}, photometric="minisblack")
            result = read_tiff_hwlt(path)
            self.assertEqual(result.shape, (4, 5, 1, 3))
            np.testing.assert_array_equal(result[:, :, 0, 2], stack[2])

    def test_tiff_z_stack_defaults_to_time(self):
        import tifffile
        from aqua2.pipeline import _read_raw
        stack = np.arange(7 * 6 * 5, dtype="uint8").reshape(7, 6, 5)
        with tempfile.TemporaryDirectory() as temp:
            for imagej in (False, True):
                with self.subTest(imagej=imagej):
                    path = Path(temp) / "z-stack.tif"
                    tifffile.imwrite(path, stack, imagej=imagej,
                                     metadata={"axes": "ZYX"}, photometric="minisblack")
                    with tifffile.TiffFile(path) as image:
                        self.assertEqual(image.series[0].axes, "ZYX")
                    result = read_tiff_hwlt(path)
                    self.assertEqual(result.shape, (6, 5, 1, 7))
                    self.assertEqual(result.dtype, stack.dtype)
                    np.testing.assert_array_equal(result[:, :, 0, :], stack.transpose(1, 2, 0))
                    raw, depth = _read_raw(path)
                    self.assertEqual(depth, 8)
                    self.assertEqual(raw.dtype, np.float32)
                    np.testing.assert_array_equal(raw, result)

    def test_tiff_explicit_volume_keeps_z_and_time(self):
        import tifffile
        from aqua2.pipeline import _read_raw
        volume = np.arange(3 * 2 * 6 * 5, dtype="uint16").reshape(3, 2, 6, 5)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "volume.tif"
            tifffile.imwrite(path, volume, metadata={"axes": "TZYX"}, photometric="minisblack")
            raw, depth = _read_raw(path)
            self.assertEqual(raw.shape, (6, 5, 2, 3))
            self.assertEqual(depth, 16)
            np.testing.assert_array_equal(raw, volume.transpose(2, 3, 1, 0))

    def test_tiff_rgb_remains_rejected(self):
        import tifffile
        from aqua2.pipeline import _read_raw
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "overlay.tif"
            tifffile.imwrite(path, np.zeros((6, 5, 3), dtype="uint8"), photometric="rgb")
            with self.assertRaisesRegex(ValueError, "Unsupported TIFF axes"):
                _read_raw(path)

    def test_unsupported_matlab_object_rejected(self):
        import h5py
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "object.mat"
            with h5py.File(path, "w") as file:
                group = file.create_group("viewer")
                group.attrs["MATLAB_class"] = np.bytes_("object")
            with self.assertRaises(TypeError):
                read_mat(path)

    @unittest.skipUnless((LOCAL / "preprocessing.mat").exists(), "Generate MATLAB preset oracle first")
    def test_preset_matches_matlab(self):
        # This reference intentionally reuses historical public values; check
        # unmodified preset parameters and acknowledge the known overrides.
        matlab = read_mat(LOCAL / "preprocessing.mat", ["opts"])["opts"]
        for key, value in read_preset(ROOT / "src/aqua2/data/parameters_for_batch.csv").items():
            if key not in ("cut", "propMetric", "networkFeatures"):
                self.assertEqual(value, matlab[key].item(), key)

    @unittest.skipUnless((LOCAL / "interop-v73.mat").exists(), "Generate local MATLAB fixtures first")
    def test_matlab_v7_v73_interop(self):
        reference, actual = read_mat(LOCAL / "interop-v7.mat"), read_mat(LOCAL / "interop-v73.mat")
        report = compare(reference, actual)
        self.assertTrue(report["passed"], report)
        self.assertEqual(actual["movie"].shape, (3, 4, 1, 4))
        self.assertEqual(actual["empty"].shape, (0, 3))
        write_mat(LOCAL / "interop-python.mat", actual)

    @unittest.skipUnless((LOCAL / "preprocessing.mat").exists(), "Generate local MATLAB fixtures first")
    def test_real_data_repeatability(self):
        first = read_mat(LOCAL / "preprocessing.mat")
        second = read_mat(LOCAL / "preprocessing-repeat.mat")
        self.assertTrue(compare(first["dF"], second["dFRepeat"])["passed"])
        self.assertTrue(compare(first["optsOut"], second["optsRepeat"])["passed"])
        self.assertTrue(compare(first["activeRegions"], second["activeRepeat"])["passed"])
        write_mat(LOCAL / "preprocessing-python.mat", first)
        self.assertTrue(compare(first, read_mat(LOCAL / "preprocessing-python.mat"))["passed"])

    @unittest.skipUnless((LOCAL / "pipeline.mat").exists(), "Generate pipeline oracle first")
    def test_pipeline_exchange(self):
        original = read_mat(LOCAL / "pipeline.mat")
        write_mat(LOCAL / "pipeline-python.mat", original)
        self.assertTrue(compare(original, read_mat(LOCAL / "pipeline-python.mat"))["passed"])


if __name__ == "__main__":
    unittest.main()
