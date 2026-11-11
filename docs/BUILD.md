# Build and verification

Use Python 3.12 x64 on Windows and MSVC C++ build tools. Install the two requirement lock files, build the native extension with `pip install -e . --no-build-isolation --no-deps`, then run PyInstaller with `packaging/AQuA2.spec`. See the commands in [README](../README.md).

Run `packaging/collect_notices.py` to retain dependency notices. The bundle must include LICENSE, THIRD_PARTY_NOTICES.md, README.md and licenses/. Keep the whole application folder together.

## Packaging constraints

- Call `multiprocessing.freeze_support()` before application imports; analysis and GPU display spawn separate processes.
- Preserve the path-only Qt runtime hook. The default PyInstaller Qt hook imports QtCore into the Tk parent and breaks process isolation.
- Exclude Conda ICU libraries so Qt resolves Windows inbox ICU. Never replace system DLLs.
- Collect application modules as `.py` files for Numba. Exclude `__pycache__`, `.pyc`, `.nbc` and `.nbi` from release archives.
- Keep required numeric `.mat` resource tables and native C++ sources. No MATLAB installation is needed.

## Verify a build

Run `python -m unittest discover -s tests`. Optional external-reference tests skip without their fixtures.

`AQuA2.exe --verify config.json` accepts `output`, `tiff`, `volume`, `analysis` and `require_gpu` for local validation. Keep configs, recordings, results and logs outside release assets. Test a fresh cache and relocated path with spaces/non-ASCII characters, with development Python/MATLAB removed from the child PATH. Check actual nopython compilation and a separate GPU PID.

Create source and portable ZIPs, test their CRCs, and generate SHA256SUMS. Never distribute experimental data, local paths, credentials, caches or build logs. Preserve upstream and dependency notices. Scientific and display limits are listed in [VALIDATION.md](VALIDATION.md).
