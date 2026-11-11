# AQuA2 Portable

**AQuA2-portable (based on Python) is out now!! [Try it out!](https://github.com/yu-lab-vt/AQuA2-portable/releases/tag/v0.1.0-rc1)**

This is an **unofficial preview (v0.1.0-rc1)**. Bugs and differences from MATLAB may remain; please check important results and report issues.

A standalone Python port of [AQuA2](https://github.com/yu-lab-vt/AQuA2) for spatiotemporal fluorescence event detection, feature analysis and CFUs.

This repository contains the Python implementation, required native C++ kernels and numeric resource tables. It contains no MATLAB application source and requires no MATLAB installation or runtime.

## Windows portable preview

1. Download the Windows x64 ZIP from [Releases](https://github.com/yu-lab-vt/AQuA2-portable/releases/tag/v0.1.0-rc1).
2. Extract the **entire folder** to a writable local directory.
3. Run `AQuA2.exe`. Keep `_internal` beside it; the exe is not a single-file distribution.

Python is bundled. OpenGL 3.3 and a working graphics driver enable GPU display; compatible 2D/Z-slice display remains available. No CUDA toolkit is required. First analysis can take longer while Numba compiles kernels; its cache and small diagnostic logs are under `%LOCALAPPDATA%/AQuA2-portable`.

## Workflow

Load Image -> Preprocessing -> Event Detection -> CFU, or use Run All. Load Parameters applies a CSV preset to all stages. TIFF, numeric MAT and supported video inputs are accepted; channels are detected from format metadata. Saved MAT results can be imported separately. RGB event-overlay movies are not grayscale analysis inputs.

Results stay in memory. Use Save / Save As explicitly to preserve a project; analysis does not automatically save. Export event/CFU tables, movies with overlays, overlay-only TIFFs and rising maps as needed. Tables include favorite flags. Rising maps use a `rising_map` subfolder; CFU image output is a mean projection. Result export does not create MAT files.

3D defaults to Gradient Opacity volume rendering. Drag to rotate, right-drag to pan, wheel to zoom and double-click to reset. Both panels share a camera. Z Slice and mesh modes remain available. Set Z Scale from the actual Z/XY acquisition spacing. Export View captures the current GPU view. This renderer is a documented approximation, not a pixel-identical reproduction of MATLAB's proprietary shader.

## Build from source

Use Python 3.12 x64 on Windows with MSVC C++ build tools:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-runtime.lock -r requirements-build.lock
.venv/Scripts/python.exe -m pip install -e . --no-build-isolation --no-deps
.venv/Scripts/python.exe -m PyInstaller --noconfirm --clean packaging/AQuA2.spec
```

The output is `dist/AQuA2-portable/AQuA2.exe`. Tests use `python -m unittest discover -s tests`; local MATLAB-reference tests skip when optional external reference fixtures are absent. Experimental datasets and historical reference outputs are not distributed.

## Status and provenance

Version **0.1.0-rc1** is a release candidate. Windows packaging validation and known limits are recorded in `docs/VALIDATION.md`; `PROVENANCE.json` identifies the upstream source snapshot. Independent 3D/multichannel/registration/bleaching scientific reference coverage is still incomplete. Preserve upstream attribution and cite the original AQuA2 work described in the upstream repository.

## License

This portable project is released under **GPL-3.0-only**, with explicit authorization from the upstream laboratory maintainer. See LICENSE. Original GPLv2 texts are retained for attribution/provenance in `licenses/UPSTREAM-GPL-2.0.txt` and `native/vendor/LICENSE`; third-party packages retain their own licenses. See `THIRD_PARTY_NOTICES.md` and `licenses/`.
