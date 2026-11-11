# AQuA2 Portable

A standalone Python port of [AQuA2](https://github.com/yu-lab-vt/AQuA2) for spatiotemporal fluorescence event detection, feature analysis and CFUs.

This is a **preview (v0.1.0-rc1)**. Bugs may remain, so please report issues when something doesn't turn out right.

This repository contains the Python implementation, required native C++ kernels and numeric resource tables. It contains no MATLAB application source and requires no MATLAB installation or runtime.

## Windows portable preview

1. Download the Windows x64 ZIP from [Releases](https://github.com/yu-lab-vt/AQuA2-portable/releases/tag/v0.1.0-rc1).
2. Extract the **entire folder** to a writable local directory.
3. Run `AQuA2.exe`.

## Workflow

Load Image -> Preprocessing -> Event Detection -> CFU, or use Run All.

**Input**: TIFF(for 2D+time), numeric MAT(3D+time) are accepted.
Load Parameters to change parameters of all stages according to a CSV.

**Output**: Use Save / Save As explicitly to preserve a project in H5. Event/CFU tables, movies with overlay etc. can be exported separately.

## Status and provenance

Version **0.1.0-rc1** is a release candidate. Windows packaging validation and known limits are recorded in `docs/VALIDATION.md`. Independent 3D/multichannel/registration/bleaching scientific reference coverage is still incomplete. 

## License

This portable project is released under GPL-3.0.
