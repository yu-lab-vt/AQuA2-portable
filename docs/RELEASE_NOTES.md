# AQuA2 Portable v0.1.0-rc1

**AQuA2-portable (based on Python) is out now!! Try it out!**

This is an **unofficial preview**, not a final stable release. Bugs and differences from MATLAB may remain. Please check important results and [report issues](https://github.com/yu-lab-vt/AQuA2-portable/issues).

## Get started

Download `AQuA2-portable-v0.1.0-rc1-windows-x64.zip`, extract the **entire folder**, and run `AQuA2.exe`. Keep `_internal` beside it. No MATLAB, MATLAB Runtime or separate Python installation is required. First analysis may take longer while kernels compile.

## What's included

- Preprocessing, event detection, event features and CFU analysis, with stage tabs and Run All.
- English UI, parameter presets, polygon regions/landmarks and favorite markers.
- GPU display and linked 3D volume views with rotation, pan and zoom.
- Manual project saving and configurable tables, movies, masks, rising maps and view exports.

## Checks and known limits

Source tests: 149 passed; 37 optional external-reference tests skipped. Real-recording checks matched source/executable event and dF/F fingerprints, project save/reopen and 116 CFU exports. 2D/3D display and relocated-folder checks passed.

The executable is unsigned and has not been tested on a separate clean Windows machine. Independent coverage of all 3D/multichannel/registration/bleaching options is incomplete. Gradient Opacity approximates MATLAB's rendering; frame rate depends on hardware and data. See [validation](https://github.com/yu-lab-vt/AQuA2-portable/blob/main/docs/VALIDATION.md).

## Downloads

- `AQuA2-portable-v0.1.0-rc1-windows-x64.zip` — portable application.
- `AQuA2-portable-v0.1.0-rc1-source.zip` — application and build sources.
- `SHA256SUMS.txt` — download checksums.

GPL-3.0-only. Upstream and dependency notices are included. No experimental data or result files are distributed.
