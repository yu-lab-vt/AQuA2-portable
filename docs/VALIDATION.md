# Validation — v0.1.0-rc1

Validated locally on Windows x64 with NVIDIA RTX 4070 SUPER. No experimental dataset or result output is shipped.

- Independent source tests: **186 run, 149 passed, 37 skipped**, no failures/errors. Skips require optional external MATLAB-reference fixtures that are not distributed.
- Frozen exe ran with Python/MATLAB removed from child PATH and development environment variables unset.
- A fresh relocated folder containing spaces and non-ASCII characters worked; the exe reported frozen=True and a real nopython JIT signature. **23 new compiled Numba cache files** were generated in a fresh assigned cache directory. No compiled cache is shipped.
- A real 100-frame recording: preprocessing -> event detection -> features -> CFU worker completed; source and exe produced identical event-index and dF/F fingerprints (**14 events**, 0 CFUs on this small default-parameter fixture).
- Save/reopen fingerprints matched; result export completed without MAT output.
- Actual Tk/OpenGL windows displayed the processed and raw 2D recording and the **78 x 78 x 9 x 1000** 3D volume. Rotation changed the rendered image; the GPU PID differed from the main PID and no QtCore/QtGui/QtWidgets module was loaded in the main process.
- A separate real saved-result fixture exported **116 CFU rows**, exactly matching source CSV bytes; a mean-projection CFU preview exactly matched source pixels.
- A generated lossless grayscale AVI was read by the frozen executable with its bundled FFmpeg.

The first complete cold-cache verification took approximately **60.5 seconds** on this workstation, including compilation, analysis and window tests. This is not a general runtime benchmark.

Packaging issues found and fixed: Conda ICU shadowing Windows ICU; PyInstaller eagerly importing QtCore into Tk; application source paths needed by Numba caching. All final checks above use the corrected package.

Limits: no separate clean Windows VM was available; isolation/relocation was tested on this host. The executable is unsigned. GPU drivers/Windows inbox libraries are system prerequisites. Independent real 3D/multichannel/registration/bleaching scientific reference coverage remains incomplete. The gradient-opacity shader is an approximation, and no 60/100 fps guarantee is made.
