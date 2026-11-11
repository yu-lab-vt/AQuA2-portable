# Third-party notices

This port derives from [AQuA2](https://github.com/yu-lab-vt/AQuA2). The portable project uses GPL-3.0-only with the upstream laboratory maintainer's explicit authorization. Original GPLv2 notices remain in licenses/UPSTREAM-GPL-2.0.txt. The DTW/BILCO C++ kernels are included with their original license text. The `mex.h` compatibility shim removes the MATLAB runtime dependency; it is not a MATLAB binary or SDK.

The portable distribution includes CPython, NumPy, SciPy, Numba/llvmlite, h5py, tifffile, scikit-image, Matplotlib/Pillow, ImageIO/FFmpeg, PyQt6/Qt6 and PyOpenGL, plus their dependencies. Exact package versions, declared licenses and bundled license files are generated in `licenses/dependency-inventory.json` and the per-package license folders during the build. FFmpeg notices refer to the bundled imageio-ffmpeg binary.

PyQt6 is GPLv3/commercial; Qt and other dependencies retain their own notices. The upstream maintainer authorized this portable project to use GPLv3. Third-party copyrights and licenses are not changed. The source/native kernel build files and locked dependency versions accompany the binary release.
