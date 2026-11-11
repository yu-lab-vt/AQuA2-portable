"""Build two isolated native libraries; the upstream kernels share globals."""
import sys
from setuptools import Extension,setup

flags=["/std:c++17","/fp:precise"] if sys.platform=="win32" else ["-std=c++17","-ffp-contract=off"]
setup(ext_modules=[Extension("aqua2._"+name+"_core",["native/bridge.cpp"],
    include_dirs=["native"],define_macros=[("AQUA_BILCO_CORE","1")] if name=="bilco" else [],
    language="c++",extra_compile_args=flags) for name in ("dtw","bilco")])
