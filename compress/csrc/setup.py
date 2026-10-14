"""
Build of the C++ rANS coder of compress/rans_codec.py (Python module igs_rans). From the repository root:
    pip install compress/csrc --no-build-isolation
The build uses the pybind11 headers bundled with PyTorch (hence --no-build-isolation); the module does not link against PyTorch.
"""
import hashlib
import os

from setuptools import Extension, setup
from torch.utils.cpp_extension import include_paths

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rans.cpp"), "rb") as file:
    SOURCE_HASH = hashlib.sha256(file.read()).hexdigest()[:16]  # compress/rans_codec.py refuses a module built from another source

setup(
    name="igs-rans",
    version="1.0",
    ext_modules=[Extension("igs_rans", sources=["rans.cpp"], include_dirs=include_paths()[:1], language="c++",
                           extra_compile_args=["-O3", "-std=c++17", "-pthread", "-ffp-contract=off", f'-DSOURCE_HASH="{SOURCE_HASH}"'],
                           extra_link_args=["-pthread"])],
)
