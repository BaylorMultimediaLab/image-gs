"""
Build of the C++ modules of compress/: the rANS coder of compress/rans_codec.py (Python module igs_rans) and the feature prediction of
compress/prediction.py (igs_prediction). From the repository root:
    pip install compress/csrc --no-build-isolation
The build uses the pybind11 headers bundled with PyTorch (hence --no-build-isolation); the modules do not link against PyTorch.
"""
import hashlib
import os

from setuptools import Extension, setup
from torch.utils.cpp_extension import include_paths


def source_hash(source):
    """
    Hash of a source file, which the module stores: compress/common.load_extension refuses a module built from another source.
    """
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), source), "rb") as file:
        return hashlib.sha256(file.read()).hexdigest()[:16]


setup(
    name="igs-rans",
    version="1.1",
    ext_modules=[Extension(module, sources=[source], include_dirs=include_paths()[:1], language="c++",
                           extra_compile_args=["-O3", "-std=c++17", "-pthread", "-ffp-contract=off", f'-DSOURCE_HASH="{source_hash(source)}"'],
                           extra_link_args=["-pthread"])
                 for module, source in [("igs_rans", "rans.cpp"), ("igs_prediction", "prediction.cpp")]],
)
