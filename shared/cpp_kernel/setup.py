from __future__ import annotations

import sys
from setuptools import setup

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    HAVE_PYBIND11 = True
except ImportError:
    HAVE_PYBIND11 = False

if HAVE_PYBIND11:
    ext_modules = [
        Pybind11Extension(
            "pricing_kernel",
            ["bindings/pybind_module.cpp"],
            include_dirs=["include"],
            cxx_std=17,
        ),
    ]
    cmdclass = {"build_ext": build_ext}
else:
    ext_modules = []
    cmdclass = {}
    sys.stderr.write("pybind11 not installed; skipping C++ extension build.\n")

setup(
    name="pricing_kernel",
    version="0.1.0",
    description="C++ Black-Scholes pricing kernel for QIS platform",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    zip_safe=False,
    python_requires=">=3.9",
)
