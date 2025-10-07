#!/usr/bin/env python3
"""
Setup script for building Cython extensions for Flashback HFT backtesting engine.

This script builds the high-performance Cython extensions used for order matching
and strategy calculations.
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

# Check if Cython is available
try:
    from Cython.Build import cythonize
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False
    print("Warning: Cython not available. Cython extensions will not be built.")

# Define extensions
extensions = []

if HAS_CYTHON:
    # Cython matching engine extension
    match_extension = Extension(
        "flashback.market._match",
        sources=["flashback/market/_match.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-ffast-math"],
        extra_link_args=["-O3"],
        language="c++"
    )
    extensions.append(match_extension)

def build_cython_extensions():
    """Build Cython extensions if available."""
    if not HAS_CYTHON:
        print("Cython not available. Skipping Cython extension build.")
        return []
    
    print("Building Cython extensions...")
    return cythonize(extensions, compiler_directives={
        'boundscheck': False,
        'wraparound': False,
        'cdivision': True,
        'language_level': 3
    })

if __name__ == "__main__":
    if HAS_CYTHON:
        setup(
            name="flashback-cython",
            ext_modules=build_cython_extensions(),
            zip_safe=False,
        )
    else:
        print("Cython not available. Please install Cython to build extensions:")
        print("pip install Cython")
